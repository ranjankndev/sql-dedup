#!/usr/bin/env python3
"""
sqlpattern - detect and drop duplicate NL->SQL pattern pairs.

The core idea: two (natural-language question, SQL) pairs are the "same kind"
when their SQL reduces to the same structural fingerprint. Fingerprints are
computed at one of three levels:

  exact    - only whitespace/case/comment normalization.
             SELECT * FROM users  ==  select *  from users
  template - literals are masked, IN-lists collapsed. Identifiers kept.
             WHERE age > 21  ==  WHERE age > 99
  skeleton - additionally all identifiers (tables, columns, aliases) are
             renamed consistently by order of first appearance.
             SELECT name FROM users WHERE age > 21
             ==  SELECT title FROM books WHERE price > 5
             (default: this is what "same type of SQL" usually means for
              SFT/RL data diversity)

Commands:
  dedupe       drop duplicate pairs from a dataset, write unique pairs + report
  build-index  build a pattern index from a dataset
  check        check new pair(s) against an index: NEW or DUPLICATE
  stats        show pattern-frequency distribution of a dataset

Input formats: .jsonl (one JSON object per line), .json (array), .csv.
Field names are auto-detected (question/nl/instruction/input + sql/query/
output/answer) or set explicitly with --nl-field / --sql-field.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import sqlglot
    from sqlglot import exp

    HAVE_SQLGLOT = True
except ImportError:  # degraded but functional
    HAVE_SQLGLOT = False

LEVELS = ("exact", "template", "skeleton")

# ---------------------------------------------------------------------------
# SQL fingerprinting
# ---------------------------------------------------------------------------

SQL_KEYWORDS = frozenset(
    """
    select from where group by having order limit offset join inner left right
    full outer cross on as and or not in is null like between exists union all
    distinct case when then else end insert into values update set delete
    create table view index drop alter add primary key foreign references
    asc desc with recursive over partition rows range unbounded preceding
    following current row using natural intersect except cast interval
    true false unknown top fetch first next only escape ilike glob regexp
    """.split()
)

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_PLACEHOLDER_LIST_RE = re.compile(r"\?(?:\s*,\s*\?)+")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*|\?|<>|<=|>=|!=|\|\||[^\sA-Za-z0-9_]")


def _regex_normalize(sql: str, level: str) -> str:
    """Fallback normalization when sqlglot can't parse the query."""
    s = _COMMENT_RE.sub(" ", sql)
    if level != "exact":
        s = _STRING_RE.sub("?", s)
        s = _NUMBER_RE.sub("?", s)
        s = _PLACEHOLDER_LIST_RE.sub("?", s)  # IN (?, ?, ?) -> IN (?)
    s = s.lower()
    if level == "skeleton":
        # rename every non-keyword word token consistently: x1, x2, ...
        mapping = {}
        out = []
        for tok in _TOKEN_RE.findall(s):
            if tok in SQL_KEYWORDS or not tok[0].isalpha() and tok[0] != "_":
                out.append(tok)
            elif tok[0].isalpha() or tok[0] == "_":
                if tok not in mapping:
                    mapping[tok] = f"x{len(mapping) + 1}"
                out.append(mapping[tok])
            else:
                out.append(tok)
        s = " ".join(out)
    return _WS_RE.sub(" ", s).strip()


def _mask_literals(node):
    """AST transform: literals -> ?, collapse IN-lists and VALUES rows."""
    if isinstance(node, exp.In):
        # IN (1, 2, 3) -> IN (?) so list length doesn't create fake variety
        if node.expressions:
            node.set("expressions", [exp.Placeholder()])
        return node
    if isinstance(node, exp.Literal):
        return exp.Placeholder()
    if isinstance(node, exp.Boolean):
        return exp.Placeholder()
    return node


def _canonical_identifiers(tree):
    """Rename every identifier consistently by order of first appearance."""
    mapping = {}
    for node in tree.walk():
        if isinstance(node, exp.Identifier):
            key = node.name.lower()
            if key not in mapping:
                mapping[key] = f"n{len(mapping) + 1}"
            node.set("this", mapping[key])
            node.set("quoted", False)
    return tree


def _operand_key(node, blank_identifiers: bool, totals=None):
    """Sort key for a boolean operand. At skeleton level identifiers are blanked
    so the key reflects only the operand's *shape*, never its column names.

    Ties are broken by how often each of the operand's identifiers is used
    elsewhere in the query (e.g. also projected in SELECT), which is
    name-independent but separates `a > 1` from an unrelated `x > 1`."""
    names = [i.name.lower() for i in node.find_all(exp.Identifier)]
    c = node.copy()
    if blank_identifiers:
        for ident in list(c.find_all(exp.Identifier)):
            ident.set("this", "_")
            ident.set("quoted", False)
    shape = c.sql(comments=False).lower()
    if totals is None:
        return (shape, ())
    local = Counter(names)
    return (shape, tuple(totals[n] - local[n] for n in names))


def _sort_commutative(tree, blank_identifiers: bool):
    """Put AND / OR operands in a canonical order so that
    `a > 1 AND b = 2`  and  `b = 2 AND a > 1`  fingerprint identically.

    Must run BEFORE _canonical_identifiers: that pass numbers identifiers by
    order of appearance, so operand order would otherwise leak into the names.
    Operands whose shapes tie keep their original relative order (stable sort).
    """
    totals = Counter(i.name.lower() for i in tree.find_all(exp.Identifier))
    roots = [
        n for n in tree.walk()
        if isinstance(n, (exp.And, exp.Or)) and type(n.parent) is not type(n)
    ]
    for root in reversed(roots):  # deepest chains first
        cls = type(root)
        operands, stack = [], [root]
        while stack:  # flatten a same-type chain, left to right
            n = stack.pop()
            if isinstance(n, cls):
                stack.append(n.args["expression"])
                stack.append(n.args["this"])
            else:
                operands.append(n)
        ordered = sorted(operands, key=lambda o: _operand_key(o, blank_identifiers, totals))
        new = ordered[0].copy()
        for o in ordered[1:]:
            new = cls(this=new, expression=o.copy())
        if root.parent is None:
            tree = new
        else:
            root.replace(new)
    return tree


def sql_fingerprint(sql: str, level: str = "skeleton", dialect: str | None = None):
    """Return (canonical_form, parsed_ok). Hash the canonical form for the key."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return "", False
    if not HAVE_SQLGLOT:
        return "regex:" + _regex_normalize(sql, level), False
    try:
        statements = sqlglot.parse(sql, read=dialect)
        if not statements or any(s is None for s in statements):
            raise ValueError("empty parse")
    except Exception:
        return "regex:" + _regex_normalize(sql, level), False

    forms = []
    for tree in statements:
        if level != "exact":
            tree = tree.transform(_mask_literals)
        if level != "exact":
            tree = _sort_commutative(tree, blank_identifiers=(level == "skeleton"))
        if level == "skeleton":
            tree = _canonical_identifiers(tree)
        try:
            rendered = tree.sql(dialect=dialect, comments=False, normalize=True)
        except Exception:
            return "regex:" + _regex_normalize(sql, level), False
        forms.append(_WS_RE.sub(" ", rendered.lower()).strip())
    return " ; ".join(forms), True


def fp_hash(canonical: str) -> str:
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NL normalization / similarity
# ---------------------------------------------------------------------------

_NL_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NL_QUOTED_RE = re.compile(r"\"[^\"]*\"|'[^']*'|“[^”]*”")
_NL_WORD_RE = re.compile(r"[a-z0-9_<>]+")

NL_STOPWORDS = frozenset(
    "a an the of for in on at to is are was were do does did be been being "
    "me my i you your we us it its this that these those and or with by from "
    "please show give get list find tell what which who whose whom how many "
    "much all can could would there".split()
)


def nl_normalize(text: str) -> str:
    s = (text or "").lower()
    s = _NL_QUOTED_RE.sub(" <val> ", s)
    s = _NL_NUM_RE.sub(" <num> ", s)
    return " ".join(_NL_WORD_RE.findall(s))


def nl_tokens(text: str) -> frozenset:
    return frozenset(t for t in nl_normalize(text).split() if t not in NL_STOPWORDS)


def nl_similarity(a: str, b: str) -> float:
    ta, tb = nl_tokens(a), nl_tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

NL_FIELD_CANDIDATES = (
    "question", "nl", "natural_language", "nl_query", "instruction",
    "prompt", "input", "text", "utterance",
)
SQL_FIELD_CANDIDATES = ("sql", "query", "sql_query", "output", "answer", "target", "gold_sql")


def detect_fields(record: dict, nl_field: str | None, sql_field: str | None):
    keys = {k.lower(): k for k in record}
    if not sql_field:
        for cand in SQL_FIELD_CANDIDATES:
            if cand in keys and (cand != "query" or "question" in keys or "nl" in keys):
                sql_field = keys[cand]
                break
    if not nl_field:
        for cand in NL_FIELD_CANDIDATES:
            if cand in keys and keys[cand] != sql_field:
                nl_field = keys[cand]
                break
    if not sql_field:
        raise SystemExit(
            f"error: could not detect the SQL field among {sorted(record)}; "
            "pass --sql-field explicitly"
        )
    return nl_field, sql_field


def load_dataset(path: Path):
    """Yield (record_dict, raw_line_or_None). Supports .jsonl, .json, .csv."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            yield from ((row, None) for row in csv.DictReader(f))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise SystemExit("error: .json input must be an array of objects")
        yield from ((row, None) for row in data)
    else:  # jsonl / anything else
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line), line


def write_dataset(path: Path, records: list[dict]):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        fieldnames = list({k: None for r in records for k in r})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    elif suffix == ".json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core dedupe pass
# ---------------------------------------------------------------------------

class PatternIndex:
    """fingerprint-hash -> group info; optionally NL-aware."""

    def __init__(self, level="skeleton", dialect=None, nl_mode="off", nl_threshold=0.6, strict=False):
        self.level = level
        self.dialect = dialect
        self.nl_mode = nl_mode  # off | fuzzy
        self.nl_threshold = nl_threshold
        self.groups: dict[str, dict] = {}
        self.parse_failures = 0
        self.parse_failure_examples: list[str] = []
        # strict: a query sqlglot can't parse is never merged by regex heuristics;
        # only a verbatim repeat (whitespace-collapsed) counts as a duplicate.
        self.strict = strict

    def key_for(self, sql: str):
        canonical, ok = sql_fingerprint(sql, self.level, self.dialect)
        if not ok:
            self.parse_failures += 1
            if len(self.parse_failure_examples) < 20:
                self.parse_failure_examples.append(sql.strip()[:200])
            if self.strict:
                canonical = "unparsed:" + _WS_RE.sub(" ", (sql or "").strip())
        return fp_hash(canonical), canonical

    def match(self, nl: str, sql: str):
        """Return (key, group_or_None). group is None => pattern is new."""
        key, canonical = self.key_for(sql)
        group = self.groups.get(key)
        if group is None:
            return key, canonical, None
        if self.nl_mode == "fuzzy":
            # same SQL shape only counts as duplicate if some kept NL is similar
            if not any(nl_similarity(nl, kept) >= self.nl_threshold for kept in group["nls"]):
                return key, canonical, None
        return key, canonical, group

    def add(self, key, canonical, nl, sql):
        group = self.groups.get(key)
        if group is None:
            group = self.groups[key] = {
                "canonical": canonical,
                "count": 0,
                "example_nl": nl,
                "example_sql": sql,
                "nls": [],
            }
        group["count"] += 1
        if self.nl_mode == "fuzzy" and len(group["nls"]) < 50:
            group["nls"].append(nl)
        return group

    # -- persistence -------------------------------------------------------
    def save(self, path: Path):
        payload = {
            "level": self.level,
            "dialect": self.dialect,
            "nl_mode": self.nl_mode,
            "nl_threshold": self.nl_threshold,
            "strict": self.strict,
            "groups": self.groups,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        idx = cls(
            level=payload.get("level", "skeleton"),
            dialect=payload.get("dialect"),
            nl_mode=payload.get("nl_mode", "off"),
            nl_threshold=payload.get("nl_threshold", 0.6),
            strict=payload.get("strict", False),
        )
        idx.groups = payload["groups"]
        return idx


def run_pass(records, nl_field, sql_field, index: PatternIndex):
    """Classify each record as kept (new pattern) or dropped (duplicate)."""
    kept, dropped = [], []
    for i, rec in enumerate(records):
        nl = str(rec.get(nl_field, "") if nl_field else "")
        sql = str(rec.get(sql_field, ""))
        key, canonical, group = index.match(nl, sql)
        if group is None:
            index.add(key, canonical, nl, sql)
            kept.append((i, rec, key))
        else:
            group["count"] += 1
            dropped.append((i, rec, key))
    return kept, dropped


def assign_all(records, nl_field, sql_field, index: PatternIndex):
    """Like run_pass, but returns every record's pattern key (dupe or not)
    grouped in original order, for the HTML report."""
    order = []
    groups: dict[str, list[int]] = defaultdict(list)
    canonical_of: dict[str, str] = {}
    for i, rec in enumerate(records):
        nl = str(rec.get(nl_field, "") if nl_field else "")
        sql = str(rec.get(sql_field, ""))
        key, canonical = index.key_for(sql)
        canonical_of[key] = canonical
        groups[key].append(i)
        order.append(key)
    return order, groups, canonical_of


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SQL Pattern Duplicate Review</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 0; background: #0e1116; color: #dde3ea; }
  @media (prefers-color-scheme: light) { body { background: #f5f6f8; color: #1a1f27; } }
  header { position: sticky; top: 0; background: #161b22; border-bottom: 1px solid #2a3038; padding: 14px 20px; z-index: 10; }
  @media (prefers-color-scheme: light) { header { background: #fff; border-bottom: 1px solid #ddd; } }
  h1 { font-size: 17px; margin: 0 0 8px; }
  .stats { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; opacity: .9; margin-bottom: 10px; }
  .stats b { font-variant-numeric: tabular-nums; }
  .controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  input[type=text] { flex: 1; min-width: 220px; padding: 7px 10px; border-radius: 6px; border: 1px solid #3a4250; background: #0e1116; color: inherit; }
  @media (prefers-color-scheme: light) { input[type=text] { background: #fff; border: 1px solid #ccc; } }
  button { padding: 7px 14px; border-radius: 6px; border: 1px solid #3a4250; background: #21262d; color: #dde3ea; cursor: pointer; font-size: 13px; }
  @media (prefers-color-scheme: light) { button { background: #eee; color: #1a1f27; border: 1px solid #ccc; } }
  button.primary { background: #2f81f7; border-color: #2f81f7; color: #fff; }
  button:hover { filter: brightness(1.15); }
  select { padding: 6px 8px; border-radius: 6px; background: #0e1116; color: inherit; border: 1px solid #3a4250; }
  @media (prefers-color-scheme: light) { select { background: #fff; border: 1px solid #ccc; } }
  main { padding: 16px 20px 60px; max-width: 1100px; margin: 0 auto; }
  .group { border: 1px solid #2a3038; border-radius: 8px; margin-bottom: 12px; overflow: hidden; }
  @media (prefers-color-scheme: light) { .group { border: 1px solid #ddd; } }
  .group-head { display: flex; align-items: center; gap: 10px; padding: 10px 12px; cursor: pointer; background: #161b22; }
  @media (prefers-color-scheme: light) { .group-head { background: #fafbfc; } }
  .group.singleton .group-head { background: #10231a; }
  @media (prefers-color-scheme: light) { .group.singleton .group-head { background: #eaf7ee; } }
  .badge { font-size: 11px; padding: 2px 7px; border-radius: 999px; font-weight: 600; white-space: nowrap; }
  .badge.dup { background: #4a2530; color: #ff9aa8; }
  .badge.uniq { background: #1e3a2a; color: #7be495; }
  .pattern { font-family: ui-monospace, Consolas, monospace; font-size: 12px; opacity: .85; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chevron { transition: transform .15s; opacity: .6; }
  .group.open .chevron { transform: rotate(90deg); }
  .rows { display: none; border-top: 1px solid #2a3038; }
  @media (prefers-color-scheme: light) { .rows { border-top: 1px solid #ddd; } }
  .group.open .rows { display: block; }
  .row { display: flex; gap: 10px; align-items: flex-start; padding: 9px 14px; border-bottom: 1px solid #1c2128; }
  @media (prefers-color-scheme: light) { .row { border-bottom: 1px solid #eee; } }
  .row:last-child { border-bottom: none; }
  .row .txt { flex: 1; min-width: 0; }
  .row .nl { font-size: 13px; margin-bottom: 3px; }
  .row .sql { font-family: ui-monospace, Consolas, monospace; font-size: 12px; opacity: .8; white-space: pre-wrap; word-break: break-word; }
  .row .idx { font-size: 11px; opacity: .5; min-width: 34px; padding-top: 2px; }
  .hidden { display: none !important; }
  footer { position: fixed; bottom: 0; left: 0; right: 0; background: #161b22; border-top: 1px solid #2a3038; padding: 10px 20px; display: flex; gap: 10px; align-items: center; }
  @media (prefers-color-scheme: light) { footer { background: #fff; border-top: 1px solid #ddd; } }
  footer .note { font-size: 12px; opacity: .7; flex: 1; }
</style>
</head>
<body>
<header>
  <h1>SQL Pattern Duplicate Review — __TITLE__</h1>
  <div class="stats">
    <span>total pairs: <b id="statTotal"></b></span>
    <span>unique patterns: <b id="statUnique"></b></span>
    <span>duplicate pairs: <b id="statDup"></b></span>
    <span>level: <b>__LEVEL__</b></span>
  </div>
  <div class="controls">
    <input type="text" id="search" placeholder="Filter by question or SQL text...">
    <select id="filterMode">
      <option value="all">Show all groups</option>
      <option value="dup">Duplicate groups only</option>
      <option value="uniq">Unique (singleton) only</option>
    </select>
    <button id="expandAll">Expand all</button>
    <button id="collapseAll">Collapse all</button>
  </div>
</header>
<main id="main"></main>
<footer>
  <span class="note">Each duplicate group keeps its first (radio-selected) row on export. Pick a different row per group if you'd rather keep that variant's wording.</span>
  <button id="exportJsonl" class="primary">Export unique dataset (.jsonl)</button>
  <button id="exportJson">Export (.json)</button>
</footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const NL_FIELD = DATA.nl_field, SQL_FIELD = DATA.sql_field;
const groups = DATA.groups; // [{key, canonical, count, rows:[{idx, record}]}]

document.getElementById('statTotal').textContent = DATA.total;
document.getElementById('statUnique').textContent = groups.length;
document.getElementById('statDup').textContent = DATA.total - groups.length;

const main = document.getElementById('main');
const els = [];

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

groups.forEach((g, gi) => {
  const dup = g.rows.length > 1;
  const div = document.createElement('div');
  div.className = 'group' + (dup ? '' : ' singleton');
  div.dataset.dup = dup ? '1' : '0';
  const searchBlob = (g.canonical + ' ' + g.rows.map(r => (r.record[NL_FIELD]||'') + ' ' + (r.record[SQL_FIELD]||'')).join(' ')).toLowerCase();
  div.dataset.search = searchBlob;

  const head = document.createElement('div');
  head.className = 'group-head';
  head.innerHTML = `
    <span class="chevron">▶</span>
    <span class="badge ${dup ? 'dup' : 'uniq'}">${dup ? g.rows.length + 'x duplicate' : 'unique'}</span>
    <span class="pattern">${escapeHtml(g.canonical)}</span>
  `;
  head.addEventListener('click', () => div.classList.toggle('open'));
  div.appendChild(head);

  const rowsWrap = document.createElement('div');
  rowsWrap.className = 'rows';
  g.rows.forEach((r, ri) => {
    const row = document.createElement('div');
    row.className = 'row';
    const nl = r.record[NL_FIELD] || '';
    const sql = r.record[SQL_FIELD] || '';
    row.innerHTML = `
      <span class="idx">#${r.idx}</span>
      ${dup ? `<input type="radio" name="keep-${gi}" value="${ri}" ${ri===0?'checked':''} title="keep this row on export">` : ''}
      <div class="txt">
        <div class="nl">${escapeHtml(nl)}</div>
        <div class="sql">${escapeHtml(sql)}</div>
      </div>
    `;
    rowsWrap.appendChild(row);
  });
  div.appendChild(rowsWrap);
  main.appendChild(div);
  els.push(div);
});

function applyFilter() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const mode = document.getElementById('filterMode').value;
  els.forEach(div => {
    const matchesText = !q || div.dataset.search.includes(q);
    const matchesMode = mode === 'all' || (mode === 'dup' && div.dataset.dup === '1') || (mode === 'uniq' && div.dataset.dup === '0');
    div.classList.toggle('hidden', !(matchesText && matchesMode));
  });
}
document.getElementById('search').addEventListener('input', applyFilter);
document.getElementById('filterMode').addEventListener('change', applyFilter);
document.getElementById('expandAll').addEventListener('click', () => els.forEach(d => d.classList.add('open')));
document.getElementById('collapseAll').addEventListener('click', () => els.forEach(d => d.classList.remove('open')));

function collectUnique() {
  return groups.map((g, gi) => {
    if (g.rows.length === 1) return g.rows[0].record;
    const checked = main.querySelectorAll(`input[name="keep-${gi}"]:checked`)[0];
    const ri = checked ? parseInt(checked.value, 10) : 0;
    return g.rows[ri].record;
  });
}

function download(filename, text) {
  const blob = new Blob([text], {type: 'application/octet-stream'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

document.getElementById('exportJsonl').addEventListener('click', () => {
  const rows = collectUnique();
  download('unique_dataset.jsonl', rows.map(r => JSON.stringify(r)).join('\n'));
});
document.getElementById('exportJson').addEventListener('click', () => {
  const rows = collectUnique();
  download('unique_dataset.json', JSON.stringify(rows, null, 2));
});
</script>
</body>
</html>
"""


def generate_html_report(path_title: str, level: str, rows, nl_field, sql_field, index: PatternIndex, out_path: Path):
    order, group_idx, canonical_of = assign_all(rows, nl_field, sql_field, index)
    groups_out = []
    for key, idxs in group_idx.items():
        groups_out.append({
            "key": key,
            "canonical": canonical_of[key],
            "count": len(idxs),
            "rows": [{"idx": i, "record": rows[i]} for i in idxs],
        })
    groups_out.sort(key=lambda g: (-g["count"], g["canonical"]))
    payload = {
        "total": len(rows),
        "nl_field": nl_field or "",
        "sql_field": sql_field,
        "groups": groups_out,
    }
    html = (
        _HTML_TEMPLATE
        .replace("__TITLE__", path_title)
        .replace("__LEVEL__", level)
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    )
    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_dedupe(args):
    path = Path(args.input)
    rows = [rec for rec, _ in load_dataset(path)]
    if not rows:
        raise SystemExit("error: input dataset is empty")
    nl_field, sql_field = detect_fields(rows[0], args.nl_field, args.sql_field)

    index = PatternIndex(args.level, args.dialect, args.nl_mode, args.nl_threshold,
                         strict=getattr(args, "strict", False))
    kept, dropped = run_pass(rows, nl_field, sql_field, index)

    out_path = Path(args.output) if args.output else path.with_name(path.stem + ".unique" + path.suffix)
    write_dataset(out_path, [rec for _, rec, _ in kept])

    dup_groups = {k: g for k, g in index.groups.items() if g["count"] > 1}
    if args.report:
        report = {
            "input": str(path),
            "level": args.level,
            "nl_mode": args.nl_mode,
            "total": len(rows),
            "unique_patterns": len(index.groups),
            "kept": len(kept),
            "dropped": len(dropped),
            "sql_parse_failures": index.parse_failures,
            "sql_parse_failure_examples": index.parse_failure_examples,
            "duplicate_groups": [
                {
                    "pattern": g["canonical"],
                    "count": g["count"],
                    "example_nl": g["example_nl"],
                    "example_sql": g["example_sql"],
                }
                for g in sorted(dup_groups.values(), key=lambda g: -g["count"])
            ],
        }
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_index:
        index.save(Path(args.save_index))

    if args.html:
        html_path = Path(args.html)
        generate_html_report(path.name, args.level, rows, nl_field, sql_field, index, html_path)

    print(f"total pairs        : {len(rows)}")
    print(f"unique patterns    : {len(index.groups)}  (level={args.level}, nl_mode={args.nl_mode})")
    print(f"kept               : {len(kept)}  -> {out_path}")
    print(f"dropped duplicates : {len(dropped)}")
    if index.parse_failures:
        mode = "kept unless verbatim repeat (--strict)" if index.strict else "regex-normalized instead of AST"
        print(f"parse fallbacks    : {index.parse_failures} ({mode})")
        for ex in index.parse_failure_examples[:3]:
            print(f"    unparsed: {ex[:100]}")
    if dup_groups:
        print("\ntop repeated patterns:")
        for g in sorted(dup_groups.values(), key=lambda g: -g["count"])[:10]:
            print(f"  x{g['count']:<4} {g['canonical'][:110]}")
    if args.report:
        print(f"\nreport             : {args.report}")
    if args.save_index:
        print(f"pattern index      : {args.save_index}")
    if args.html:
        print(f"html review        : {args.html}  (open in a browser)")


def cmd_build_index(args):
    path = Path(args.input)
    rows = [rec for rec, _ in load_dataset(path)]
    if not rows:
        raise SystemExit("error: input dataset is empty")
    nl_field, sql_field = detect_fields(rows[0], args.nl_field, args.sql_field)
    index = PatternIndex(args.level, args.dialect, args.nl_mode, args.nl_threshold,
                         strict=getattr(args, "strict", False))
    run_pass(rows, nl_field, sql_field, index)
    index.save(Path(args.output))
    print(f"indexed {len(rows)} pairs -> {len(index.groups)} unique patterns -> {args.output}")


def cmd_check(args):
    index = PatternIndex.load(Path(args.index))
    pairs = []
    if args.sql:
        pairs.append({"nl": args.nl or "", "sql": args.sql})
        nl_field, sql_field = "nl", "sql"
    elif args.file:
        rows = [rec for rec, _ in load_dataset(Path(args.file))]
        if not rows:
            raise SystemExit("error: --file dataset is empty")
        nl_field, sql_field = detect_fields(rows[0], args.nl_field, args.sql_field)
        pairs = rows
    else:
        raise SystemExit("error: pass --sql (with optional --nl) or --file")

    new_count = 0
    for rec in pairs:
        nl = str(rec.get(nl_field, "") if nl_field else "")
        sql = str(rec.get(sql_field, ""))
        key, canonical, group = index.match(nl, sql)
        if group is None:
            new_count += 1
            print(f"NEW        {sql[:100]}")
            if args.add:
                index.add(key, canonical, nl, sql)
        else:
            print(f"DUPLICATE  {sql[:100]}")
            print(f"           pattern seen {group['count']}x, e.g. NL: {group['example_nl'][:90]}")
            print(f"           SQL: {group['example_sql'][:100]}")
    if len(pairs) > 1:
        print(f"\n{new_count}/{len(pairs)} pairs are NEW patterns")
    if args.add:
        index.save(Path(args.index))
        print(f"index updated: {args.index}")
    sys.exit(0 if new_count == len(pairs) else 3)  # exit 3 => duplicates found


def cmd_stats(args):
    path = Path(args.input)
    rows = [rec for rec, _ in load_dataset(path)]
    if not rows:
        raise SystemExit("error: input dataset is empty")
    nl_field, sql_field = detect_fields(rows[0], args.nl_field, args.sql_field)
    print(f"{'level':<10} {'unique':>8} {'dup pairs':>10} {'dup %':>7}")
    for level in LEVELS:
        index = PatternIndex(level, args.dialect)
        kept, dropped = run_pass(rows, nl_field, sql_field, index)
        pct = 100.0 * len(dropped) / len(rows)
        print(f"{level:<10} {len(index.groups):>8} {len(dropped):>10} {pct:>6.1f}%")
    print(f"\ntotal pairs: {len(rows)}  (fields: nl={nl_field!r}, sql={sql_field!r})")


# ---------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--nl-field", help="name of the natural-language field (auto-detected)")
    p.add_argument("--sql-field", help="name of the SQL field (auto-detected)")
    p.add_argument("--dialect", help="SQL dialect for parsing (e.g. mysql, postgres, tsql, sqlite)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sqlpattern", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dedupe", help="drop duplicate-pattern pairs from a dataset")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="unique-pairs output (default: <input>.unique.<ext>)")
    p.add_argument("--report", help="write a JSON report of duplicate groups")
    p.add_argument("--html", help="write a browsable HTML report (view/filter duplicate groups, "
                                   "pick which row to keep per group, export unique dataset from the browser)")
    p.add_argument("--save-index", help="also save the pattern index for later `check` calls")
    p.add_argument("--level", choices=LEVELS, default="skeleton")
    p.add_argument("--nl-mode", choices=("off", "fuzzy"), default="off",
                   help="'fuzzy': same SQL shape is a duplicate only if the NL is also similar")
    p.add_argument("--nl-threshold", type=float, default=0.6)
    p.add_argument("--strict", action="store_true",
                   help="never merge SQL that sqlglot can't parse (only verbatim repeats are dropped)")
    add_common(p)
    p.set_defaults(func=cmd_dedupe)

    p = sub.add_parser("build-index", help="build a pattern index from a dataset")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="pattern_index.json")
    p.add_argument("--level", choices=LEVELS, default="skeleton")
    p.add_argument("--nl-mode", choices=("off", "fuzzy"), default="off")
    p.add_argument("--nl-threshold", type=float, default=0.6)
    p.add_argument("--strict", action="store_true",
                   help="never merge SQL that sqlglot can't parse (only verbatim repeats are dropped)")
    add_common(p)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("check", help="check new pair(s) against an index")
    p.add_argument("--index", default="pattern_index.json")
    p.add_argument("--nl", help="natural-language question of the new pair")
    p.add_argument("--sql", help="SQL of the new pair")
    p.add_argument("--file", help="or: a dataset file of new pairs to check")
    p.add_argument("--add", action="store_true", help="add NEW patterns to the index")
    add_common(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("stats", help="pattern counts at every level")
    p.add_argument("input")
    add_common(p)
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    if not HAVE_SQLGLOT:
        print("warning: sqlglot not installed - using regex fallback only "
              "(pip install sqlglot for accurate AST-based fingerprints)", file=sys.stderr)
    args.func(args)


if __name__ == "__main__":
    main()
