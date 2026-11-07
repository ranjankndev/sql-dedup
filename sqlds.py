#!/usr/bin/env python3
"""
sqlds - end-to-end workbench for building an NL->SQL training dataset.

Workflow:
  1. sqlds init --db bank.db                 configure workspace (DB + dataset file)
  2. sqlds import existing.jsonl             bulk-load pairs, duplicates dropped
  3. sqlds add --nl "..." --sql "..."        add one pair: dedupe check + live DB
                                             verification (row counts, join result)
  4. sqlds gaps                              which PostgreSQL/SQLite syntax patterns
                                             the dataset does NOT cover yet
  5. sqlds suggest                           generate candidate SQL from your schema
                                             for every gap, verified against the DB;
                                             you write the NL and `add` them
  6. sqlds verify --sql "..."                dry-run any SQL: per-table row counts +
                                             result sample (never commits)
  7. sqlds status / coverage / export        progress and SFT-ready output

Config lives in sqlds.json in the current directory.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from sqldb import Database, referenced_tables
from sqlfeatures import TAXONOMY, coverage, detect, generate
from sqlpattern import (
    PatternIndex,
    detect_fields,
    load_dataset,
    write_dataset,
)

CONFIG_FILE = "sqlds.json"


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class Workspace:
    def __init__(self, root: Path):
        self.root = root
        cfg_path = root / CONFIG_FILE
        if not cfg_path.exists():
            raise SystemExit(f"error: no {CONFIG_FILE} here - run `sqlds init --db <db>` first")
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        self.dataset_path = root / self.cfg["dataset"]
        self.level = self.cfg.get("level", "skeleton")
        self.db = Database(self.cfg["db"]) if self.cfg.get("db") else None
        self.dialect = self.db.dialect if self.db else self.cfg.get("dialect")

    def load_pairs(self) -> list[dict]:
        if not self.dataset_path.exists():
            return []
        return [rec for rec, _ in load_dataset(self.dataset_path)]

    def append_pair(self, rec: dict):
        with self.dataset_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def build_index(self, pairs: list[dict]) -> PatternIndex:
        index = PatternIndex(self.level, self.dialect)
        for rec in pairs:
            nl, sql = rec.get("question", ""), rec.get("sql", "")
            key, canonical, group = index.match(nl, sql)
            if group is None:
                index.add(key, canonical, nl, sql)
            else:
                group["count"] += 1
        return index


def make_record(nl: str, sql: str, verified: bool, dialect: str | None) -> dict:
    return {
        "question": nl,
        "sql": sql.strip().rstrip(";"),
        "features": sorted(detect(sql, dialect)),
        "verified": verified,
        "added_at": datetime.date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Verification rendering
# ---------------------------------------------------------------------------

def _fmt_table(columns, rows, max_width=28):
    def clip(v):
        s = "NULL" if v is None else str(v)
        return s if len(s) <= max_width else s[: max_width - 1] + "…"

    cols = [clip(c) for c in columns]
    body = [[clip(v) for v in row] for row in rows]
    widths = [len(c) for c in cols]
    for row in body:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    out = [line, sep]
    out += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in body]
    return "\n".join(out)


def verify_sql(ws: Workspace, sql: str, limit: int = 10, commit: bool = False) -> bool:
    """Print the verification report; return True when the query ran OK."""
    if ws.db is None:
        print("  (no database configured - skipping verification)")
        return True
    tables = referenced_tables(sql, ws.dialect)
    if tables:
        print("  tables referenced:")
        for t in tables:
            n = ws.db.row_count(t)
            print(f"    {t:<24} {'<not found>' if n is None else f'{n} rows'}")
    result = ws.db.run(sql, limit=limit, commit=commit)
    if not result["ok"]:
        print(f"  EXECUTION FAILED: {result['error']}")
        return False
    if result["is_select"]:
        total = result["total_rows"]
        print(f"  query result: {total} row(s)" + (f", showing first {len(result['rows'])}:" if result["rows"] else ""))
        if result["rows"]:
            print("    " + _fmt_table(result["columns"], result["rows"]).replace("\n", "\n    "))
        elif total == 0:
            print("    (empty result - the SQL ran, but check whether 0 rows is expected)")
    else:
        action = "committed" if result["committed"] else "rolled back (dry run)"
        print(f"  DML ok: {result['affected']} row(s) affected - {action}")
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    cfg_path = Path.cwd() / CONFIG_FILE
    cfg = {
        "db": args.db,
        "dataset": args.dataset,
        "level": args.level,
    }
    if args.db and not args.db.startswith(("postgresql://", "postgres://")):
        p = Path(args.db)
        if not p.exists():
            print(f"note: SQLite file {args.db} does not exist yet (it will be created on first use)")
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    dataset = Path.cwd() / args.dataset
    if not dataset.exists():
        dataset.touch()
    print(f"workspace ready: {cfg_path}")
    print(f"  db      : {args.db or '(none - verification disabled)'}")
    print(f"  dataset : {args.dataset}")
    print(f"  level   : {args.level}")


def cmd_add(args):
    ws = Workspace(Path.cwd())
    pairs = ws.load_pairs()
    index = ws.build_index(pairs)

    key, canonical, group = index.match(args.nl, args.sql)
    print(f"pattern: {canonical[:120]}")
    if group is not None and not args.force:
        print(f"\nDUPLICATE - this SQL pattern is already in the dataset ({group['count']}x):")
        print(f"  existing NL : {group['example_nl']}")
        print(f"  existing SQL: {group['example_sql']}")
        print("\nnot added. Use --force to add anyway.")
        sys.exit(3)
    if group is not None:
        print(f"note: duplicate pattern (seen {group['count']}x) - adding anyway (--force)")
    else:
        print("pattern check: NEW")

    verified = False
    if not args.no_verify:
        print("\nverification:")
        ok = verify_sql(ws, args.sql, limit=args.limit)
        if not ok and not args.force:
            print("\nnot added - the SQL failed to execute. Fix it, or use --no-verify/--force.")
            sys.exit(2)
        verified = ok

    feats = sorted(detect(args.sql, ws.dialect))
    if args.dry_run:
        print(f"\ndry run - would add with features: {', '.join(feats) or '(none detected)'}")
        return
    ws.append_pair(make_record(args.nl, args.sql, verified, ws.dialect))
    new_feats = [f for f in feats if f in _missing_features(pairs, ws)]
    print(f"\nadded pair #{len(pairs) + 1}  features: {', '.join(feats) or '-'}")
    if new_feats:
        print(f"coverage improved - first query using: {', '.join(new_feats)}")


def _missing_features(pairs: list[dict], ws: Workspace) -> set[str]:
    sets = [set(r.get("features") or detect(r.get("sql", ""), ws.dialect)) for r in pairs]
    _, missing = coverage(sets)
    return set(missing)


def cmd_import(args):
    ws = Workspace(Path.cwd())
    pairs = ws.load_pairs()
    index = ws.build_index(pairs)

    rows = [rec for rec, _ in load_dataset(Path(args.file))]
    if not rows:
        raise SystemExit("error: import file is empty")
    nl_field, sql_field = detect_fields(rows[0], args.nl_field, args.sql_field)

    added = dup = failed = 0
    for rec in rows:
        nl = str(rec.get(nl_field, "") if nl_field else "")
        sql = str(rec.get(sql_field, ""))
        key, canonical, group = index.match(nl, sql)
        if group is not None:
            group["count"] += 1
            dup += 1
            continue
        verified = False
        if args.verify and ws.db is not None:
            result = ws.db.run(sql, limit=1)
            verified = result["ok"]
            if not verified:
                failed += 1
                print(f"  exec-failed (kept, verified=false): {sql[:80]}")
        index.add(key, canonical, nl, sql)
        ws.append_pair(make_record(nl, sql, verified, ws.dialect))
        added += 1
    print(f"imported {added} new pattern(s), dropped {dup} duplicate(s)"
          + (f", {failed} failed execution" if args.verify else ""))
    print(f"dataset now has {len(pairs) + added} pairs")


def cmd_status(args):
    ws = Workspace(Path.cwd())
    pairs = ws.load_pairs()
    index = ws.build_index(pairs)
    sets = [set(r.get("features") or detect(r.get("sql", ""), ws.dialect)) for r in pairs]
    counts, missing = coverage(sets)
    verified = sum(1 for r in pairs if r.get("verified"))
    print(f"dataset  : {ws.dataset_path.name}  ({len(pairs)} pairs, {verified} verified)")
    print(f"db       : {ws.cfg.get('db') or '(none)'}")
    print(f"patterns : {len(index.groups)} unique at level={ws.level}")
    print(f"coverage : {len(TAXONOMY) - len(missing)}/{len(TAXONOMY)} syntax patterns covered")
    if missing:
        print(f"gaps     : {', '.join(missing[:8])}" + (" ..." if len(missing) > 8 else ""))
        print("\nrun `sqlds gaps` for the full list, `sqlds suggest` for candidate SQL")


def cmd_coverage(args):
    ws = Workspace(Path.cwd())
    pairs = ws.load_pairs()
    sets = [set(r.get("features") or detect(r.get("sql", ""), ws.dialect)) for r in pairs]
    counts, missing = coverage(sets)
    print(f"{'feature':<20} {'queries':>8}  description")
    print("-" * 76)
    for key, desc in TAXONOMY.items():
        n = counts[key]
        marker = "  " if n else "* "
        print(f"{marker}{key:<18} {n:>8}  {desc}")
    print(f"\n* = not covered ({len(missing)} gap(s) of {len(TAXONOMY)})")


def cmd_gaps(args):
    ws = Workspace(Path.cwd())
    pairs = ws.load_pairs()
    sets = [set(r.get("features") or detect(r.get("sql", ""), ws.dialect)) for r in pairs]
    _, missing = coverage(sets)
    if not missing:
        print("no gaps - every taxonomy pattern is covered at least once")
        return
    print(f"{len(missing)} uncovered syntax pattern(s):\n")
    for key in missing:
        print(f"  {key:<20} {TAXONOMY[key]}")
    print("\nrun `sqlds suggest` to generate candidate SQL for these from your schema")


def cmd_suggest(args):
    ws = Workspace(Path.cwd())
    if ws.db is None:
        raise SystemExit("error: suggest needs a database (sqlds init --db ...)")
    schema = ws.db.schema()
    if not schema.tables:
        raise SystemExit("error: the database has no tables")
    pairs = ws.load_pairs()
    missing = sorted(_missing_features(pairs, ws)) if not args.all else list(TAXONOMY)

    out_path = Path(args.out)
    candidates = []
    print(f"generating candidates for {len(missing)} pattern(s) "
          f"using schema of {ws.cfg['db']}:\n")
    for key in missing:
        if key in ("insert", "update", "delete"):
            continue  # write DML by hand; auto-running it is not worth the risk
        sql = generate(schema, key, ws.dialect or "sqlite")
        if sql is None:
            print(f"  {key:<20} (schema cannot support this pattern - skipped)")
            continue
        result = ws.db.run(sql, limit=3)
        status = "ok" if result["ok"] else "FAILED"
        rows = result["total_rows"] if result["ok"] and result["is_select"] else "-"
        print(f"  {key:<20} [{status}, {rows} rows]  {sql[:80]}")
        if not result["ok"]:
            print(f"    error: {result['error']}")
        candidates.append({
            "pattern": key,
            "description": TAXONOMY[key],
            "question": "",  # <- you write the business NL here
            "sql": sql,
            "runs_ok": result["ok"],
            "result_rows": result["total_rows"] if result["ok"] else None,
        })
    write_dataset(out_path, candidates)
    print(f"\n{len(candidates)} candidate(s) -> {out_path}")
    print("next: fill in the empty \"question\" fields (adapt SQL values to real business")
    print(f"terms), then:  python sqlds.py import {out_path} --verify")


def cmd_verify(args):
    ws = Workspace(Path.cwd())
    if args.sql:
        items = [("", args.sql)]
    elif args.file:
        rows = [rec for rec, _ in load_dataset(Path(args.file))]
        nl_field, sql_field = detect_fields(rows[0], None, None)
        items = [(str(r.get(nl_field, "")), str(r.get(sql_field, ""))) for r in rows]
    else:
        raise SystemExit("error: pass --sql \"...\" or --file <dataset>")
    ok_count = 0
    for nl, sql in items:
        if nl:
            print(f"NL : {nl}")
        print(f"SQL: {sql}")
        if verify_sql(ws, sql, limit=args.limit, commit=args.commit):
            ok_count += 1
        print()
    if len(items) > 1:
        print(f"{ok_count}/{len(items)} queries executed successfully")
    sys.exit(0 if ok_count == len(items) else 2)


def cmd_schema(args):
    ws = Workspace(Path.cwd())
    if ws.db is None:
        raise SystemExit("error: no database configured")
    schema = ws.db.schema()
    for t in schema.tables.values():
        n = ws.db.row_count(t.name)
        print(f"{t.name}  ({n if n is not None else '?'} rows)")
        for c in t.columns:
            tags = []
            if c.name in t.pk:
                tags.append("PK")
            for fk in t.fks:
                if fk.column == c.name:
                    tags.append(f"FK -> {fk.ref_table}.{fk.ref_column}")
            print(f"    {c.name:<24} {c.type:<16} {' '.join(tags)}")
        print()


def cmd_export(args):
    ws = Workspace(Path.cwd())
    pairs = ws.load_pairs()
    out = []
    skipped_unverified = skipped_no_nl = 0
    for r in pairs:
        if args.verified_only and not r.get("verified"):
            skipped_unverified += 1
            continue
        if not str(r.get("question", "")).strip():
            skipped_no_nl += 1
            continue
        out.append({"question": r["question"], "sql": r["sql"]})
    write_dataset(Path(args.out), out)
    print(f"exported {len(out)} pairs -> {args.out}")
    if skipped_unverified:
        print(f"  skipped {skipped_unverified} unverified (drop --verified-only to include)")
    if skipped_no_nl:
        print(f"  skipped {skipped_no_nl} with empty question field")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="sqlds", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="configure this directory as a dataset workspace")
    p.add_argument("--db", help="SQLite file path or postgresql:// DSN")
    p.add_argument("--dataset", default="dataset.jsonl")
    p.add_argument("--level", choices=("exact", "template", "skeleton"), default="skeleton")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="add one NL+SQL pair (dedupe + verify)")
    p.add_argument("--nl", required=True)
    p.add_argument("--sql", required=True)
    p.add_argument("--no-verify", action="store_true", help="skip DB execution")
    p.add_argument("--force", action="store_true", help="add even if duplicate / failing")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=10, help="result rows to display")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("import", help="bulk import a dataset file, dropping duplicates")
    p.add_argument("file")
    p.add_argument("--verify", action="store_true", help="also execute each query")
    p.add_argument("--nl-field")
    p.add_argument("--sql-field")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("status", help="dataset size, unique patterns, coverage summary")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("coverage", help="per-feature query counts")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("gaps", help="taxonomy patterns with zero coverage")
    p.set_defaults(func=cmd_gaps)

    p = sub.add_parser("suggest", help="generate candidate SQL from the schema for gaps")
    p.add_argument("--out", default="candidates.jsonl")
    p.add_argument("--all", action="store_true", help="generate for every pattern, not only gaps")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("verify", help="run SQL against the DB: table row counts + result")
    p.add_argument("--sql")
    p.add_argument("--file")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--commit", action="store_true",
                   help="commit DML instead of rolling back")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("schema", help="show tables, columns, keys, row counts")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("export", help="write clean SFT-ready pairs")
    p.add_argument("--out", default="sft_dataset.jsonl")
    p.add_argument("--verified-only", action="store_true")
    p.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
