#!/usr/bin/env python3
"""
sqlfeatures - SQL syntax taxonomy for PostgreSQL / SQLite.

Two jobs:
  1. detect(sql)            -> set of feature keys a query exercises
  2. generate(schema, key)  -> a candidate SQL query (using the real schema)
                               for a feature the dataset does not cover yet

The taxonomy is the checklist of "kinds of SQL" worth having in an SFT/RL
dataset. Coverage = union of detected features across the dataset;
gaps = taxonomy - coverage.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from sqldb import Schema, Table

# ---------------------------------------------------------------------------
# Taxonomy: key -> human description
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, str] = {
    # basic retrieval
    "select_star":        "SELECT * over a single table",
    "select_columns":     "SELECT of specific columns",
    "distinct":           "SELECT DISTINCT",
    "order_by":           "ORDER BY",
    "top_n":              "ORDER BY ... LIMIT (top-N)",
    "limit_offset":       "LIMIT with OFFSET (pagination)",
    # filtering
    "where_equality":     "WHERE col = value",
    "where_comparison":   "WHERE with < > <= >= comparison",
    "where_between":      "WHERE ... BETWEEN",
    "where_in_list":      "WHERE col IN (list)",
    "where_like":         "WHERE col LIKE pattern",
    "where_null":         "IS NULL / IS NOT NULL",
    "where_and_or":       "compound WHERE with AND/OR",
    "where_not":          "negation with NOT / !=",
    # aggregation
    "count":              "COUNT(*) / COUNT(col)",
    "agg_sum_avg":        "SUM / AVG aggregate",
    "agg_min_max":        "MIN / MAX aggregate",
    "group_by":           "GROUP BY",
    "having":             "GROUP BY ... HAVING",
    "count_distinct":     "COUNT(DISTINCT col)",
    # joins
    "inner_join":         "INNER JOIN via foreign key",
    "left_join":          "LEFT OUTER JOIN",
    "multi_join":         "3+ tables joined",
    "self_join":          "self join",
    "cross_join":         "CROSS JOIN / cartesian",
    "join_agg":           "JOIN combined with GROUP BY aggregation",
    # subqueries
    "subquery_in":        "WHERE col IN (SELECT ...)",
    "subquery_scalar":    "comparison against scalar subquery (e.g. > (SELECT AVG ...))",
    "subquery_exists":    "WHERE EXISTS (SELECT ...)",
    "derived_table":      "subquery in FROM (derived table)",
    "correlated":         "correlated subquery",
    # advanced
    "cte":                "WITH common table expression",
    "recursive_cte":      "WITH RECURSIVE",
    "union":              "UNION / UNION ALL",
    "intersect_except":   "INTERSECT / EXCEPT",
    "window_ranking":     "window function: ROW_NUMBER / RANK / DENSE_RANK",
    "window_agg":         "window aggregate: SUM/AVG/COUNT OVER (...)",
    "case_when":          "CASE WHEN expression",
    "date_function":      "date/time function or arithmetic",
    "string_function":    "string function (UPPER, LOWER, SUBSTR, ...)",
    "arithmetic_expr":    "arithmetic in SELECT/WHERE (col * 1.1, a - b)",
    "coalesce_ifnull":    "COALESCE / IFNULL / NULLIF",
    # DML (detected so coverage is visible; not auto-generated)
    "insert":             "INSERT statement",
    "update":             "UPDATE statement",
    "delete":             "DELETE statement",
}

_RANKING_FUNCS = ("ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE")
_DATE_FUNC_NAMES = {
    "strftime", "date", "datetime", "julianday", "now", "current_date",
    "current_timestamp", "extract", "date_part", "date_trunc", "age",
    "to_date", "to_timestamp", "to_char",
}
_STRING_FUNC_TYPES = (exp.Upper, exp.Lower, exp.Substring, exp.Trim, exp.Concat,
                      exp.Length, exp.Replace) if hasattr(exp, "Replace") else \
                     (exp.Upper, exp.Lower, exp.Substring, exp.Trim, exp.Concat, exp.Length)


def detect(sql: str, dialect: str | None = None) -> set[str]:
    """Feature keys exercised by one SQL query. Empty set if unparseable."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except Exception:
        return set()
    feats: set[str] = set()
    for tree in statements:
        feats |= _detect_tree(tree)
    return feats


def _detect_tree(tree) -> set[str]:
    f: set[str] = set()

    if isinstance(tree, exp.Insert):
        f.add("insert")
    if isinstance(tree, exp.Update):
        f.add("update")
    if isinstance(tree, exp.Delete):
        f.add("delete")

    selects = list(tree.find_all(exp.Select))
    for sel in selects:
        exprs = sel.expressions
        if any(isinstance(e, exp.Star) for e in exprs):
            f.add("select_star")
        if any(isinstance(e, (exp.Column, exp.Alias)) for e in exprs):
            f.add("select_columns")
        if sel.args.get("distinct"):
            f.add("distinct")
        if sel.args.get("group"):
            f.add("group_by")
        if sel.args.get("having"):
            f.add("having")

    if list(tree.find_all(exp.Order)):
        f.add("order_by")
    limit = list(tree.find_all(exp.Limit))
    if limit and "order_by" in f:
        f.add("top_n")
    if list(tree.find_all(exp.Offset)):
        f.add("limit_offset")

    # -- where-level features
    for where in tree.find_all(exp.Where):
        for node in where.walk():
            if isinstance(node, exp.EQ):
                f.add("where_equality")
            if isinstance(node, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                f.add("where_comparison")
            if isinstance(node, exp.Between):
                f.add("where_between")
            if isinstance(node, exp.In):
                if node.args.get("query"):
                    f.add("subquery_in")
                elif node.expressions:
                    f.add("where_in_list")
            if isinstance(node, (exp.Like, exp.ILike)):
                f.add("where_like")
            if isinstance(node, exp.Is):
                f.add("where_null")
            if isinstance(node, (exp.And, exp.Or)):
                f.add("where_and_or")
            if isinstance(node, (exp.Not, exp.NEQ)):
                f.add("where_not")
            if isinstance(node, exp.Exists):
                f.add("subquery_exists")

    # scalar subquery used in a comparison
    for node in tree.walk():
        if isinstance(node, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ)):
            for side in (node.this, node.expression):
                if isinstance(side, exp.Subquery) or isinstance(side, exp.Select):
                    f.add("subquery_scalar")

    # -- aggregates
    for node in tree.find_all(exp.Count):
        f.add("count")
        if isinstance(node.this, exp.Distinct):
            f.add("count_distinct")
    if list(tree.find_all(exp.Sum)) or list(tree.find_all(exp.Avg)):
        f.add("agg_sum_avg")
    if list(tree.find_all(exp.Min)) or list(tree.find_all(exp.Max)):
        f.add("agg_min_max")

    # -- joins
    joins = list(tree.find_all(exp.Join))
    join_tables: list[str] = []
    for j in joins:
        side = (j.side or "").upper()
        kind = (j.kind or "").upper()
        if side == "LEFT":
            f.add("left_join")
        elif side == "RIGHT":
            f.add("left_join")  # mirror of left; same pattern family
        elif kind == "CROSS" or (not j.args.get("on") and not j.args.get("using")):
            f.add("cross_join")
        else:
            f.add("inner_join")
        tgt = j.this
        if isinstance(tgt, exp.Table):
            join_tables.append(tgt.name.lower())
    from_tables = [t.name.lower() for t in tree.find_all(exp.Table)]
    if joins:
        if len(set(from_tables)) >= 3 or len(joins) >= 2:
            f.add("multi_join")
        seen: set[str] = set()
        for name in from_tables:
            if name in seen:
                f.add("self_join")
            seen.add(name)
        if "group_by" in f:
            f.add("join_agg")

    # -- subquery in FROM
    for sub in tree.find_all(exp.Subquery):
        parent = sub.parent
        if isinstance(parent, (exp.From, exp.Join)):
            f.add("derived_table")

    # -- correlated subquery: inner select referencing an outer alias
    outer_aliases = set()
    for t in tree.find_all(exp.Table):
        outer_aliases.add((t.alias or t.name).lower())
    for sub in tree.find_all(exp.Subquery):
        inner = sub.this
        if not isinstance(inner, exp.Select):
            continue
        inner_sources = {(t.alias or t.name).lower() for t in inner.find_all(exp.Table)}
        for col in inner.find_all(exp.Column):
            tbl = (col.table or "").lower()
            if tbl and tbl not in inner_sources and tbl in outer_aliases:
                f.add("correlated")

    # -- CTEs / set ops
    for with_ in tree.find_all(exp.With):
        f.add("cte")
        if with_.args.get("recursive"):
            f.add("recursive_cte")
    if list(tree.find_all(exp.Union)):
        f.add("union")
    if list(tree.find_all(exp.Intersect)) or list(tree.find_all(exp.Except)):
        f.add("intersect_except")

    # -- windows
    for w in tree.find_all(exp.Window):
        fn = w.this
        name = fn.sql_name().upper() if hasattr(fn, "sql_name") else ""
        if name in _RANKING_FUNCS:
            f.add("window_ranking")
        else:
            f.add("window_agg")

    # -- expressions
    if list(tree.find_all(exp.Case)):
        f.add("case_when")
    if list(tree.find_all(exp.Coalesce)):
        f.add("coalesce_ifnull")
    for node in tree.walk():
        if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            f.add("arithmetic_expr")
        if isinstance(node, _STRING_FUNC_TYPES):
            f.add("string_function")
        if isinstance(node, exp.Anonymous) and node.name.lower() in _DATE_FUNC_NAMES:
            f.add("date_function")
        if isinstance(node, (exp.Extract,)) or (
            hasattr(exp, "DateTrunc") and isinstance(node, exp.DateTrunc)
        ):
            f.add("date_function")
        if isinstance(node, exp.CurrentDate) or isinstance(node, exp.CurrentTimestamp):
            f.add("date_function")
    return f


# ---------------------------------------------------------------------------
# Generators: feature key -> candidate SQL built from the actual schema
# ---------------------------------------------------------------------------

def _pick_table(schema: Schema) -> Table | None:
    if not schema.tables:
        return None
    # prefer a table that has both numeric and text columns
    ranked = sorted(
        schema.tables.values(),
        key=lambda t: (bool(t.numeric_columns()), bool(t.text_columns()), len(t.columns)),
        reverse=True,
    )
    return ranked[0]


def _cols(table: Table, n=2):
    return [c.name for c in table.columns[:n]]


def generate(schema: Schema, key: str, dialect: str = "sqlite") -> str | None:
    """A plausible business-style query for `key`, or None if the schema
    can't support it (e.g. no foreign keys for a join pattern)."""
    t = _pick_table(schema)
    if t is None:
        return None
    num = t.numeric_columns()
    txt = t.text_columns()
    dat = t.date_columns()
    fk_pairs = schema.fk_pairs()
    chains = schema.fk_chains()
    self_refs = schema.self_referencing()

    def join_clause():
        if not fk_pairs:
            return None
        child, fk, parent = fk_pairs[0]
        return child, fk, parent

    g: str | None = None

    if key == "select_star":
        g = f"SELECT * FROM {t.name}"
    elif key == "select_columns" and len(t.columns) >= 2:
        g = f"SELECT {', '.join(_cols(t, 3))} FROM {t.name}"
    elif key == "distinct" and txt:
        g = f"SELECT DISTINCT {txt[0].name} FROM {t.name}"
    elif key == "order_by":
        col = (num or t.columns)[0].name
        g = f"SELECT * FROM {t.name} ORDER BY {col}"
    elif key == "top_n":
        col = (num or t.columns)[-1].name
        g = f"SELECT * FROM {t.name} ORDER BY {col} DESC LIMIT 5"
    elif key == "limit_offset":
        g = f"SELECT * FROM {t.name} ORDER BY {t.columns[0].name} LIMIT 10 OFFSET 10"
    elif key == "where_equality" and txt:
        g = f"SELECT * FROM {t.name} WHERE {txt[0].name} = 'VALUE'"
    elif key == "where_comparison" and num:
        g = f"SELECT * FROM {t.name} WHERE {num[0].name} > 100"
    elif key == "where_between" and num:
        g = f"SELECT * FROM {t.name} WHERE {num[0].name} BETWEEN 100 AND 500"
    elif key == "where_in_list" and txt:
        g = f"SELECT * FROM {t.name} WHERE {txt[0].name} IN ('A', 'B', 'C')"
    elif key == "where_like" and txt:
        g = f"SELECT * FROM {t.name} WHERE {txt[0].name} LIKE 'A%'"
    elif key == "where_null":
        col = t.columns[-1].name
        g = f"SELECT * FROM {t.name} WHERE {col} IS NULL"
    elif key == "where_and_or" and num and txt:
        g = (f"SELECT * FROM {t.name} WHERE {num[0].name} > 100 "
             f"AND ({txt[0].name} = 'A' OR {txt[0].name} = 'B')")
    elif key == "where_not" and txt:
        g = f"SELECT * FROM {t.name} WHERE {txt[0].name} <> 'VALUE'"
    elif key == "count":
        g = f"SELECT COUNT(*) FROM {t.name}"
    elif key == "agg_sum_avg" and num:
        g = f"SELECT SUM({num[0].name}), AVG({num[0].name}) FROM {t.name}"
    elif key == "agg_min_max" and num:
        g = f"SELECT MIN({num[0].name}), MAX({num[0].name}) FROM {t.name}"
    elif key == "count_distinct" and txt:
        g = f"SELECT COUNT(DISTINCT {txt[0].name}) FROM {t.name}"
    elif key in ("group_by", "having"):
        gt, gcol, agg = None, None, None
        for cand in schema.tables.values():
            grp = cand.text_columns() or [c for c in cand.columns
                                          if c.name in {fk.column for fk in cand.fks}]
            if grp and cand.numeric_columns():
                gt, gcol, agg = cand, grp[0].name, cand.numeric_columns()[0].name
                break
        if gt:
            g = f"SELECT {gcol}, SUM({agg}) FROM {gt.name} GROUP BY {gcol}"
            if key == "having":
                g += f" HAVING SUM({agg}) > 1000"
    elif key == "inner_join" and fk_pairs:
        child, fk, parent = fk_pairs[0]
        g = (f"SELECT p.*, c.* FROM {parent.name} p "
             f"JOIN {child.name} c ON p.{fk.ref_column} = c.{fk.column}")
    elif key == "left_join" and fk_pairs:
        child, fk, parent = fk_pairs[0]
        g = (f"SELECT p.*, c.* FROM {parent.name} p "
             f"LEFT JOIN {child.name} c ON p.{fk.ref_column} = c.{fk.column}")
    elif key == "multi_join" and chains:
        gc, fk1, ch, fk2, par = chains[0]
        g = (f"SELECT a.*, b.*, c.* FROM {gc.name} a "
             f"JOIN {ch.name} b ON a.{fk1.column} = b.{fk1.ref_column} "
             f"JOIN {par.name} c ON b.{fk2.column} = c.{fk2.ref_column}")
    elif key == "self_join" and self_refs:
        st, fk = self_refs[0]
        g = (f"SELECT a.*, b.* FROM {st.name} a "
             f"JOIN {st.name} b ON a.{fk.column} = b.{fk.ref_column}")
    elif key == "cross_join" and len(schema.tables) >= 2:
        names = list(schema.tables)[:2]
        g = f"SELECT * FROM {names[0]} CROSS JOIN {names[1]} LIMIT 20"
    elif key == "join_agg" and fk_pairs:
        for child, fk, parent in fk_pairs:
            cnum = child.numeric_columns()
            plabel = parent.text_columns() or parent.columns
            if cnum:
                g = (f"SELECT p.{plabel[0].name}, SUM(c.{cnum[0].name}) "
                     f"FROM {parent.name} p "
                     f"JOIN {child.name} c ON p.{fk.ref_column} = c.{fk.column} "
                     f"GROUP BY p.{plabel[0].name}")
                break
    elif key == "subquery_in" and fk_pairs:
        child, fk, parent = fk_pairs[0]
        cond = ""
        if child.numeric_columns():
            cond = f" WHERE {child.numeric_columns()[0].name} > 100"
        g = (f"SELECT * FROM {parent.name} WHERE {fk.ref_column} IN "
             f"(SELECT {fk.column} FROM {child.name}{cond})")
    elif key == "subquery_scalar" and num:
        g = (f"SELECT * FROM {t.name} "
             f"WHERE {num[0].name} > (SELECT AVG({num[0].name}) FROM {t.name})")
    elif key == "subquery_exists" and fk_pairs:
        child, fk, parent = fk_pairs[0]
        g = (f"SELECT * FROM {parent.name} p WHERE EXISTS "
             f"(SELECT 1 FROM {child.name} c WHERE c.{fk.column} = p.{fk.ref_column})")
    elif key == "derived_table" and fk_pairs:
        for child, fk, parent in fk_pairs:
            cnum = child.numeric_columns()
            if cnum:
                g = (f"SELECT p.*, s.total FROM {parent.name} p JOIN "
                     f"(SELECT {fk.column}, SUM({cnum[0].name}) AS total "
                     f"FROM {child.name} GROUP BY {fk.column}) s "
                     f"ON p.{fk.ref_column} = s.{fk.column}")
                break
    elif key == "correlated" and fk_pairs:
        for child, fk, parent in fk_pairs:
            cnum = child.numeric_columns()
            if cnum:
                g = (f"SELECT * FROM {child.name} c1 "
                     f"WHERE {cnum[0].name} = (SELECT MAX({cnum[0].name}) "
                     f"FROM {child.name} c2 WHERE c2.{fk.column} = c1.{fk.column})")
                break
    elif key == "cte":
        gg = generate(schema, "group_by", dialect)
        if gg:
            g = f"WITH agg AS ({gg}) SELECT * FROM agg"
    elif key == "recursive_cte" and self_refs:
        st, fk = self_refs[0]
        pk = st.pk[0] if st.pk else st.columns[0].name
        g = (f"WITH RECURSIVE tree AS ("
             f"SELECT * FROM {st.name} WHERE {fk.column} IS NULL "
             f"UNION ALL "
             f"SELECT s.* FROM {st.name} s JOIN tree ON s.{fk.column} = tree.{pk}"
             f") SELECT * FROM tree")
    elif key == "union" and txt:
        g = (f"SELECT {txt[0].name} FROM {t.name} WHERE {txt[0].name} LIKE 'A%' "
             f"UNION SELECT {txt[0].name} FROM {t.name} WHERE {txt[0].name} LIKE 'B%'")
    elif key == "intersect_except" and fk_pairs:
        child, fk, parent = fk_pairs[0]
        g = (f"SELECT {fk.ref_column} FROM {parent.name} EXCEPT "
             f"SELECT {fk.column} FROM {child.name}")
    elif key == "window_ranking" and num:
        part = ""
        if t.fks:
            part = f"PARTITION BY {t.fks[0].column} "
        g = (f"SELECT *, ROW_NUMBER() OVER ({part}ORDER BY {num[0].name} DESC) "
             f"AS rn FROM {t.name}")
    elif key == "window_agg" and num:
        g = (f"SELECT *, SUM({num[0].name}) OVER (ORDER BY {t.columns[0].name}) "
             f"AS running_total FROM {t.name}")
    elif key == "case_when" and num:
        g = (f"SELECT *, CASE WHEN {num[0].name} > 1000 THEN 'high' "
             f"WHEN {num[0].name} > 100 THEN 'medium' ELSE 'low' END AS bucket "
             f"FROM {t.name}")
    elif key == "date_function":
        dt = None
        for cand in schema.tables.values():
            if cand.date_columns():
                dt = cand
                break
        if dt:
            col = dt.date_columns()[0].name
            if dialect == "postgres":
                g = (f"SELECT DATE_TRUNC('month', {col}) AS month, COUNT(*) "
                     f"FROM {dt.name} GROUP BY 1 ORDER BY 1")
            else:
                g = (f"SELECT STRFTIME('%Y-%m', {col}) AS month, COUNT(*) "
                     f"FROM {dt.name} GROUP BY 1 ORDER BY 1")
    elif key == "string_function" and txt:
        g = f"SELECT UPPER({txt[0].name}) FROM {t.name}"
    elif key == "arithmetic_expr" and num:
        g = f"SELECT {num[0].name}, {num[0].name} * 1.1 AS adjusted FROM {t.name}"
    elif key == "coalesce_ifnull":
        col = t.columns[-1].name
        g = f"SELECT COALESCE({col}, 'N/A') FROM {t.name}"
    return g


def coverage(feature_sets: list[set[str]]):
    """Aggregate: {feature: count} plus missing list."""
    counts: dict[str, int] = {k: 0 for k in TAXONOMY}
    for fs in feature_sets:
        for k in fs:
            if k in counts:
                counts[k] += 1
    covered = {k for k, v in counts.items() if v > 0}
    missing = [k for k in TAXONOMY if k not in covered]
    return counts, missing
