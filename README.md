# sql-dedup — NL→SQL dataset workbench

> Full illustrated guide with architecture diagrams: open [README.html](README.html) in a browser.
> The CLI keeps its file names: `sqlds.py` (workbench) and `sqlpattern.py` (standalone dedupe).

An end-to-end toolkit for building a **distinct, verified** natural-language → SQL
training dataset (SFT / RL). It solves three problems:

1. **Duplicate patterns** — you may write 2000 SQL queries but 600 are just the
   same query shape with different values/names. Those add nothing to training.
   Every pair is fingerprinted by SQL *structure* and duplicates are dropped.
2. **Coverage gaps** — a taxonomy of ~45 PostgreSQL/SQLite syntax patterns
   (joins, aggregates, subqueries, CTEs, window functions, set ops, …) tracks
   what your dataset already covers, and generates candidate SQL **from your
   actual schema** for what's missing.
3. **Validation** — every new SQL runs against your real database. You see
   the row count of each referenced table plus the actual result rows (e.g. the
   customer↔account join output), so you can confirm the query does what the NL
   says. Runs are always rolled back — verification never modifies data.

Requirements: Python 3.10+, `sqlglot` (required for accurate fingerprints),
`psycopg` or `psycopg2` only if you use PostgreSQL.

## Quick start

```bash
# 1. point the workspace at your DB (SQLite file or postgres DSN)
python sqlds.py init --db demo_bank.db
#    or: python sqlds.py init --db postgresql://user:pass@host:5432/mydb

# 2. bulk-load anything you already have (jsonl / json / csv; fields auto-detected)
python sqlds.py import existing_pairs.jsonl --verify

# 3. author new pairs one by one — duplicate check + live verification
python sqlds.py add --nl "Total balance per city for premium customers" \
  --sql "SELECT c.city, SUM(a.balance) FROM customers c JOIN accounts a ON c.id=a.customer_id WHERE c.segment='premium' GROUP BY c.city"

# 4. what kinds of SQL are still missing?
python sqlds.py status
python sqlds.py gaps

# 5. generate runnable candidate SQL from your schema for every gap
python sqlds.py suggest            # -> candidates.jsonl (each pre-executed)
#    fill in the "question" fields, adapt values to real business terms, then:
python sqlds.py import candidates.jsonl --verify

# 6. sanity-check any SQL by hand (row counts + join result, never commits)
python sqlds.py verify --sql "SELECT ... FROM customers c JOIN accounts a ON ..."

# 7. ship it
python sqlds.py export --verified-only     # -> sft_dataset.jsonl  {question, sql}
```

Other commands: `schema` (tables/columns/keys/row counts), `coverage`
(per-feature query counts).

## How duplicate detection works (`sqlpattern.py`)

Each SQL is parsed to an AST (sqlglot) and reduced to a canonical **skeleton**:

| level      | what is normalized                          | example collapse |
|------------|---------------------------------------------|------------------|
| `exact`    | whitespace, case, comments, trailing `;`    | `select *` = `SELECT  *` |
| `template` | + literals masked, `IN (1,2,3)` → `IN (?)`  | `age > 21` = `age > 99` |
| `skeleton` | + identifiers renamed by first appearance   | `name FROM users WHERE age>21` = `title FROM books WHERE price>5` |

Default is `skeleton` (set in `sqlds.json`), which is what "same *kind* of
query" usually means for training diversity. Two pairs with the same skeleton
are duplicates — the second is rejected (`add` exits with code 3; `--force`
overrides). Unparseable SQL falls back to a regex-based normalizer.

`sqlpattern.py` is also a standalone tool for one-off jobs on files, without a
workspace:

```bash
python sqlpattern.py stats data.jsonl                  # dup % at each level
python sqlpattern.py dedupe data.jsonl --report report.json
python sqlpattern.py build-index data.jsonl -o index.json
python sqlpattern.py check --index index.json --sql "SELECT ..."   # NEW / DUPLICATE
```

Optional: `--nl-mode fuzzy` treats a matching SQL skeleton as a duplicate only
when the natural-language questions are also similar (token Jaccard ≥
`--nl-threshold`), for cases where one SQL shape legitimately answers several
distinct business questions.

## Files

| file              | role |
|-------------------|------|
| `sqlds.py`        | end-to-end CLI (init / add / import / gaps / suggest / verify / export) |
| `sqlpattern.py`   | SQL structural fingerprinting + standalone dedupe CLI |
| `sqlfeatures.py`  | syntax taxonomy: feature detectors + schema-aware generators |
| `sqldb.py`        | SQLite/Postgres connection, schema introspection, safe (rollback) runner |
| `sqlds.json`      | workspace config: db, dataset file, dedupe level |
| `dataset.jsonl`   | the growing dataset: `{question, sql, features, verified, added_at}` |
| `make_demo_db.py` | builds `demo_bank.db` (branches/employees/customers/accounts/transactions) to try everything |

## Notes

- Verification always runs in a transaction that is **rolled back**; use
  `verify --commit` only when you explicitly want DML applied.
- `suggest` skips INSERT/UPDATE/DELETE generation on purpose — write DML pairs
  by hand and verify them with `verify` (rollback keeps the DB safe).
- Generated candidates use placeholder values (`'VALUE'`, `> 100`); replace
  them with realistic business values before adding, and reword the SQL freely —
  the dedupe check is structural, so cosmetic edits won't create false "new"
  patterns.

## License and credits

- The code is released under the [MIT License](LICENSE).
- The sample queries in `queries_221.json`, `queries_300.jsonl` and the
  `bird_*` reports are taken from the
  [BIRD benchmark](https://bird-bench.github.io/) (Li et al., 2023,
  "Can LLM Already Serve as A Database Interface? A BIg Bench for
  Large-Scale Database Grounded Text-to-SQLs"), which is licensed under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Those files remain under CC BY-SA 4.0, not MIT.
