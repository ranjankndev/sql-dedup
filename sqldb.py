#!/usr/bin/env python3
"""
sqldb - database access for the SQL dataset workbench.

Supports:
  * SQLite  - db spec is a file path, or sqlite:///path/to/file.db
  * Postgres- db spec is a DSN: postgresql://user:pass@host:port/dbname
              (uses psycopg 3 if installed, else psycopg2)

Provides schema introspection (tables, columns, foreign keys) and a safe
query runner used for verification: every statement runs inside a
transaction that is rolled back unless commit=True.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

NUMERIC_TYPES = re.compile(r"int|real|float|double|decimal|numeric|money|serial", re.I)
TEXT_TYPES = re.compile(r"char|text|clob|string|uuid|enum", re.I)
DATE_TYPES = re.compile(r"date|time|timestamp|interval", re.I)


@dataclass
class Column:
    name: str
    type: str

    @property
    def is_numeric(self) -> bool:
        return bool(NUMERIC_TYPES.search(self.type or ""))

    @property
    def is_text(self) -> bool:
        return bool(TEXT_TYPES.search(self.type or ""))

    @property
    def is_date(self) -> bool:
        return bool(DATE_TYPES.search(self.type or ""))


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    pk: list[str] = field(default_factory=list)
    fks: list[ForeignKey] = field(default_factory=list)

    def numeric_columns(self, exclude_keys=True):
        keys = set(self.pk) | {fk.column for fk in self.fks}
        return [c for c in self.columns
                if c.is_numeric and not (exclude_keys and c.name in keys)]

    def text_columns(self):
        return [c for c in self.columns if c.is_text]

    def date_columns(self):
        return [c for c in self.columns if c.is_date]


@dataclass
class Schema:
    tables: dict[str, Table] = field(default_factory=dict)

    def fk_pairs(self):
        """[(child_table, fk, parent_table)] for every foreign key."""
        out = []
        for t in self.tables.values():
            for fk in t.fks:
                if fk.ref_table in self.tables:
                    out.append((t, fk, self.tables[fk.ref_table]))
        return out

    def self_referencing(self):
        return [(t, fk) for t in self.tables.values()
                for fk in t.fks if fk.ref_table == t.name]

    def fk_chains(self):
        """[(grandchild, fk1, child, fk2, parent)] two-hop join paths."""
        out = []
        pairs = self.fk_pairs()
        for child, fk1, parent in pairs:
            for child2, fk2, parent2 in pairs:
                if child2.name == parent.name and parent2.name != child.name:
                    out.append((child, fk1, parent, fk2, parent2))
        return out


class Database:
    """Thin wrapper over sqlite3 / psycopg with introspection + safe runs."""

    def __init__(self, spec: str):
        self.spec = spec
        if spec.startswith(("postgresql://", "postgres://")):
            self.kind = "postgres"
            self.dialect = "postgres"
        else:
            self.kind = "sqlite"
            self.dialect = "sqlite"
        self._conn = None

    # -- connection --------------------------------------------------------
    def connect(self):
        if self._conn is not None:
            return self._conn
        if self.kind == "sqlite":
            path = self.spec
            if path.startswith("sqlite:///"):
                path = path[len("sqlite:///"):]
            self._conn = sqlite3.connect(path)
            self._conn.execute("PRAGMA foreign_keys = ON")
        else:
            try:
                import psycopg
                self._conn = psycopg.connect(self.spec)
            except ImportError:
                import psycopg2
                self._conn = psycopg2.connect(self.spec)
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- introspection -----------------------------------------------------
    def schema(self) -> Schema:
        return self._schema_sqlite() if self.kind == "sqlite" else self._schema_postgres()

    def _schema_sqlite(self) -> Schema:
        conn = self.connect()
        schema = Schema()
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (tname,) in rows:
            table = Table(name=tname)
            for _, cname, ctype, _notnull, _default, pk in conn.execute(
                f"PRAGMA table_info('{tname}')"
            ):
                table.columns.append(Column(cname, ctype or ""))
                if pk:
                    table.pk.append(cname)
            for row in conn.execute(f"PRAGMA foreign_key_list('{tname}')"):
                # (id, seq, ref_table, from_col, to_col, ...)
                ref_table, from_col, to_col = row[2], row[3], row[4]
                table.fks.append(ForeignKey(from_col, ref_table, to_col or "id"))
            schema.tables[tname] = table
        return schema

    def _schema_postgres(self) -> Schema:
        conn = self.connect()
        schema = Schema()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """
        )
        for tname, cname, ctype in cur.fetchall():
            schema.tables.setdefault(tname, Table(name=tname)).columns.append(
                Column(cname, ctype)
            )
        cur.execute(
            """
            SELECT tc.table_name, kcu.column_name, tc.constraint_type,
                   ccu.table_name, ccu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            LEFT JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema = ccu.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')
            """
        )
        for tname, cname, ctype, ref_t, ref_c in cur.fetchall():
            table = schema.tables.get(tname)
            if table is None:
                continue
            if ctype == "PRIMARY KEY":
                table.pk.append(cname)
            elif ctype == "FOREIGN KEY" and ref_t:
                table.fks.append(ForeignKey(cname, ref_t, ref_c or "id"))
        cur.close()
        conn.rollback()
        return schema

    # -- execution ---------------------------------------------------------
    def row_count(self, table: str) -> int | None:
        try:
            cur = self.connect().cursor()
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            n = cur.fetchone()[0]
            cur.close()
            return n
        except Exception:
            try:
                self.connect().rollback()
            except Exception:
                pass
            return None

    def run(self, sql: str, limit: int = 20, commit: bool = False):
        """
        Execute one statement in a transaction.

        Returns dict:
          ok, error, is_select, columns, rows (first `limit`), total_rows,
          affected (for DML), committed
        """
        conn = self.connect()
        result = {
            "ok": False, "error": None, "is_select": False, "columns": [],
            "rows": [], "total_rows": None, "affected": None, "committed": False,
        }
        cur = conn.cursor()
        try:
            cur.execute(sql)
            if cur.description:  # produced a result set
                result["is_select"] = True
                result["columns"] = [d[0] for d in cur.description]
                rows = cur.fetchmany(limit)
                result["rows"] = [tuple(r) for r in rows]
                extra = 0
                while True:
                    chunk = cur.fetchmany(1000)
                    if not chunk:
                        break
                    extra += len(chunk)
                result["total_rows"] = len(result["rows"]) + extra
            else:
                result["affected"] = cur.rowcount
            result["ok"] = True
            if commit:
                conn.commit()
                result["committed"] = True
            else:
                conn.rollback()  # verification runs never change the DB
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            cur.close()
        return result


def referenced_tables(sql: str, dialect: str | None = None) -> list[str]:
    """Table names referenced by a query (via sqlglot; regex fallback)."""
    try:
        import sqlglot
        from sqlglot import exp

        names = []
        for stmt in sqlglot.parse(sql, read=dialect):
            if stmt is None:
                continue
            cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
            for t in stmt.find_all(exp.Table):
                name = t.name
                if name and name.lower() not in cte_names and name not in names:
                    names.append(name)
        return names
    except Exception:
        found = re.findall(r"\b(?:from|join|into|update)\s+([A-Za-z_][\w$]*)", sql, re.I)
        out = []
        for name in found:
            if name.lower() not in ("select",) and name not in out:
                out.append(name)
        return out
