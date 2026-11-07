"""Regression tests for the review fixes. Run: python -m pytest test_sqlpattern.py  (or python test_sqlpattern.py)"""
from sqlpattern import sql_fingerprint as fp, PatternIndex


def same(a, b, level="skeleton"):
    return fp(a, level)[0] == fp(b, level)[0]


def test_and_order_independent():
    assert same("SELECT * FROM e WHERE age>30 AND name='x'", "SELECT * FROM e WHERE name='x' AND age>30")
    assert same("SELECT a FROM t WHERE x>1 AND y>1 AND a>1", "SELECT a FROM t WHERE a>1 AND x>1 AND y>1")
    assert same("SELECT * FROM t WHERE a=1 OR b>2", "SELECT * FROM t WHERE b>2 OR a=1")


def test_and_order_independent_template_level():
    assert same("SELECT * FROM e WHERE age>30 AND name='x'", "SELECT * FROM e WHERE name='x' AND age>30", "template")


def test_different_structure_still_different():
    assert not same("SELECT * FROM t WHERE a>1 AND b>1", "SELECT * FROM t WHERE a>1 OR b>1")
    assert not same("SELECT * FROM t WHERE a>1", "SELECT * FROM t WHERE a>1 AND b>1")


def test_select_column_order_by_design():
    # names are anonymised at skeleton level -> same pattern...
    assert same("SELECT a,b FROM t", "SELECT b,a FROM t")
    # ...but column roles elsewhere in the query are preserved
    assert not same("SELECT a,b FROM t WHERE b>1", "SELECT b,a FROM t WHERE b>1")


def test_regex_fallback_never_collides_with_ast():
    canon, ok = fp("SELECT FROM WHERE ((", "skeleton")
    assert not ok and canon.startswith("regex:")


def test_strict_keeps_distinct_unparseable():
    idx = PatternIndex(strict=True)
    k1 = idx.key_for("SELECT FROM WHERE ((a")[0]
    k2 = idx.key_for("SELECT FROM WHERE ((b")[0]
    k3 = idx.key_for("SELECT  FROM WHERE ((a")[0]  # whitespace-only difference
    assert k1 != k2 and k1 == k3
    assert idx.parse_failures == 3 and len(idx.parse_failure_examples) == 3


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
