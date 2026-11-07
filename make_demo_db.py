"""Create demo_bank.db - a small banking schema to try the workbench on."""
import random
import sqlite3
from pathlib import Path

random.seed(7)
path = Path(__file__).parent / "demo_bank.db"
path.unlink(missing_ok=True)
conn = sqlite3.connect(path)
conn.executescript(
    """
    PRAGMA foreign_keys = ON;

    CREATE TABLE branches (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT NOT NULL,
        manager_id INTEGER REFERENCES employees(id)
    );

    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        title TEXT,
        supervisor_id INTEGER REFERENCES employees(id),
        branch_id INTEGER REFERENCES branches(id)
    );

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT,
        segment TEXT,             -- retail / premium / corporate
        joined_on DATE,
        branch_id INTEGER REFERENCES branches(id)
    );

    CREATE TABLE accounts (
        id INTEGER PRIMARY KEY,
        acct_no TEXT NOT NULL,
        type TEXT NOT NULL,       -- checking / savings / loan
        balance NUMERIC NOT NULL,
        opened_on DATE,
        customer_id INTEGER NOT NULL REFERENCES customers(id)
    );

    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES accounts(id),
        txn_date DATE NOT NULL,
        amount NUMERIC NOT NULL,
        kind TEXT NOT NULL        -- deposit / withdrawal / fee
    );
    """
)

cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Pune"]
for i, city in enumerate(cities, 1):
    conn.execute("INSERT INTO branches (id, name, city) VALUES (?, ?, ?)",
                 (i, f"{city} Main", city))

names = ["Asha", "Ravi", "Meera", "Karan", "Divya", "Suresh", "Priya", "Amit",
         "Neha", "Vikram", "Sunita", "Rahul"]
for i, name in enumerate(names, 1):
    supervisor = None if i <= 2 else random.randint(1, 2)
    conn.execute(
        "INSERT INTO employees (id, name, title, supervisor_id, branch_id) VALUES (?,?,?,?,?)",
        (i, name, "Manager" if i <= 2 else "Officer", supervisor, (i % 5) + 1),
    )
conn.execute("UPDATE branches SET manager_id = 1 WHERE id IN (1,2,3)")
conn.execute("UPDATE branches SET manager_id = 2 WHERE id IN (4,5)")

segments = ["retail", "retail", "retail", "premium", "corporate"]
for i in range(1, 41):
    conn.execute(
        "INSERT INTO customers (id, name, city, segment, joined_on, branch_id) VALUES (?,?,?,?,?,?)",
        (i, f"Customer {i:02d}", random.choice(cities), random.choice(segments),
         f"20{random.randint(15, 24)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
         random.randint(1, 5)),
    )

acct_id = 0
for cust in range(1, 41):
    for _ in range(random.randint(1, 3)):
        acct_id += 1
        conn.execute(
            "INSERT INTO accounts (id, acct_no, type, balance, opened_on, customer_id) "
            "VALUES (?,?,?,?,?,?)",
            (acct_id, f"AC{acct_id:05d}", random.choice(["checking", "savings", "loan"]),
             round(random.uniform(-5000, 250000), 2),
             f"20{random.randint(16, 25)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
             cust),
        )

txn_id = 0
for acct in range(1, acct_id + 1):
    for _ in range(random.randint(2, 8)):
        txn_id += 1
        conn.execute(
            "INSERT INTO transactions (id, account_id, txn_date, amount, kind) VALUES (?,?,?,?,?)",
            (txn_id, acct,
             f"2025-{random.randint(1, 9):02d}-{random.randint(1, 28):02d}",
             round(random.uniform(-20000, 50000), 2),
             random.choice(["deposit", "withdrawal", "fee"])),
        )

conn.commit()
conn.close()
print(f"created {path.name}: 5 branches, 12 employees, 40 customers, "
      f"{acct_id} accounts, {txn_id} transactions")
