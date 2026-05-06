# /// script
# description = "Session SQL - query attached tables with SQL through a session"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10"]
# ///

import daft
from daft import Catalog, Session


def build_company_session() -> Session:
    """Build a session with a company catalog of employees and departments."""
    catalog = Catalog.from_pydict(
        {
            "employees": {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "department": ["Engineering", "Sales", "Engineering", "Sales", "Engineering"],
                "salary": [120_000, 85_000, 110_000, 90_000, 130_000],
            },
            "departments": {
                "name": ["Engineering", "Sales", "Marketing"],
                "budget": [500_000, 300_000, 200_000],
            },
        },
        name="company",
    )
    sess = Session()
    sess.attach_catalog(catalog)
    sess.set_catalog("company")
    return sess


def basic_query(sess: Session) -> None:
    """Run a basic SELECT/WHERE/ORDER BY against an attached table."""
    sess.sql("SELECT * FROM employees WHERE department = 'Engineering' ORDER BY salary DESC").show()


def distinct_query(sess: Session) -> None:
    """Use DISTINCT to find unique values in a column."""
    sess.sql("SELECT DISTINCT department FROM employees").show()


def join_query(sess: Session) -> None:
    """Join two catalog tables with SQL."""
    sess.sql("""
        SELECT e.name, e.department, e.salary, d.budget
        FROM employees e
        JOIN departments d ON e.department = d.name
        ORDER BY e.salary DESC
    """).show()


def global_sql(sess: Session) -> None:
    """Use module-level daft.sql against the active session."""
    with sess:
        daft.sql("SELECT name, salary FROM employees WHERE salary > 100000").show()


def sql_with_temp_table(sess: Session) -> None:
    """Join a session-scoped temp table against catalog tables in SQL."""
    sess.create_temp_table(
        "bonuses",
        daft.from_pydict({"employee_id": [1, 3, 5], "bonus": [10_000, 8_000, 15_000]}),
    )
    sess.sql("""
        SELECT e.name, e.salary, b.bonus, (e.salary + b.bonus) AS total_comp
        FROM employees e
        JOIN bonuses b ON e.id = b.employee_id
        ORDER BY total_comp DESC
    """).show()


if __name__ == "__main__":
    sess = build_company_session()
    basic_query(sess)
    distinct_query(sess)
    join_query(sess)
    global_sql(sess)
    sql_with_temp_table(sess)
