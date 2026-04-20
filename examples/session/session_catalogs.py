# /// script
# description = "Session catalogs - attach catalogs, create tables, and manage existence"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8"]
# ///

import daft
from daft import Catalog, Session


def build_shop_catalog() -> Catalog:
    """Wrap a dict of tables into a named in-memory Catalog."""
    return Catalog.from_pydict(
        {
            "users": {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]},
            "orders": {"order_id": [10, 20], "user_id": [1, 2], "amount": [99.99, 149.50]},
        },
        name="shop",
    )


def attach_to_session(catalog: Catalog) -> Session:
    """Attach a catalog to a session and set it as active."""
    sess = Session()
    sess.attach_catalog(catalog)
    sess.set_catalog("shop")
    sess.read_table("users").show()
    return sess


def attach_globally(catalog: Catalog) -> None:
    """Attach a catalog to the global default session."""
    daft.attach_catalog(catalog)
    daft.set_catalog("shop")
    daft.read_table("users").show()


def create_table(sess: Session) -> None:
    """Create a new persistent table in the active catalog from a DataFrame."""
    tiers = daft.from_pydict({"user_id": [1, 2, 3], "tier": ["gold", "silver", "bronze"]})
    sess.create_table("tiers", tiers)
    sess.read_table("tiers").show()


def create_temp_table(sess: Session) -> None:
    """Create a session-scoped temp table that doesn't persist to the catalog."""
    scratch = daft.from_pydict({"x": [1, 2, 3], "y": [4, 5, 6]})
    sess.create_temp_table("scratch", scratch)
    sess.read_table("scratch").show()


def existence_checks(sess: Session) -> None:
    """Check for catalogs and tables before acting on them."""
    print(f"has_catalog('shop'):   {sess.has_catalog('shop')}")
    print(f"has_table('users'):    {sess.has_table('users')}")
    print(f"has_table('missing'):  {sess.has_table('missing')}")


def external_catalog_factories() -> None:
    """Reference for connecting Daft to external catalog systems.

    Each factory requires real credentials - shown here as documentation:

        Catalog.from_unity(unity_client)                      # Databricks Unity Catalog
        Catalog.from_iceberg(pyiceberg_catalog)               # Apache Iceberg (PyIceberg)
        Catalog.from_glue("my_glue_db")                       # AWS Glue
        Catalog.from_s3tables(table_bucket_arn="arn:aws:..")  # AWS S3 Tables
        Catalog.from_postgres("postgresql://user:pass@host")  # PostgreSQL
        Catalog.from_gravitino(endpoint="http://...")         # Apache Gravitino
        Catalog.from_paimon(paimon_catalog)                   # Apache Paimon
    """


if __name__ == "__main__":
    catalog = build_shop_catalog()
    sess = attach_to_session(catalog)
    attach_globally(catalog)
    create_table(sess)
    create_temp_table(sess)
    existence_checks(sess)
    external_catalog_factories()
