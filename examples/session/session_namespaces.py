# /// script
# description = "Session namespaces - organize tables into logical groups within a catalog"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8"]
# ///

import daft
from daft import Catalog, Identifier, Session


def build_warehouse_session() -> Session:
    """Attach an empty in-memory catalog to a session for namespace demos."""
    sess = Session()
    sess.attach_catalog(Catalog.from_pydict({}, name="warehouse"))
    sess.set_catalog("warehouse")
    return sess


def create_namespaces(sess: Session) -> None:
    """Create namespaces to group related tables, like schemas in a database."""
    sess.create_namespace("sales")
    sess.create_namespace("marketing")
    sess.create_namespace_if_not_exists("sales")  # idempotent
    print(f"namespaces: {sess.list_namespaces()}")


def create_tables_in_namespaces(sess: Session) -> None:
    """Create tables with dotted identifiers that place them inside a namespace."""
    sess.create_table(
        "sales.orders",
        daft.from_pydict({"order_id": [1, 2, 3], "amount": [99.99, 149.50, 29.99]}),
    )
    sess.create_table(
        "sales.customers",
        daft.from_pydict({"customer_id": [1, 2], "name": ["Alice", "Bob"]}),
    )
    sess.create_table(
        "marketing.campaigns",
        daft.from_pydict({"campaign_id": [100, 101], "name": ["Spring", "Summer"]}),
    )
    print(f"tables: {sess.list_tables()}")


def read_fully_qualified(sess: Session) -> None:
    """Read tables by their fully-qualified namespace.table name."""
    sess.read_table("sales.orders").show()
    sess.read_table("marketing.campaigns").show()


def set_active_namespace(sess: Session) -> None:
    """Set an active namespace so unqualified names resolve within it."""
    sess.set_namespace("sales")
    print(f"active namespace: {sess.current_namespace()}")
    sess.read_table("orders").show()  # resolves to sales.orders
    sess.set_namespace(None)
    print(f"active namespace: {sess.current_namespace()}")


def read_with_identifier(sess: Session) -> None:
    """Use Identifier objects for programmatic multi-part table names."""
    ident = Identifier("sales", "orders")
    print(f"identifier: {ident!r}  parts: {list(ident)}")
    sess.read_table(ident).show()


def existence_checks(sess: Session) -> None:
    """Check for namespaces and tables before creating or dropping them."""
    print(f"has_namespace('sales'):     {sess.has_namespace('sales')}")
    print(f"has_namespace('missing'):   {sess.has_namespace('missing')}")
    print(f"has_table('sales.orders'):  {sess.has_table('sales.orders')}")


def pattern_filtered_listing(sess: Session) -> None:
    """Filter catalog listings with a glob pattern."""
    print(f"tables matching 'sales.*': {sess.list_tables('sales.*')}")


def drop_namespace_and_tables(sess: Session) -> None:
    """Drop tables first, then their parent namespace."""
    sess.drop_table("marketing.campaigns")
    sess.drop_namespace("marketing")
    print(f"namespaces after drop: {sess.list_namespaces()}")
    print(f"tables after drop:     {sess.list_tables()}")


def backend_support_notes() -> None:
    """Namespace operations vary by backend.

    Support matrix:
        In-memory (from_pydict):    full support
        Iceberg, Glue, Postgres:    full support
        Unity Catalog:              read-only (use pre-existing schemas)
    """


if __name__ == "__main__":
    sess = build_warehouse_session()
    create_namespaces(sess)
    create_tables_in_namespaces(sess)
    read_fully_qualified(sess)
    set_active_namespace(sess)
    read_with_identifier(sess)
    existence_checks(sess)
    pattern_filtered_listing(sess)
    drop_namespace_and_tables(sess)
    backend_support_notes()
