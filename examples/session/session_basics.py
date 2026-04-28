# /// script
# description = "Session basics - creation, context manager, and state management"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8"]
# ///

import daft
from daft import Session, Table


def explicit_session() -> None:
    """Create a standalone session. Attach/read operations are scoped to it and don't touch the global default."""
    sess = Session()
    sess.attach_table(
        Table.from_df("students", daft.from_pydict({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})),
    )
    sess.read_table("students").show()


def context_manager_session() -> None:
    """Use a session as a context manager to set it as the active global session for the block."""
    with daft.session() as sess:
        sess.attach_table(
            Table.from_df(
                "cities",
                daft.from_pydict({"city": ["NYC", "SF", "LA"], "pop": [8_336_817, 808_437, 3_898_747]}),
            ),
        )
        # daft.read_table resolves through the active session
        daft.read_table("cities").show()


def global_session() -> None:
    """Operate on the default global session via module-level helpers."""
    daft.attach_table(
        Table.from_df(
            "languages",
            daft.from_pydict({"lang": ["Python", "Rust", "SQL"], "year": [1991, 2010, 1974]}),
        ),
    )
    daft.read_table("languages").show()
    daft.detach_table("languages")


if __name__ == "__main__":
    explicit_session()
    context_manager_session()
    global_session()
