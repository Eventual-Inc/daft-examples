# /// script
# description = "Native extension via load_extension() - daft-h3 for H3 geospatial indexing"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10", "daft-h3"]
# ///

import daft_h3

import daft
from daft import Session, col


def load_h3_extension() -> Session:
    """Register the daft-h3 native extension with a session.

    load_extension takes the extension's Python module, links its shared
    library, and makes its Rust-native functions available inside the session.
    """
    sess = Session()
    sess.load_extension(daft_h3)
    return sess


def cities() -> daft.DataFrame:
    """Sample DataFrame of cities with latitude/longitude."""
    return daft.from_pydict(
        {
            "city": ["San Francisco", "Paris", "Tokyo", "New York"],
            "lat": [37.7749, 48.8566, 35.6762, 40.7128],
            "lng": [-122.4194, 2.3522, 139.6503, -74.0060],
        }
    )


def latlng_to_cells(df: daft.DataFrame) -> daft.DataFrame:
    """Encode lat/lng pairs as H3 cells at two different resolutions."""
    return df.select(
        col("city"),
        daft_h3.h3_latlng_to_cell(col("lat"), col("lng"), 7).alias("cell_r7"),
        daft_h3.h3_latlng_to_cell(col("lat"), col("lng"), 9).alias("cell_r9"),
    )


def cells_to_hex(df: daft.DataFrame) -> daft.DataFrame:
    """Convert UInt64 cell indices to human-readable hex strings.

    Operate on UInt64 in hot paths; convert to hex only for display or export.
    """
    return df.select(
        col("city"),
        daft_h3.h3_cell_to_str(col("cell_r7")).alias("hex_r7"),
        daft_h3.h3_cell_to_str(col("cell_r9")).alias("hex_r9"),
    )


def inspect_cells(df: daft.DataFrame) -> daft.DataFrame:
    """Derive resolution, validity, and center lat/lng from cell indices."""
    return df.select(
        col("city"),
        col("hex_r9"),
        daft_h3.h3_cell_resolution(col("hex_r9")).alias("resolution"),
        daft_h3.h3_cell_is_valid(col("hex_r9")).alias("is_valid"),
        daft_h3.h3_cell_to_lat(col("hex_r9")).alias("center_lat"),
        daft_h3.h3_cell_to_lng(col("hex_r9")).alias("center_lng"),
    )


def rollup_to_parents(df: daft.DataFrame) -> daft.DataFrame:
    """Climb the H3 hierarchy to coarser resolutions for multi-zoom aggregation."""
    return df.select(
        col("city"),
        col("hex_r9"),
        daft_h3.h3_cell_to_str(daft_h3.h3_cell_parent(col("hex_r9"), 5)).alias("parent_r5"),
        daft_h3.h3_cell_to_str(daft_h3.h3_cell_parent(col("hex_r9"), 3)).alias("parent_r3"),
    )


def grid_distance() -> daft.DataFrame:
    """Compute hop distance between H3 cells at the same resolution."""
    pairs = daft.from_pydict(
        {
            "from_city": ["SF", "SF", "NYC"],
            "to_city": ["NYC", "LA", "LA"],
            "from_hex": ["8928308280fffff", "8928308280fffff", "89283470d93ffff"],
            "to_hex": ["89283470d93ffff", "8928347606bffff", "8928347606bffff"],
        }
    )
    return pairs.select(
        col("from_city"),
        col("to_city"),
        daft_h3.h3_grid_distance(col("from_hex"), col("to_hex")).alias("hops"),
    )


if __name__ == "__main__":
    sess = load_h3_extension()
    with sess:
        df = latlng_to_cells(cities())
        df.show()

        df = cells_to_hex(df)
        df.show()

        inspect_cells(df).show()
        rollup_to_parents(df).show()
        grid_distance().show()
