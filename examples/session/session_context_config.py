# /// script
# description = "Session context config - execution settings, planning config, and IO config"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8"]
# ///

import daft
from daft.io import IOConfig, S3Config

PUBLIC_PARQUET = "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-6f24a7497df494ae.parquet"


def anonymous_s3_io_config() -> IOConfig:
    """Return an IOConfig for anonymous access to public S3 datasets."""
    return IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))


def enable_dynamic_batching() -> None:
    """Enable dynamic batching globally to optimize batch sizes for AI/GPU workloads."""
    daft.set_execution_config(enable_dynamic_batching=True)


def set_default_io_config(io_config: IOConfig) -> None:
    """Set a default IOConfig so every read uses it without passing io_config= explicitly."""
    daft.set_planning_config(default_io_config=io_config)
    daft.read_parquet(PUBLIC_PARQUET).limit(5).show()


def scoped_execution_config() -> None:
    """Apply execution config for a single block, then restore the previous settings."""
    with daft.execution_config_ctx(enable_dynamic_batching=True, num_preview_rows=3):
        daft.from_pydict({"x": list(range(10))}).show()


def scoped_planning_config(io_config: IOConfig) -> None:
    """Apply planning config for a single block, then restore the previous settings."""
    with daft.planning_config_ctx(default_io_config=io_config):
        daft.read_parquet(PUBLIC_PARQUET).limit(3).show()


def fine_tuned_execution_config() -> None:
    """Set multiple execution knobs at once for advanced tuning."""
    daft.set_execution_config(
        enable_dynamic_batching=True,
        num_preview_rows=5,
        shuffle_aggregation_default_partitions=16,
        default_morsel_size=1024,
    )


def enable_strict_filter_pushdown() -> None:
    """Enable strict filter pushdown for Parquet/Delta/Iceberg predicate optimization."""
    daft.set_planning_config(enable_strict_filter_pushdown=True)


if __name__ == "__main__":
    io_config = anonymous_s3_io_config()

    enable_dynamic_batching()
    set_default_io_config(io_config)
    scoped_execution_config()
    scoped_planning_config(io_config)
    fine_tuned_execution_config()
    enable_strict_filter_pushdown()
