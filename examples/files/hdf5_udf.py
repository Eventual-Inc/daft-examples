# /// script
# description = "Read a HDF5 file and return a struct of the requested fields"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[hdf5]>=0.7.16"]
# ///
import h5py

import daft
from daft import DataType, Hdf5File, col


# Build the UDF that will read the trajectory data and return a struct of the requested fields
@daft.func(
    return_dtype=DataType.struct({
        "action/gripper_position": DataType.tensor(DataType.float64()),
        "action/target_gripper_position": DataType.tensor(DataType.float64()),
        "observation/robot_state/gripper_position": DataType.tensor(DataType.float64()),
    }),
    use_process=False,
    unnest=True,
)
def read_droid_trajectory(file: Hdf5File):
    with file.to_tempfile() as tmp, h5py.File(tmp.name, "r") as h5:
        return {
            "action/gripper_position": h5["action/gripper_position"][()],
            "action/target_gripper_position": h5["action/target_gripper_position"][()],
            "observation/robot_state/gripper_position": h5["observation/robot_state/gripper_position"][()],
        }

if __name__ == "__main__":

    df = (
        daft.datasets.droid.raw()
        .where(col("success")) # filter out failed episodes
        .select(col("current_task"), read_droid_trajectory(col("trajectory")))
    )

    df.show(3) 