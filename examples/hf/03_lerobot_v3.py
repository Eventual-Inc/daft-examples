# /// script
# description = "Filter LeRobot v3 episodes before expanding selected episodes into frames"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import daft
from daft.datasets import lerobot

DATASET = "lerobot/aloha_sim_insertion_human"


def main() -> None:
    # This public dataset uses the LeRobot v3 layout: meta/episodes, data, and videos.
    episodes = lerobot.read_episodes(DATASET).where(daft.col("length") >= 200).limit(2)

    # Keep the episode filter in the lazy plan before expanding into frame-level rows.
    (
        lerobot.load_episode_frames(episodes, DATASET)
        .select("episode_index", "frame_index", "task_index")
        .limit(5)
        .show()
    )


if __name__ == "__main__":
    main()
