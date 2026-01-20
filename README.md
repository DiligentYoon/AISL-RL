# Reinforcement Learning Environment of Autonomous & Intelligent Systems Lab.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.0-red.svg)](https://github.com/isaac-sim/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)


## Getting Started

### Prerequisites

-   **NVIDIA Isaac Sim**: Ensure Isaac Sim is installed correctly.
-   **Isaac Lab**: This project is built upon Isaac Lab. Follow the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).


### Installation

1.  Clone this repository to a location of your choice.
2.  Install the project package in editable mode. This allows you to modify the source code without reinstalling.

    ```bash
    # From the repository's root directory
    # Ensure your python environment has Isaac Lab's dependencies
    pip install -e .
    ```
3. Adjust VScode settings by making `settings.json` file in reference to `settings_sample.json` file.

  ```json
  {
    "python.analysis.extraPaths": [
      "<your>/<filename>/<saved IsaacLab>/source/isaaclab",
      "<your>/<filename>/<saved IsaacLab>/source/isaaclab_tasks",
      "<your>/<filename>/<saved IsaacLab>/source/isaaclab_rl",
      "<your>/<filename>/<saved IsaacLab>/source/isaaclab_mimic",
      "<your>/<filename>/<saved IsaacLab>/source/isaaclab_assets"
    ],
    "python.defaultInterpreterPath": "<your>/<filename>/<saved>/<virtual env>/<env name>/python.exe"
  }
  