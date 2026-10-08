# China-eBus

## Overview

This repository contains all codes and (sample) dataset of the paper -
***Network structure shapes system costs and emissions of bus electrification***.

Note that the **full dataset** can be requested through our [Global EV Data Initiative](https://globalevdata.github.io/data.html).

## Requirements and Installation

The analysis code is designed to run in a **Python** environment across Windows and macOS systems.

### Prerequisites

It is highly recommended to use the following software versions:

**Python Packages**:
- `python`: 3.10+ (3.12 recommended)
- `pandas`: 2.0+
- `numpy`: 1.24+
- `scikit-learn`: 1.0+
- `matplotlib`: 3.9+
- `seaborn`: 0.12+
- `geopandas`: 0.12+
- `shapely`: 2.0+
- `networkx`: 3.0+ (for network analysis)
- `scipy`: 1.10+ (for statistical analysis)
- `numba`: for accelerated simulation

See `requirements.txt` for the full package list. The sample workflow has been tested on Windows with Python 3.12; macOS has not been independently tested.

### Installation

It is highly recommended to download [AnaConda](https://www.anaconda.com) to create/manage Python environments.
You can create a new Python environment and install the required packages via both the GUI or Command Line.
Typically, the installation should be prompt (around _10-20 min_ from a "_clean_" machine to "_ready-to-use_" machine, but highly dependent on the Internet speed).

- via **Anaconda GUI**
  1. Open the Anaconda
  2. Find and click "_Environments_" at the left sidebar
  3. Click "_Create_" to create a new Python environment
  4. Select the created Python environment in the list, and then search and install all packages one by one.

- via **Command Line** (using **_Terminal_** for macOS machine and **_Anaconda Prompt_** for Windows machine, respectively)
  1. Create your new Python environment
     ```bash
     conda create --name <input_your_environment_name> python=3.12
     ```
  2. Activate the new environment
     ```bash
     conda activate <input_your_environment_name>
     ```
  3. Install the required packages from the repository root
     ```bash
     python -m pip install -r requirements.txt
     ```

## Usage

1. Clone or download the repository to your local machine.
2. Prepare the required input data under `./data`. Due to the large dataset size, we only keep the data for one example city in this repository. The full datasets for the other cities are available upon request through our Global EV Data Initiative at [https://globalevdata.github.io/datasets](https://globalevdata.github.io/datasets).
3. **Preparation**
   1. Run the scripts in `./preparation/`, starting with `main.py`, for the main data preparation workflow, including road processing, timetable preparation, duration estimation, vehicle scheduling, and route type assignment.
   2. This step generates the key input data required by the simulation-based optimisation program in `amosa4CN/run_amosa.py`.
4. **Simulation-based optimisation**
   1. Run `amosa4CN/run_amosa.py` to execute the simulation-based optimisation program for all cities.
   2. You may adjust parameters such as the city list, number of processes, maximum iterations, and convergence settings using command-line arguments (`--help`). Run `amosa4CN/run_scenarios.py` to generate what-if or sensitivity-analysis results.
   3. The optimisation outputs will be stored in the corresponding directory under `data/output/results/`.
5. **Result analysis and plotting**
   1. Use `./preparation/indicator_calculation.py` for city-level indicator calculations. The prepared indicators are stored in `data/224city_indicators.csv`.
   2. Run the plotting scripts in `./result_plot/` to generate the three figures in the paper:
      - `fig_1.py`
      - `fig_2.py`
      - `fig_3.py`
   3. Run `./result_plot/tab_1.py` to reproduce the regression results in Table 1.
   4. Note that `tab_1.py` requires `data/224city_indicators.csv` and the corresponding optimisation archives, which should be prepared in advance through the steps above. Full analyses require the full datasets.

## Contact

- Leave questions in [Issues on GitHub](https://github.com/huajide/China-eBus/issues).
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information
