# China-eBus

## Overview

This repository contains model, analysis and selected preprocessing code, together with sample data, for the paper -
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
2. The repository includes prepared operational inputs and a baseline optimisation archive for **Puyang_Henan**, alongside national city indicators and plotting tables. Full operational inputs and archives for other cities are available upon request through our [Global EV Data Initiative](https://globalevdata.github.io/data.html).
3. **Preparation**
   1. The prepared Puyang inputs can be used directly; preprocessing is not required for the sample run.
   2. `preparation/main.py` exposes road processing (`--stage roads`) and slope/base-energy preparation (`--stage energy`). It requires raw GIS data, a raw city registry and, for the energy stage, existing vehicle schedules and SRTM data. The other preparation modules provide timetable, duration, scheduling and vehicle-class routines; they are not a single end-to-end pipeline.
   3. Raw preparation uses the original Chinese administrative names and writes Chinese-named city files under `data/raw_data/generated/`. Released inputs use names such as `Puyang_Henan` under `data/input/`. Translation and transfer between these directories are separate packaging steps, not performed by `main.py`.
4. **Simulation-based optimisation**
   1. Run the bundled example from the repository root:
      ```bash
      python amosa4CN/run_amosa.py --cities Puyang_Henan --output-root data/output/results/sample_run
      ```
      Running without `--cities` selects all 224 registered cities and requires their complete operational inputs.
   2. Command-line arguments (`--help`) control the city list, processes, iteration limit and convergence settings. Existing completed outputs are reused only when their recorded run settings match; use a separate output directory for a different configuration. `run_scenarios.py` runs what-if and sensitivity cases. Operating-time and imputed-headway cases additionally require separately prepared inputs under `data/input/SA/`; their generation pipeline is not included in this sample release.
   3. The optimisation outputs will be stored in the corresponding directory under `data/output/results/`.
5. **Result analysis and plotting**
   1. Prepared national indicators are stored in `data/224city_indicators.csv`. `preparation/indicator_calculation.py` calculates stop counts and population coverage from raw GIS inputs; it does not regenerate every column of that table.
   2. Run the plotting scripts in `./result_plot/` to generate the three figures in the paper:
      - `fig_1.py`
      - `fig_2.py`
      - `fig_3.py`
   3. Run `./result_plot/tab_1.py` to reproduce the regression results in Table 1.
   4. `tab_1.py` requires the national indicator table, all corresponding baseline archives and operational city CSVs to recover fleet counts. Full analyses require the full datasets.

## Contact

- Leave questions in [Issues on GitHub](https://github.com/huajide/China-eBus/issues).
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information
