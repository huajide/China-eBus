# China-eBus

## Overview
This repository contains all codes and (sample) dataset of the paper - 
***How does bus network structure shape system costs and emissions of fleet electrification? Evidence from 224 Chinese cities***. 

Note that the **full dataset** can be requested through our [Global EV Data Initiative](https://globalevdata.github.io/data.html).

## Requirements and Installation

The analysis code is designed to run in a **Python** environment across Windows and macOS systems.

### Prerequisites

It is highly recommended to use the following software versions:

**Python Packages**:
- `python`: 3.9+ (3.10+ recommended)
- `pandas`: 2.0+
- `numpy`: 1.24+
- `scikit-learn`: 1.0+
- `matplotlib`: 3.5+
- `seaborn`: 0.12+
- `geopandas`: 0.12+
- `shapely`: 2.0+
- `networkx`: 3.0+ (for network analysis)
- `scipy`: 1.10+ (for statistical analysis)
- `arcpy`: For advanced GIS operations (requires ArcGIS Pro 2.8+)

### Installation

It is highly recommended to download [AnaConda](https://www.anaconda.com) to create/manage Python environments.
You can create a new Python environment and install required aforementioned packages (except for `arcpy`) via both the GUI or Command Line.
Typically, the installation should be prompt (around _10-20 min_ from a "_clean_" machine to "_ready-to-use_" machine, but highly dependent on the Internet speed).

- via **Anaconda GUI**
  1. Open the Anaconda
  2. Find and click "_Environments_" at the left sidebar
  3. Click "_Create_" to create a new Python environment
  4. Select the created Python environment in the list, and then search and install all packages one by one.


- via **Command Line** (using **_Terminal_** for macOS machine and **_Anaconda Prompt_** for Windows machine, respectively)
  1. Create your new Python environment
     ```
     conda create --name <input_your_environment_name> python=3.10.6
     ```
  2. Activate the new environment 
     ```
     conda activate <input_your_environment_name>
     ```
  3. Install all packages one by one 
     ```
     conda install <package_name>=<specific_version>
     ```

# Usage

1. Clone or download the repository to your local machine.
2. Prepare the required input data under `./data`. Due to the large dataset size, we only keep the data for one example city in this repository. The full datasets for the other cities are available upon request through our Global EV Data Initiative at [https://globalevdata.github.io/datasets](https://globalevdata.github.io/datasets).
3. **Preparation**
   1. Run `./preparation/main.py` for the main data preparation workflow, including road processing, timetable preparation, duration estimation, trajectory generation, vehicle scheduling, and route type assignment.
   2. This step generates the key input data required by the simulation-based optimization program in `amosa4CN/test4mosa_multi_cities.py`.
4. **Simulation-based optimization**
   1. If needed, run `amosa4CN/test4mosa_cs_dict.py` in advance to generate `all_d2s_dict.pkl`.
   2. If needed, run `amosa4CN/test4mosa_simplified.py` in advance to generate `vs_parking_nodeid_simplified.csv`.
   3. Run `amosa4CN/test4mosa_multi_cities.py` to execute the simulation-based optimization program for all cities.
   4. In this script, you may adjust parameters such as the city list, number of processes, simplified inputs, preloaded dictionaries, `SIM_N`, `min_delta_hv`, and `what_if` settings to generate baseline or what-if scenario results.
   5. The optimization outputs will be stored in the corresponding directory under `../data/output/mosa/`.
5. **Result analysis and plotting**
   1. Run `amosa4CN/TestingResults/analyse_report.ipynb` to summarize the optimization outputs and generate `../data/224cities_output.csv`.
   2. Run `./resultplot/indicator_calculation.py` to compute the city-level indicators and generate `../data/224city_indicators.csv`.
   3. Run the plotting scripts in `./resultplot/` to generate the three figures in the paper:
      1. `fig_1.py`
      2. `fig_2.py`
      3. `fig_3.py`
   4. Run `./resultplot/tab_1.py` to reproduce the regression results in Table 1.
   5. Note that `tab_1.py` requires both `../data/224city_indicators.csv` and `../data/224cities_output.csv`, which should be prepared in advance through the indicator calculation and optimization result summary steps above.

## Contact
- Leave questions in [Issues on GitHub](https://github.com/XanderPENG/global-evcs/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information
