# China-eBus

## Overview
This repository contains all codes and (sample) dataset of the paper - 
***Variances in Operational Feasibility, Investment Cost and Environmental Benefit of Bus Electrification across 224 Chinese Cities***. 

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
1. Git clone/download the repository to your local disk.
2. The full datasets will be provided upon request through our Global EV Data Initiative at https://globalevdata.github.io/datasets.
3. Run
   1. **data collection**: run each script in the dir ``./Code/Data collection``
   2. **topic and sentiment analysis**: run each script in the dir ``./Code/Analysis/Topic_and_sentiment_analysis`` (Details can be seen in the document ``./Code/Analysis/Topic_and_sentiment_analysis/Workflow_for_topic_and_sentiment_analysis_of_EV_charging_station_reviews.docx``)
   3. **statistical analysis**: run each script in the dir ``./Code/Analysis/Statistical_analysis``
   4. **plot**: run each script in the dir ``./Code/Figure plotting``
4. Outputs (including text files and figures) will be stored in the dir ``./Data/Interim`` and ``./Data/Figure_plots``, respectively.

# Usage
1. Clone or download the repository to your local machine.
2. Prepare the required input data under `./data`. The full datasets will be provided upon request through our Global EV Data Initiative at https://globalevdata.github.io/datasets.
3. Run `./preparation/main.py` for the main data preparation workflow, including road processing, timetable preparation, duration estimation, trajectory generation, vehicle scheduling, and route type assignment, in order to generate the key input data required by `amosa4CN/test4mosa` for the simulation-based optimization program.


## Contact
- Leave questions in [Issues on GitHub](https://github.com/XanderPENG/global-evcs/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information
