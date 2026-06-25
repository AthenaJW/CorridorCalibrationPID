# **SUMO-in-the-loop Corridor Calibration (with PID control!)**  

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg) 

This work is an extension of the following repository that adds on a PID method: https://github.com/yanb514/CorridorCalibration

which contains the code used for this paper:
```
@misc{wang2024calibrating,
  title={Calibrating Microscopic Traffic Models with Macroscopic Data},
  author={Wang, Yanbing and de Souza, Felipe and Zhang, Yaozhong and Karbowski, Dominik},
  note={https://ssrn.com/abstract=5065262},
  year={2024}
}
```
Please refer to the CorridorCalibration repository for initial setup instructions
---

## **Setup**  
- [Package Dependencies](#package-dependencies)
- [Simulation Files](#simulation-files)  
- [Plotting Files](#plotting-files)  
- [Relevant Directories](#relevant-directories)      

---
## **Package Dependencies**
This project is managed using uv, a fast Python package installer and resolver.
Prerequisites
Ensure you have uv installed on your system:
```
# Using curl (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using pip
pip install uv
```

To install the project and all required dependencies into a synchronized virtual environment, run the following command in the root directory:
```
uv sync
```
This will automatically read the pyproject.toml and create a locked virtual environment in .venv/.

## **Simulation Files**  
The simulation files for the onramp, i24, and i24b scenarios are the following
- /sumo/onramp/onramp_calibrate_PID.py
- /sumo/i24/i24_calibrate_PID.py
- /sumo/i24b/i24b_calibrate_PID.py
---
The onramp scenario only supports synthetic data, while the i24 and i24b scenario support real and synthetic data.

The i24 scenario supports the PID method, an OD Estimation method adapted from the CorridorCalibration project, and a flowrouter baseline comparison.

To toggle between real and synthetic data, and the different methods supported, ctrl-f for this block of code:
```
RERUN_GT = False # whether to rerun the ground truth simulation and regenerate synthetic measurements (set to False to save time if already done)
REAL_DATA = False
method = "PID" # or "FLOWROUTER" "OD_ESTIMATION"
```
and correct the field to your desired configuration.

The i24b scenario supports only flowrouter, the od_estimation method in that file is not functional. Additionally, through obtaining a fcd output file from the microsim calibration benchmark library (currently private) one can obtain a comparison between the PID method and genetic algorithm by using it in the data analysis and plotting methods.

File output naming convention (this will be in the appropriate method directory, example: sumo/onramp/fcd_output/fcd_pid_sim_3hr.xml)

**fcd_output/fcd_pid_sim_3hr.xml** - fcd file output from pid method

**pid_log_sim_3hr.csv** - log of observed vehicle count versus actual at controlled sensors from pid method

**fcd_output/fcd_fr_sim_3hr.xml** - fcd file output from flowrouter method

**fr_log.csv** - log of observed vehicle count versus actual at controlled sensors from flowrouter method

**fcd_output/fcd_od_sim_3hr.xml** - fcd file output from od method

**od_log.csv** - log of observed vehicle count versus actual at controlled sensors from od estimation method

Following this naming convention, should you also want to compare genetic algorithm, or any other method on the i24b benchmark, add your fcd file to fcd_output in the appropriate directory.

## **Plotting Files**

/sumo/utils_vis_PID.py is the primary file responsible for generating time space .npy files, that can be further visualized however the user likes.

### Dependencies  
This package is built on Python 3.11, and requires installation of [optuna](https://optuna.org/), and [sumo](https://sumo.dlr.de/docs/Installing/index.html)

## **Relevant Directories**  



### 3. Running calibration  
To run the calibration of any scenario, navigate to the SCENARIO folder and run `SCENARIO_calibrate.py`. For example, to run the `i24b` scenario:
```bash  
cd sumo/i24b
python i24b_calibrate.py
```  
The calibration progress such as current best parameters will be saved in `sumo/i24/_log`.

### 4. Evaluation and plotting (TODO: i24b in progress)
All evaluation related computations are located in `sumo/SCENARIO/SCENARIO_results.py`. Current evaluation & visualization support:
1. Calibrated vs. measured speed at the locations of stationary sensors
![detector speeds](det_b.png)
2. Macroscopic quantities (speed, flow, density)
![macroscopic quantities](asm_5hr.png)
3. Lane-specific travel time
![travel time](travel_time_rds.png)

### Key utility functions
In summary,
- `utils_data_read.py` contains functions to read and process RDS and .xml data
- `utils_vis.py` contains all visualization functions
- `utils_macro.py` contains Edie's method to compute macroscopic traffic quantities from trajectory data

The detailed descriptions of these methods are documented inline. To highlight a few:
- `utils_data_read.parse_and_reorder_xml()` takes the SUMO floating car data (fcd) output `.xml` file, reorders by trajectory and time into NGSIM data format.
- `utils_macro.compute_macro_generalized()` implements the generalized Edie's method, and processes trajectory data into macroscopic quantities for the specified spatial and temporal window.
- `utils_macro.plot_macro()` plots the macroscopic quantities of flow, density and speed computed using `macro.compute_macro_generalized()`.
- `utils_vis.visualize_fcd()` plots the time-space diagram given the fcd file.
- `utils_vis.plot_line_detectors()` plot the aggregated traffic data generated from SUMO at the specified detector locations.

### Using calibrated SUMO
If you only want to work with the calibrated SUMO scenarios without the calibration, you are in good hands!
All calibrated scenarios are located in `sumo/SCENARIO/calibrated`, which contains all the necessary files to run SUMO. You can run `SCENARIO.sumocfg` directly using SUMO-gui, or using command line 
```bash
cd sumo/SCENARIO/calibrated
sumo -c SCENARIO.sumocfg
```

### TODOS
- temp files handling in i24 and onramp scenarios
- add calibrated results for i24b scenario (i24b_results.py and plotting)
- calibrate only westbound?
---
