import csv
import subprocess
import os
import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
import shutil
import pickle
import logging
import json
from datetime import datetime
import traci
from simple_pid import PID

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader

# import utils_macro as macro
# import utils_vis as vis

# Initialize PID: PID(Kp, Ki, Kd, setpoint)
# Kp: Proportional (current error)
# Ki: Integral (accumulated error)
# Kd: Derivative (rate of change)
# ================ on-ramp scenario setup ====================
SCENARIO = "onramp"
EXP = "1b"
N_TRIALS = 10000
SUMO_DIR = os.path.dirname(os.path.abspath(__file__)) # current script directory

with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('HOSTNAME', 'Unknown')

if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_PATH']["CSI"]
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_PATH']["VMS"]
else: # run on SOL
    SUMO_EXE = config['SUMO_PATH']["SOL"]

if "1" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
    min_val = [30.0, 1.0, 1.0, 1.0, 0.5]  
    max_val = [35.0, 3.0, 4.0, 3.0, 2.0] 
elif "2" in EXP:
    param_names = ['lcSublane', 'maxSpeedLat', 'lcAccelLat', 'minGapLat', 'lcStrategic', 
                   'lcCooperative', 'lcPushy','lcImpatience','lcSpeedGain', ]
    min_val = [0,  0,  0,  0, 0,  0, 0, 0, 0, 0, 0]  
    max_val = [10, 10, 5,  5, 10, 1, 1, 1, 1, 1, 1] 
elif "3" in EXP:
    param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 
                   'lcSublane', 'maxSpeedLat', 'lcAccelLat', 'minGapLat', 'lcStrategic', 
                   'lcCooperative', 'lcPushy','lcImpatience','lcSpeedGain']
    min_val = [30.0, 1.0, 1.0, 1.0, 0.5, 0,  0,  0,  0, 0,  0, 0, 0, 0]  
    max_val = [35.0, 3.0, 4.0, 3.0, 2.0, 10, 10, 5,  5, 10, 1, 1, 1, 1] 
if "a" in EXP:
    MEAS = "volume"
elif "b" in EXP:
    MEAS = "speed"
elif "c" in EXP:
    MEAS = "occupancy"

DEFAULT_PARAMS = {
    "maxSpeed": 30.55,
    "minGap": 2.5,
    "accel": 1.5,
    "decel": 2,
    "tau": 1.4,
    "emergencyDecel": 4.0,
    "laneChangeModel": "SL2015",
    "lcSublane": 1.0,
    "latAlignment": "arbitrary",
    "maxSpeedLat": 1.4,
    "lcAccelLat": 0.7,
    "minGapLat": 0.4,
    "lcStrategic": 10.0,
    "lcCooperative": 1.0,
    "lcPushy": 0.4,
    "lcImpatience": 0.9,
    "lcSpeedGain": 1.5,
    "lcKeepRight": 0.0,
    "lcOvertakeRight": 0.0
}
measurement_locations = ['upstream_0', 'upstream_1', 
                            'merge_0', 'merge_1', 'merge_2', 
                            'downstream_0', 'downstream_1', 'ramp_0']

num_timesteps = config["onramp"]["SIMULATION_TIME"]
step_length = config["onramp"]["STEP_LENGTH"]
detector_interval = config["onramp"]["DETECTOR_INTERVAL"]

def run_sumo(sim_config, tripinfo_output=None, fcd_output=None):
    """Run a SUMO simulation with the given configuration."""
    # command = ['sumo', '-c', sim_config, '--tripinfo-output', tripinfo_output, '--fcd-output', fcd_output]

    command = [SUMO_EXE, '-c', sim_config, 
               '--no-step-log',  '--xml-validation', 'never', 
               '--lateral-resolution', '0.5']
    if tripinfo_output is not None:
        command.extend(['--tripinfo-output', tripinfo_output])
        
    if fcd_output is not None:
        command.extend([ '--fcd-output', fcd_output])
        
    subprocess.run(command, check=True)

def run_PID_closed_loop_sumo(sim_config, controlled_flow_routes, sensing_detectors, debug_detectors, pid_controllers, synth_data, tripinfo_output=None, fcd_output=None):
    """Run PID closed loop calibration on SUMO simulation with the 
    given configuration and synthetic data measurements"""
    # command = ['sumo', '-c', sim_config, '--tripinfo-output', tripinfo_output, '--fcd-output', fcd_output]

    ''' TO DO: ramp, detectors, and sensing interval are hardcoded right now'''
    command = [SUMO_EXE, '-c', sim_config, 
               '--no-step-log',  '--xml-validation', 'never', 
               '--lateral-resolution', '0.5']
    if tripinfo_output is not None:
        command.extend(['--tripinfo-output', tripinfo_output])
        
    if fcd_output is not None:
        command.extend([ '--fcd-output', fcd_output])
    
    pid_sensor_log = "pid_log.csv"
    debug_sensor_log = "debug_log.csv"
    header_pid = ['step', 'time', 'sensors', 'target', 'observed', 'error', 'control_output', 'new_flow', 'veh_ids']
    header_debug = ['step', 'time', 'sensors', 'target', 'observed']

    with open(pid_sensor_log, mode='w', newline='') as f1, \
        open(debug_sensor_log, mode='w', newline='') as f2:
        
        writer_sensor = csv.writer(f1)
        writer_debug = csv.writer(f2)

        writer_sensor.writerow(header_pid)
        writer_debug.writerow(header_debug)
        # 2. Start Simulation
        traci.start(command)
            
        closed_loop_flows = [[0.0]*len(route_tuple) for route_tuple in controlled_flow_routes]
        next_departure_times = [[0.0]*len(route_tuple) for route_tuple in controlled_flow_routes]
        observed_flows = [0.0]*len(sensing_detectors)
        observed_flows_debug = [0.0]*len(debug_detectors)
        try:
            for step in range(int(num_timesteps//step_length)):
                traci.simulationStep()
                time = step * step_length
                
                #manually adding traffic flow
                for i, route_tuple in enumerate(controlled_flow_routes):
                    closed_loop_flow_list = closed_loop_flows[i]
                    next_departure_time_list = next_departure_times[i]
                    for j, closed_loop_flow in enumerate(closed_loop_flow_list):
                        next_departure_time = next_departure_time_list[j]
                        if closed_loop_flow > 0:
                            num_vehicles_to_add = 1
                            # Calculate seconds between cars
                            interval = 3600.0 / closed_loop_flow
                            if closed_loop_flow > 3600:
                                interval = 1.0  # Cap at 1 vehicle per second
                                num_vehicles_to_add = int(closed_loop_flow // 3600)
                            
                            
                            if time >= next_departure_time:
                                veh_id = f"ctrl_veh_{route_tuple[j]}_{time}"
                                
                                # 3. Add the vehicle to the simulation
                                for k in range(num_vehicles_to_add):
                                    traci.vehicle.add(vehID=veh_id + str(k), routeID=route_tuple[j], typeID="hdv", departLane="random", departSpeed="desired")
                                
                                # Schedule the next one
                                next_departure_times[i][j] = time + interval
                        else:
                            # If flow is 0 or negative, we just wait
                            pass

                if time % (detector_interval) == float((detector_interval-1)): # measure right before detector reset interval
                    for i, det in enumerate(sensing_detectors):
                        if isinstance(det, tuple):
                            cumulative = sum([traci.inductionloop.getIntervalVehicleNumber(d) for d in det])
                        else:
                            raise Exception("sensing_detectors should be a list of tuples")
                        observed_flows[i] = (cumulative / detector_interval) * 3600
                        print("observed flow for det: ", det, observed_flows[i])
                    for i, det in enumerate(debug_detectors):
                        if isinstance(det, tuple):
                            cumulative_debug = sum([traci.inductionloop.getIntervalVehicleNumber(d) for d in det])
                        else:
                            raise Exception("debug_detectors should be a list of tuples")
                        observed_flows_debug[i] = (cumulative_debug / detector_interval) * 3600
                        # --- Apply control logic here (traci.vehicle.add, etc.) ---
                if time % detector_interval != 0 or time == 0.0:
                    continue
                # reset the PID setpoint based on synthetic data
                for i, det in enumerate(sensing_detectors):
                    pid = pid_controllers[i]
                    observed_flow = observed_flows[i]
                    if isinstance(det, tuple):
                        cumulative_gt = sum(synth_data[measurement_locations.index(d)][int(time//(detector_interval))] for d in det)
                    else:
                        raise Exception("sensing_detectors should be a list of tuples")
                    pid.setpoint = cumulative_gt
                    
                    # Get your data
                    control_signal = pid(observed_flow)
                    error = pid.setpoint - observed_flow
                    
                    # --- Apply control logic here (traci.vehicle.add, etc.) ---
                    flow_change = int(round(control_signal))
                    # proportionally adjust the closed-loop flow tuple
                    total_closed_loop_flow = sum(closed_loop_flows[i])

                    # 2. Calculate target (floor at 0)
                    target_total = max(0, total_closed_loop_flow + flow_change)

                    # 3. Handle Scaling
                    if total_closed_loop_flow > 0:
                        # Standard Proportional Scaling
                        scaling_factor = target_total / total_closed_loop_flow
                        closed_loop_flows[i] = [
                            max(0, int(round(val * scaling_factor))) 
                            for val in closed_loop_flows[i]
                        ]
                        
                    elif target_total > 0:
                        # INITIAL STARTUP: Current is 0, but we need to increase.
                        # Distribute the flow_change evenly across the list.
                        num_elements = len(closed_loop_flows[i])
                        share = target_total // num_elements
                        remainder = target_total % num_elements
                        
                        # Create the new list, adding the remainder to the first element 
                        # so the total sum matches target_total exactly.
                        new_flow = [share] * num_elements
                        new_flow[0] += remainder 
                        closed_loop_flows[i] = new_flow
            

                    # 3. Log data to CSV
                    print(f"Step {step} for detector {det}: Setpoint={pid.setpoint}, Observed={observed_flow}, Error={error}, Control Signal={control_signal}, New Flow = {closed_loop_flows[i]}")
                    writer_sensor.writerow([step, time, det, pid.setpoint, observed_flow, error, control_signal, closed_loop_flows[i]])
                
                for i, det in enumerate(debug_detectors):
                    observed_flow_debug = observed_flows_debug[i]
                    if isinstance(det, tuple):
                        cumulative_gt_debug = sum(synth_data[measurement_locations.index(d)][int(time//(detector_interval))] for d in det)
                    else:
                        raise Exception("debug_detectors should be a list of tuples")
                    # 3. Log data to CSV
                    print(f"Step {step} for debug detector {det}: Target={cumulative_gt_debug}, Observed={observed_flow_debug}")
                    writer_debug.writerow([step, time, det, cumulative_gt_debug, observed_flow_debug])
        finally:
            traci.close()





# def get_vehicle_ids_from_routes(route_file):
#     tree = ET.parse(route_file)
#     root = tree.getroot()

#     vehicle_ids = []
#     for route in root.findall('.//vehicle'):
#         vehicle_id = route.get('id')
#         vehicle_ids.append(vehicle_id)

#     return vehicle_ids



def update_sumo_configuration(param):
    """
    Update the SUMO configuration file with the given parameters.
    All parameters in .rou.xml not present in the given param will be removed
    
    Parameters:
        param (dict): Dictionary of parameter values {attribute_name: value}
    """
    # Define the path to your rou.xml file
    file_path = SCENARIO + '.rou.xml'
    
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find the vType element with id="hdv"
    for vtype in root.findall('vType'):
        if vtype.get('id') == 'hdv':
            # Remove all existing attributes
            for attr in list(vtype.attrib.keys()):
                # These params are not calibrated
                if attr not in ["id", "length", "carFollowModel","emergencyDecel", "laneChangeModel", "latAlignment", "lcKeepRight", "lcOvertakeRight"]:
                    del vtype.attrib[attr]
            
            # Set new attributes from param
            for key, val in param.items():
                vtype.set(key, str(val))
            break
    
    # Write the updated XML content back to the file
    tree.write(file_path, encoding='UTF-8', xml_declaration=True)


def create_temp_config(param, trial_number):
    """
    Update the SUMO configuration file with the given parameters and save it as a new file.
    create new .rou.xml and .sumocfg files for each trial
    
    Parameters:
        param (dict): List of parameter values [maxSpeed, minGap, accel, decel, tau]
        trial_number (int): The trial number to be used for naming the new file.
    """
    
    # Define the path to your original rou.xml and sumocfg files
    original_rou_file_path = SCENARIO + '.rou.xml'
    original_net_file_path = SCENARIO + '.net.xml'
    original_sumocfg_file_path = SCENARIO + '.sumocfg'
    original_add_file_path = 'detectors.add.xml'
    
    # Create the directory for the new files if it doesn't exist
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== .Parse the original rou.xml file ==========================
    rou_tree = ET.parse(original_rou_file_path)
    rou_root = rou_tree.getroot()

    # Find the vType element with id="hdv"
    for vtype in rou_root.findall('vType'):
        if vtype.get('id') == 'hdv':
            # Update the attributes with the provided parameters
            for key, val in param.items():
                vtype.set(key, str(val))
            break

    new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
    rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)

    # ==================== copy original net.xml file ==========================
    shutil.copy(original_net_file_path, os.path.join(output_dir, f"{trial_number}_{SCENARIO}.net.xml"))

    # ==================== copy original add.xml file ==========================
    new_add_file_path = os.path.join(output_dir, f"{trial_number}_{original_add_file_path}")
    shutil.copy(original_add_file_path, new_add_file_path)
    
    #  ==================== parse original sumocfg.xml file ==========================
    sumocfg_tree = ET.parse(original_sumocfg_file_path)
    sumocfg_root = sumocfg_tree.getroot()
    input_element = sumocfg_root.find('input')
    if input_element is not None:
        input_element.find('route-files').set('value', f"{trial_number}_{SCENARIO}.rou.xml")
        input_element.find('net-file').set('value', f"{trial_number}_{SCENARIO}.net.xml")
        input_element.find('additional-files').set('value',  f"{trial_number}_{original_add_file_path}")

    new_sumocfg_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.sumocfg")
    sumocfg_tree.write(new_sumocfg_file_path, encoding='UTF-8', xml_declaration=True)
    
    return new_sumocfg_file_path, output_dir


def objective(trial):
    """Objective function for optimization."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    }
    # print(driver_param, trial.number)
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param, trial.number)

    # Run SUMO simulation
    # run_sumo(SCENARIO+'.sumocfg')
    run_sumo(temp_config_path)
    
    # Extract simulated traffic volumes
    simulated_output = reader.extract_sim_meas(["trial_"+ location for location in measurement_locations],
                                        file_dir = temp_path)
    
    # RMSE
    diff = simulated_output[MEAS] - measured_output[MEAS] # measured output may have nans
    error = np.sqrt(np.nanmean(diff.flatten()**2))

    clear_directory(os.path.join("temp", str(trial.number)))
    # logging.info(f'Trial {trial.number}: param={driver_param}, error={error}')
    
    return error


def logging_callback(study, trial):
    # if trial.state == optuna.trial.TrialState.COMPLETE:
    #     logging.info(f'Trial {trial.number} succeeded: value={trial.value}, params={trial.params}')
    if trial.state == optuna.trial.TrialState.FAIL:
        logging.error(f'Trial {trial.number} failed: exception={trial.user_attrs.get("exception")}')
    
    if study.best_trial.number == trial.number:
        logging.info(f'Current Best Trial: {study.best_trial.number}, best value: {study.best_value}')
        logging.info(f'Current Best Parameters: {study.best_params}')

def clear_directory(directory_path):
    """
    Clear all files within the specified directory.
    
    Parameters:
        directory_path (str): The path to the directory to be cleared.
    """
    try:
        shutil.rmtree(directory_path)
        # print(f"Directory {directory_path} and all its contents have been removed.")
    except FileNotFoundError:
        print(f"Directory {directory_path} does not exist.")
    except Exception as e:
        print(f"Error removing directory {directory_path}: {e}")


if __name__ == "__main__":
    # ================ Configure the logging module ====================
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = '_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{current_time}_optuna_log_{EXP}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # ================================= initialize calibration: reset default parameters 
    update_sumo_configuration(DEFAULT_PARAMS)

    # ================================= run ground truth and generate synthetic measurements
    run_sumo(sim_config=SCENARIO+"_gt.sumocfg", fcd_output="fcd_output/fcd_sim.xml") #, fcd_output ="trajs_gt.xml")
    measured_output = reader.extract_sim_meas(measurement_locations)

    # ================================= run PID closed-loop calibration with synthetic measurements
    synth_data = measured_output['volume'].tolist()
    controlled_flows_route = [("ramp",), ("mainlane",)]
    sensing_detectors = [("merge_1", "merge_2"), ("merge_0", "merge_1")]
    debug_detectors = [("upstream_0", "upstream_1"), ("ramp_0",)]
    pid_controllers = [PID(0.05, 0, 0, setpoint=0.0), PID(0.1, 0, 0, setpoint=0.0)]
    for pid in pid_controllers:
        pid.output_limits = (-1000, 1000)
    ## TODO: add a configuration checking function here
    run_PID_closed_loop_sumo(sim_config=SCENARIO+".sumocfg", controlled_flow_routes=controlled_flows_route, sensing_detectors=sensing_detectors, debug_detectors=debug_detectors, pid_controllers=pid_controllers, synth_data=synth_data, fcd_output="fcd_output/fcd_pid.xml") #, fcd_output ="trajs_pid.xml")
    

