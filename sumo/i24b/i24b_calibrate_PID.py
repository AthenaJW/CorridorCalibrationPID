from collections import deque
import csv
import subprocess
import os
import os
from xml.dom import minidom
import xml.etree.ElementTree as ET
import numpy as np
import sys
import shutil
import pickle
import logging
import json
from datetime import datetime, timedelta
import traci
from simple_pid import PID
import pandas as pd
import random
from generate_flow import get_flow_data_list

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
from utils_vis import od_estimation_large

# ================ CONFIGURATION ====================
SCENARIO = "i24b"
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

# SUMO_EXE = config['SUMO_PATH'] # customize SUMO_PATH in config.json
SUMO_EXE = os.getenv("SUMO_PATH", config['SUMO_PATH']["SOL"])  # Fallback to config

N_TRIALS = config["N_TRIALS"]  # config["N_TRIALS"] # optimization trials
N_JOBS = config["N_JOBS"]  # config["N_JOBS"] # cores
EXP = config["EXP"] # experiment label
RDS_DIR = config[SCENARIO]["RDS_DIR"] # directory of the RDS data
DEFAULT_PARAMS = config["DEFAULT_PARAMS"]
# ================================================


if "1" in EXP:
    params_range = config["PARAMS_RANGE"]["cf"]
elif "2" in EXP:
    params_range = config["PARAMS_RANGE"]["lc"]
elif "3" in EXP:
    params_range = {**config["PARAMS_RANGE"]["cf"], **config["PARAMS_RANGE"]["lc"]}
param_names, ranges = zip(*params_range.items())
min_val, max_val = zip(*ranges)

if "a" in EXP:
    MEAS = "flow"
elif "b" in EXP:
    MEAS = "speed"
elif "c" in EXP:
    MEAS = "occupancy"

initial_guess = {key: DEFAULT_PARAMS[key] for key in param_names if key in DEFAULT_PARAMS}

num_timesteps = config["i24"]["SIMULATION_TIME"]
step_length = config["i24"]["STEP_LENGTH"]
detector_interval = config["i24"]["DETECTOR_INTERVAL"]
rds_file =  "data/RDS/detections_0360-0600.csv"

def extract_detector_locations(csv_file, direction="westbound"):
    """
    Reads a detector measurement CSV file and extracts a list of unique detector locations.

    Parameters:
    - csv_file: Path to the detector measurement CSV file.

    Returns:
    - det_locations: A list of unique detector locations in the format "milemarker-eastbound_lane".
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, delimiter=';')
    
    # Extract unique detector IDs from the 'Detector' column
    detector_ids = df['Detector'].unique()
    
    # Convert detector IDs to the desired format (e.g., "555-eastbound_0")
    det_locations = [detector_id for detector_id in detector_ids if direction in detector_id]
    
    return det_locations


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
    
def run_sumo_flowrouter(sim_config, tripinfo_output=None, fcd_output=None, det_interval=30,
                        sensing_detectors=None, tracking_log = "fr_log.csv"):
    """
    Runs the pre-generated Flow Router simulation while logging 
    sensor data at fixed intervals to match the PID logic.
    """
    # 1. Build the command
    command = [SUMO_EXE, '-c', sim_config, 
               '--no-step-log', '--xml-validation', 'never', 
               '--lateral-resolution', '0.5']
    
    if tripinfo_output:
        command.extend(['--tripinfo-output', tripinfo_output])
    if fcd_output:
        command.extend(['--fcd-output', fcd_output])

    # 2. Start SUMO via TraCI
    traci.start(command)
    
    # Setup Logging (matching your PID CSV format)
    # This ensures your "Baseline" data looks exactly like your "PID" data
    f_log = open(tracking_log, 'w', newline='')
    writer = csv.writer(f_log)
    writer.writerow(["step", "time", "sensors", "target", "observed", "speed"])

    step = 0
    end_time = num_timesteps # Set your desired simulation end time in seconds
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            time = traci.simulation.getTime()
            
            # 3. Check if we reached the measurement interval
            if time % detector_interval == float(detector_interval - 1):
                for i, det_tuple in enumerate(sensing_detectors):
                    
                    # SUM the vehicle counts from every lane in the tuple
                    # This gives you the total flow for that cross-section
                    cumulative = sum(
                        traci.inductionloop.getLastIntervalVehicleNumber(d_id) 
                        for d_id in det_tuple
                    )
                    observed_flows = (cumulative / detector_interval) * 3600

                    
                    # Calculate target_flow from your synth_data to keep the CSV consistent
                    # Assuming target_idx aligns with your 30s buckets
                    target_idx = int(time // det_interval) - 1
                    
                    # We use the same lookup logic as your PID script
                    current_target = sum(
                        synth_data["volume"][measurement_locations.index(d)][target_idx]
                        for d in det_tuple
                    )

                    # Log the results
                    writer.writerow([step, time, str(det_tuple), current_target, observed_flows])
            
            step += 1
            if time > end_time:
                break
                
    finally:
        traci.close()
        f_log.close()

def run_PID_closed_loop_sumo(sim_config, controlled_flow_routes, sensing_detectors, 
                             debug_detectors, pid_controllers_rectifier, pid_controllers_feedback, synth_data, 
                             measurement_locations, detector_interval, num_timesteps, 
                             step_length, SUMO_EXE, transfer_delays=None, 
                             ma_windows=None, num_lanes=None, # New argument: List of seconds for MA
                             route_tracking_sensors=None, routes=None, pid_flows=None,
                             detectors_to_routes=None,
                             pid_sensor_log="pid_log_sim_3hr.csv",
                             tripinfo_output=None, fcd_output=None):
    
    command = [SUMO_EXE, '-c', sim_config, '--no-step-log', '--xml-validation', 'never', '--lateral-resolution', '0.5']
    if tripinfo_output: command.extend(['--tripinfo-output', tripinfo_output])
    if fcd_output: command.extend(['--fcd-output', fcd_output])

    # --- 1. Initialize Buffers ---
    if transfer_delays is None:
        transfer_delays = [0] * len(sensing_detectors)
    
    # Initialize Moving Average Windows (Default to 1 interval if None)
    if ma_windows is None:
        ma_windows = [detector_interval] * len(sensing_detectors)

    # Action Buffers (for Transfer Delay)
    action_buffers = []
    # Sensing Buffers (for Moving Average)
    sensing_history_buffers = []

    for i in range(len(sensing_detectors)):
        # Delay Logic
        d_steps = int(transfer_delays[i] // detector_interval)
        action_buffers.append(deque([0] * d_steps, maxlen=max(1, d_steps)))
        
        # Moving Average Logic (Window size in steps)
        # Note: We divide by detector_interval because the PID updates only every interval
        ma_steps = max(1, int(ma_windows[i] // detector_interval))
        sensing_history_buffers.append(deque(maxlen=ma_steps))

   
    header_pid = ['step', 'time', 'sensors', 'target', 'observed', 'smoothed_error', 'raw_control_signal', 'delayed_signal', 'new_total_injection']
    header_tracking = ['step', 'time', 'sensor_id', 'total_count'] + [f'route_{i}_prop' for i in range(len(controlled_flow_routes))]
    header_debug = ['step', 'time', 'sensors', 'target', 'observed']

    

    with open(pid_sensor_log, mode='w', newline='') as f1, \
        open("route_proportions.csv", mode='w', newline='') as f2, \
        open("debug_log.csv", mode = 'w', newline='') as f3:

        writer_sensor = csv.writer(f1)
        writer_tracking = csv.writer(f2)
        writer_debug = csv.writer(f3)
        writer_sensor.writerow(header_pid)
        writer_tracking.writerow(header_tracking)
        writer_debug.writerow(header_debug)
        
        traci.start(command)
        
        current_injected_flows = [[0.0]*len(r) for r in controlled_flow_routes]
        next_departure_times = [[0.0]*len(r) for r in controlled_flow_routes]
        observed_flows_raw = [0.0] * len(sensing_detectors)
        debug_observed_flows_raw = [0.0] * len(debug_detectors) if debug_detectors else [0.0] * len(sensing_detectors)

        route_to_id = {routes[i]: i for i in range(len(routes))}
        interval_route_counts = {s: [0]*len(controlled_flow_routes) for s in (route_tracking_sensors or [])}


        try:
            steps_total = int(num_timesteps // step_length)
            for step in range(steps_total):
                traci.simulationStep()
                time = step * step_length

                # --- PART A: INJECT VEHICLES (Unchanged) ---
                for i, route_tuple in enumerate(controlled_flow_routes):
                    for j, route_id in enumerate(route_tuple):
                        flow = current_injected_flows[i][j]  # veh/hour
                        
                        if flow <= 0:
                            continue
                            
                        # 1. Calculate the average number of vehicles expected in this specific step
                        # (flow / 3600) converts vph to vps. Multiplying by step_length gives expected count.
                        expected_in_step = (flow / 3600.0) * step_length
                        
                        # 2. Determine how many vehicles to add this step
                        # For low flows (like your 94 vph), this will usually be 0 or 1.
                        # For very high flows (>3600 vph), it might occasionally be more than 1.
                        num_to_add = 0
                        if expected_in_step < 1.0:
                            # Poisson "roll of the dice" for low flow
                            if random.random() < expected_in_step:
                                num_to_add = 1
                        else:
                            # For high flow, we use the integer part and roll for the remainder
                            num_to_add = int(expected_in_step)
                            if random.random() < (expected_in_step - num_to_add):
                                num_to_add += 1

                        # 3. Inject the vehicles
                        for k in range(num_to_add):
                            # Unique ID using a random suffix to avoid collisions in the same timestep
                            v_id = f"ctrl_{route_id}_{time}_{k}_{random.randint(0, 999)}"
                            try:
                                traci.vehicle.add(v_id, route_id, typeID="hdv", 
                                                departLane="best", departSpeed="desired")
                            except traci.TraCIException as e:
                                # If the road is too congested to fit the car, TraCI throws an error.
                                # In a 'line' jam, this happens often.
                                pass

                                for group_idx, s_id in enumerate(route_tracking_sensors or []):
                                    # Use the index or a joined string of IDs as the key for interval_route_counts
                                    veh_data = traci.inductionloop.getVehicleData(s_id)
                                        
                                    for v_entry in veh_data:
                                        v_id = v_entry[0]  # Vehicle ID is the first element in the tuple
                                        try:
                                            # Identify the origin/control group of this vehicle
                                            r_id = traci.vehicle.getRouteID(v_id)
                                            interval_route_counts[s_id][route_to_id[r_id]] += 1
                                                
                                        except traci.TraCIException:
                                            # Handle cases where vehicle exists in veh_data but left the sim
                                            continue
                ### COMMENT OUT IF NOT DOING POISSON INFLOW ABLATION STUDY ###
                # for i, route_tuple in enumerate(controlled_flow_routes):
                #     for j, route_id in enumerate(route_tuple):
                #         flow = current_injected_flows[i][j]
                #         if flow > 0 and time >= next_departure_times[i][j]:
                #             interval = 3600.0 / flow
                #             num_to_add = int(flow // 3600) if flow > 3600 else 1
                #             for k in range(num_to_add):
                #                 v_id = f"ctrl_{route_id}_{time}_{k}"
                #                 traci.vehicle.add(v_id, route_id, typeID="hdv", departLane="best", departSpeed="desired")
                #             next_departure_times[i][j] = time + (interval if flow <= 3600 else 1.0)
                ### COMMENT OUT IF NOT DOING POISSON INFLOW ABLATION STUDY ###

                # --- PART B: MEASURE FLOW ---
                if time % detector_interval == float(detector_interval - 1):
                    target_idx = int(time // detector_interval) + 1
                    route_proportions = [1] * len(routes)
                    try:
                        for det_route in detectors_to_routes:
                            shared_routes = detectors_to_routes[det_route]
                            for r in shared_routes:
                                route_idx = route_to_id[(r,)]
                                route_proportions[route_idx] = pid_flows[r][target_idx] / sum(pid_flows[s][target_idx] for s in shared_routes)
                    except:
                        for det_route in detectors_to_routes:
                            shared_routes = detectors_to_routes[det_route]
                            for r in shared_routes:
                                route_idx = route_to_id[(r,)]
                                route_proportions[route_idx] = 1 / len(shared_routes)

                    ###### COMMENT OUT IF NOT DOING ROUTE ABLATION STUDY ######
                    # route_proportions[2] = 0.5
                    # route_proportions[3] = 0.5
                    # route_proportions[4] = 0.7
                    ###### COMMENT OUT IF NOT DOING ROUTE ABLATION STUDY ######

                    for i, det in enumerate(sensing_detectors):
                        cumulative = sum([traci.inductionloop.getIntervalVehicleNumber(d) * route_proportions[i] for d in det])
                        observed_flows_raw[i] = (cumulative / detector_interval) * 3600
                        # Store the raw measurement in the history buffer
                        sensing_history_buffers[i].append(observed_flows_raw[i])

                    for i, det in enumerate(debug_detectors): # logging debug detectors
                        cumulative = traci.inductionloop.getIntervalVehicleNumber(det)
                        debug_observed_flows_raw[i] = (cumulative / detector_interval) * 3600


                # --- PART C: PID UPDATE & ROUTE-SPECIFIC DELAY ---
                if time % detector_interval == 0 and time > 0:
                    target_idx = int(time // detector_interval)
                    route_proportions = [1] * len(routes)
                    try:
                        for det_route in detectors_to_routes:
                            shared_routes = detectors_to_routes[det_route]
                            for r in shared_routes:
                                route_idx = route_to_id[(r,)]
                                route_proportions[route_idx] = pid_flows[r][target_idx] / sum(pid_flows[s][target_idx] for s in shared_routes)
                    except:
                        for det_route in detectors_to_routes:
                            shared_routes = detectors_to_routes[det_route]
                            for r in shared_routes:
                                route_idx = route_to_id[(r,)]
                                route_proportions[route_idx] = 1 / len(shared_routes)
                    
                    ###### COMMENT OUT IF NOT DOING ROUTE ABLATION STUDY ######
                    # route_proportions[2] = 0.5
                    # route_proportions[3] = 0.5
                    # route_proportions[4] = 0.7
                    ###### COMMENT OUT IF NOT DOING ROUTE ABLATION STUDY ######

                    ### ABLATION VERSION
                    MODE = "HYBRID"
                    ### FEEDFORWARD FEEDBACK VERSION
                    for i, det in enumerate(sensing_detectors):
                        if MODE == "HYBRID":
                            pid = pid_controllers_rectifier[i]
                            
                            # 1. Get Target
                            
                            target_flow = sum(
                                synth_data["volume"][measurement_locations.index(d)][target_idx] * route_proportions[i]
                                for d in det
                            )

                            

                            # 2. Calculate the Instantaneous Residual (Error)
                            # We measure the gap between the target and the raw observed flow
                            raw_observed = observed_flows_raw[i]

                            current_input = -(target_flow - raw_observed) 
                            sensing_history_buffers[i].append(current_input)

                            # 4. Smooth the actual input, not the raw error
                            history = sensing_history_buffers[i]
                            smoothed_input = sum(history) / len(history)
                            
                            # 4. PID Update
                            # We want the PID to drive the ERROR to zero.
                            pid.setpoint = 0 
                            
                            # Expand limits to allow the "Rectifier" full authority to fix the flow
                            # This fixes the "flow too low" issue.
                            pid.output_limits = (-2100, 2100) 
                            
                            # Feed negative error because PID calculates: setpoint - input
                            # So: 0 - (-smoothed_error) = +smoothed_error
                            raw_control_signal = pid(smoothed_input)
                            
                            # 5. Apply Delay Logic per Route
                            d_steps = int(transfer_delays[i] // detector_interval)
                            if d_steps > 0:
                                action_buffers[i].append(raw_control_signal)
                                delayed_signal = action_buffers[i].popleft()
                            else:
                                delayed_signal = raw_control_signal
                            
                            # 6. ADDITIVE INJECTION UPDATE (The PPI Formula)
                            # New Total = Prediction Baseline + Rectifier Correction
                            new_total = max(0, target_flow + delayed_signal)
                            max_allowable_flow = 2100 * num_lanes[i]
                            # Physical cap for a single lane (adjust if multi-lane)
                            new_total = min(new_total, max_allowable_flow)
                            
                            # Distribute the total flow evenly across the route options
                            current_injected_flows[i] = [new_total / len(controlled_flow_routes[i])] * len(controlled_flow_routes[i])

                            # Log metrics for variance analysis
                            # Note: Writing target_flow - smoothed_error gives you the "Corrected Baseline"
                            writer_sensor.writerow([
                                step, time, det, target_flow, raw_observed, 
                                smoothed_input, raw_control_signal, delayed_signal, new_total, max_allowable_flow
                            ])
                        elif MODE == "FEEDBACK":
                            pid = pid_controllers_feedback[i]
                            
                            # 1. Get Target
                            
                            target_flow = sum(
                                synth_data["volume"][measurement_locations.index(d)][target_idx] * route_proportions[i]
                                for d in det
                            )

                            # 2. Calculate the Instantaneous Residual (Error)
                            # We measure the gap between the target and the raw observed flow
                            raw_observed = observed_flows_raw[i]
                            
                            # 3. Update Error History & Calculate Smoothed Error
                            # This removes sensor noise without lagging the response to target changes
                            sensing_history_buffers[i].append(raw_observed)
                            history = sensing_history_buffers[i]
                            smoothed_observation = sum(history) / len(history)
                            
                            # 4. PID Update
                            # We want the PID to drive the ERROR to zero.
                            pid.setpoint = target_flow
                            
                            # Expand limits to allow the "Rectifier" full authority to fix the flow
                            # This fixes the "flow too low" issue.
                            
                            # Feed negative error because PID calculates: setpoint - input
                            # So: 0 - (-smoothed_error) = +smoothed_error
                            raw_control_signal = pid(smoothed_observation)
                            
                            # 5. Apply Delay Logic per Route
                            if d_steps > 0:
                                action_buffers[i].append(raw_control_signal)
                                delayed_signal = action_buffers[i].popleft()
                            else:
                                delayed_signal = raw_control_signal
                            
                            # New Total = Prediction Baseline + Rectifier Correction
                            new_total = max(0, current_injected_flows[i][0] + delayed_signal)
                            max_allowable_flow = 2100 * num_lanes[i]
                            # Physical cap for a single lane (adjust if multi-lane)
                            new_total = min(new_total, max_allowable_flow)
                            
                            # Distribute the total flow evenly across the route options
                            current_injected_flows[i] = [new_total / len(controlled_flow_routes[i])] * len(controlled_flow_routes[i])

                            # Log metrics for variance analysis
                            # Note: Writing target_flow - smoothed_error gives you the "Corrected Baseline"
                            writer_sensor.writerow([
                                step, time, route_id, target_flow, raw_observed, 
                                smoothed_observation, raw_control_signal, delayed_signal, new_total, max_allowable_flow
                            ])
                        else:

                            target_flow = sum(
                                synth_data["volume"][measurement_locations.index(d)][target_idx] * route_proportions[i]
                                for d in det
                            )

                            # 2. Skip Error Calculation & PID Update
                            # In Feedforward, we don't care about 'raw_observed' or 'instant_error'.
                            # We set the correction (delayed_signal) to 0.
                            delayed_signal = 0 
                            
                            # 3. Injection Update
                            # New Total is just the target. No PID correction is added.
                            new_total = max(0, target_flow)
                            
                            # 4. Physical Constraints
                            # We still apply these so we don't crash the simulation with impossible volumes.
                            max_allowable_flow = 2100 * num_lanes[i]
                            new_total = min(new_total, max_allowable_flow)
                            
                            # 5. Distribute the flow
                            current_injected_flows[i] = [new_total / len(controlled_flow_routes[i])] * len(controlled_flow_routes[i])

                            # 6. Simplified Log
                            # raw_observed is recorded only for your analysis to see how much 'error' 
                            # existed that you chose NOT to fix.
                            writer_sensor.writerow([
                                step, time, det, target_flow, observed_flows_raw[i], 
                                0, 0, 0, new_total, max_allowable_flow
                            ])

                    # for i, det in enumerate(debug_detectors):
                        
                    #     # 1. Get Target
                    #     target_idx = int(time // detector_interval)
                    #     target_flow = synth_data["volume"][measurement_locations.index(det)][target_idx]

                    #     debug_raw_observed = debug_observed_flows_raw[i]
       
                    #     # Log metrics for variance analysis
                    #     # Note: Writing target_flow - smoothed_error gives you the "Corrected Baseline"
                    #     writer_debug.writerow([
                    #         step, time, det, target_flow, debug_raw_observed 
                    #     ])
                        
                    for group_idx, det_tuple in enumerate(route_tracking_sensors or []):
                        counts = interval_route_counts[det_tuple]
                        total = sum(counts)
                        print(counts)
                        proportions = [(c / total if total > 0 else 0) for c in counts]
                        
                        writer_tracking.writerow([step, time, det_tuple, total] + proportions)
                        # Reset for next interval
                        interval_route_counts[det_tuple] = [0]*len(controlled_flow_routes)
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

def probe_simulation_transfer_delays(sim_config, controlled_flow_routes, sensing_detectors, SUMO_EXE):
    """
    Runs a brief probe simulation to find travel time from route start to detector.
    """
    print("--- Running Pre-Analysis to determine Transfer Delays ---")
    # Run headless sumo for speed
    traci.start([SUMO_EXE, "-c", sim_config, "--no-step-log", "true", "--waiting-time-memory", "0"])
    
    delays = []
    
    for i, route_tuple in enumerate(controlled_flow_routes):
        route_id = route_tuple[0]  # Take the first route in the tuple
        det_tuple = sensing_detectors[i]
        
        veh_id = f"probe_{route_id}"
        # Insert vehicle at time 0
        traci.vehicle.add(veh_id, route_id, typeID="hdv", departLane="best", departSpeed="max")
        
        start_time = traci.simulation.getTime()
        travel_time = 0
        found = False
        
        # Run until the vehicle hits the detector or 10 minutes pass (timeout)
        while traci.simulation.getTime() < 600:
            traci.simulationStep()
            # Check if our probe vehicle is on any of the target detectors
            for d_id in det_tuple:
                if veh_id in traci.inductionloop.getLastStepVehicleIDs(d_id):
                    travel_time = traci.simulation.getTime() - start_time
                    found = True
                    break
            if found: break
        
        if found:
            print(f"Route {route_id} delay to {det_tuple[0]}: {travel_time}s")
            delays.append(int(travel_time))
        else:
            print(f"Warning: Probe for {route_id} timed out. Using default 30s.")
            delays.append(30)
            
    traci.close()
    return delays

def route_correlations(sim_config, routes, detector_list, duration=1200):
    all_results = []

    for r_id in routes:
        print(f"--- Analyzing Route: {r_id} ---")
        # Start a fresh simulation for EACH route to isolate the signal
        traci.start(["sumo", "-c", sim_config, "--no-step-log", "true"])
        
        input_signal = []
        # Dictionary to store output for all detectors for THIS specific route
        route_output = {d: [] for d in detector_list}

        for t in range(duration):
            traci.simulationStep()
            
            # 1. Inject White Noise for the current route ONLY
            flow_rate = np.random.poisson(2) 
            input_signal.append(flow_rate)
            
            for _ in range(flow_rate):
                veh_id = f"{r_id}_{t}_{_}"
                traci.vehicle.add(veh_id, r_id, departSpeed="max", typeID="hdv")
            
            # 2. Record all detectors
            for d_id in detector_list:
                val = traci.inductionloop.getLastStepVehicleNumber(d_id)
                route_output[d_id].append(val)
        
        traci.close()

        # 3. Perform Cross-Correlation for this isolated route
        for d_id, out_sig in route_output.items():
            # Standardize signals (mean subtraction)
            x = np.array(input_signal) - np.mean(input_signal)
            y = np.array(out_sig) - np.mean(out_sig)
            
            # Compute correlation
            correlation = np.correlate(y, x, mode='full')
            lags = np.arange(-len(x) + 1, len(x))
            
            peak_idx = np.argmax(correlation)
            best_lag = lags[peak_idx]
            peak_val = correlation[peak_idx]

            # Only store causal relationships (lag > 0)
            if best_lag > 0 and peak_val > 0:
                all_results.append({
                    'route': r_id, 
                    'detector': d_id, 
                    'lag': best_lag, 
                    'score': peak_val
                })

    return pd.DataFrame(all_results)

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

def save_sim_to_rds_csv(detector_data, measurement_locations, output_filename="simulated_rds_data.csv", interval_seconds=30):
    """
    Converts detector_data dict into the specific CSV format required by od_estimation.
    """
    # 1. Define the mapping from your simulation detector IDs to the RDS labels
    # ADJUST THESE to match your SUMO detector IDs!

    mapping = {
        # 55.3 (4 lanes)
        '553-westbound_0': {"milemarker": 55.3, "link_name": "55_3", "lane": 1}, 
        '553-westbound_1': {"milemarker": 55.3, "link_name": "55_3", "lane": 2}, 
        '553-westbound_2': {"milemarker": 55.3, "link_name": "55_3", "lane": 3}, 
        '553-westbound_3': {"milemarker": 55.3, "link_name": "55_3", "lane": 4}, 
        # 55.5 (5 lanes)
        '555-westbound_0': {"milemarker": 55.5, "link_name": "55_5", "lane": 1}, 
        '555-westbound_1': {"milemarker": 55.5, "link_name": "55_5", "lane": 2}, 
        '555-westbound_2': {"milemarker": 55.5, "link_name": "55_5", "lane": 3}, 
        '555-westbound_3': {"milemarker": 55.5, "link_name": "55_5", "lane": 4}, 
        '555-westbound_4': {"milemarker": 55.5, "link_name": "55_5", "lane": 5}, 
        # 55.6 (5 lanes)
        '556-westbound_0': {"milemarker": 55.6, "link_name": "55_6", "lane": 1}, 
        '556-westbound_1': {"milemarker": 55.6, "link_name": "55_6", "lane": 2}, 
        '556-westbound_2': {"milemarker": 55.6, "link_name": "55_6", "lane": 3}, 
        '556-westbound_3': {"milemarker": 55.6, "link_name": "55_6", "lane": 4}, 
        '556-westbound_4': {"milemarker": 55.6, "link_name": "55_6", "lane": 5}, 
        # 55.8 (5 lanes)
        '558-westbound_0': {"milemarker": 55.8, "link_name": "55_8_ramp", "lane": 1},
        '558-westbound_1': {"milemarker": 55.8, "link_name": "55_8", "lane": 2},
        '558-westbound_2': {"milemarker": 55.8, "link_name": "55_8", "lane": 3},
        '558-westbound_3': {"milemarker": 55.8, "link_name": "55_8", "lane": 4},
        '558-westbound_4': {"milemarker": 55.8, "link_name": "55_8", "lane": 5},
        # 55.9 (4 lanes)
        '559-westbound_0': {"milemarker": 55.9, "link_name": "55_9", "lane": 1},
        '559-westbound_1': {"milemarker": 55.9, "link_name": "55_9", "lane": 2},
        '559-westbound_2': {"milemarker": 55.9, "link_name": "55_9", "lane": 3},
        '559-westbound_3': {"milemarker": 55.9, "link_name": "55_9", "lane": 4},
        # 56.0 (5 lanes)
        '560-westbound_0': {"milemarker": 56.0, "link_name": "56_0_off", "lane": 1},
        '560-westbound_1': {"milemarker": 56.0, "link_name": "56_0", "lane": 2},
        '560-westbound_2': {"milemarker": 56.0, "link_name": "56_0", "lane": 3},
        '560-westbound_3': {"milemarker": 56.0, "link_name": "56_0", "lane": 4},
        '560-westbound_4': {"milemarker": 56.0, "link_name": "56_0", "lane": 5},
        # 56.1 (4 lanes)
        '561-westbound_0': {"milemarker": 56.1, "link_name": "56_1", "lane": 1},
        '561-westbound_1': {"milemarker": 56.1, "link_name": "56_1", "lane": 2},
        '561-westbound_2': {"milemarker": 56.1, "link_name": "56_1", "lane": 3},
        '561-westbound_3': {"milemarker": 56.1, "link_name": "56_1", "lane": 4},
        # 56.2 (4 lanes)
        '562-westbound_0': {"milemarker": 56.2, "link_name": "56_2", "lane": 1},
        '562-westbound_1': {"milemarker": 56.2, "link_name": "56_2", "lane": 2},
        '562-westbound_2': {"milemarker": 56.2, "link_name": "56_2", "lane": 3},
        '562-westbound_3': {"milemarker": 56.2, "link_name": "56_2", "lane": 4},
        # 56.3 (4 lanes)
        '563-westbound_0': {"milemarker": 56.3, "link_name": "56_3", "lane": 1},
        '563-westbound_1': {"milemarker": 56.3, "link_name": "56_3", "lane": 2},
        '563-westbound_2': {"milemarker": 56.3, "link_name": "56_3", "lane": 3},
        '563-westbound_3': {"milemarker": 56.3, "link_name": "56_3", "lane": 4},
        # 56.4 (5 lanes)
        '564-westbound_0': {"milemarker": 56.4, "link_name": "56_4_on", "lane": 1},
        '564-westbound_1': {"milemarker": 56.4, "link_name": "56_4", "lane": 2},
        '564-westbound_2': {"milemarker": 56.4, "link_name": "56_4", "lane": 3},
        '564-westbound_3': {"milemarker": 56.4, "link_name": "56_4", "lane": 4},
        '564-westbound_4': {"milemarker": 56.4, "link_name": "56_4", "lane": 5},
        # 56.5 (4 lanes)
        '565-westbound_0': {"milemarker": 56.5, "link_name": "56_5", "lane": 1},
        '565-westbound_1': {"milemarker": 56.5, "link_name": "56_5", "lane": 2},
        '565-westbound_2': {"milemarker": 56.5, "link_name": "56_5", "lane": 3},
        '565-westbound_3': {"milemarker": 56.5, "link_name": "56_5", "lane": 4},
    }

    # Calculate fixed number of intervals for 24 hours
    # 24 hours * 3600 seconds / interval_seconds
    intervals = int(num_timesteps / interval_seconds) 
    
    num_detectors = detector_data['speed'].shape[0]
    actual_intervals = detector_data['speed'].shape[1]
    
    all_rows = []
    base_time = datetime.strptime("00:00:00", "%H:%M:%S")

    for t in range(intervals):
        current_time_str = (base_time + timedelta(seconds=interval_seconds * t)).strftime("%H:%M:%S")
        
        for i in range(num_detectors):
            
            det_id = measurement_locations[i]
            if det_id not in mapping:
                continue

            #print(f"Processing detector {det_id} at time {current_time_str} i {i})")
            
            metadata = mapping[det_id]
            
            # Use actual data if available, otherwise default to 0
            if t < actual_intervals:
                spd = detector_data["speed"][i, t]
                vol = detector_data["volume"][i, t]
                occ = detector_data["occupancy"][i, t]
            else:
                spd, vol, occ = 0.0, 0.0, 0.0 # Default padding
            
            all_rows.append({
                "timestamp": current_time_str,
                "link_name": metadata["link_name"],
                "milemarker": metadata["milemarker"],
                "lane": metadata["lane"],
                "speed": spd,
                "volume": vol,
                "occupancy": occ
            })

    df_out = pd.DataFrame(all_rows)
    
    df_out.to_csv(output_filename, index=False)
    print(f"Fixed 24h file saved. Total intervals: {intervals}")
    print(f"Time range: {df_out['timestamp'].iloc[0]} to {df_out['timestamp'].iloc[-1]}")

def extract_detector_mapping(network_xml_file):
    """
    Parses a network or detector XML file to build a mapping dictionary.
    Format: {"detector_id": ("lane_id", "position")}
    """
    detector_mapping = {}
    
    try:
        # Load and parse the XML
        tree = ET.parse(network_xml_file)
        root = tree.getroot()
        
        # We look for inductionLoop, e1Detector, or detector tags
        # Depending on your file, you can add more tag types here
        detector_tags = root.findall('.//inductionLoop') + \
                        root.findall('.//e1Detector') + \
                        root.findall('.//detector')

        for det in detector_tags:
            det_id = det.get('id')
            lane_id = det.get('lane')
            # Extract position and convert to string (dropping decimals if needed)
            pos = str(int(float(det.get('pos', 0)))) 
            
            if det_id and lane_id:
                detector_mapping[det_id] = (lane_id, pos)
                
        return detector_mapping

    except FileNotFoundError:
        print(f"Error: File {network_xml_file} not found.")
        return {}
    except ET.ParseError:
        print(f"Error: Failed to parse XML. Is it a valid network file?")
        return {}

def generate_fcd(csv_file, fcd_file="fcd_output/rds_fcd.xml", start_window=None, end_window=None):
    csv_data = pd.read_csv(csv_file, delimiter=';')

    # --- NEW: Filter by time window (before normalization) ---
    # Assuming start_window and end_window are in the same units as the CSV 'Time' column
    if start_window is not None:
        csv_data = csv_data[csv_data['Time'] >= start_window]
    if end_window is not None:
        csv_data = csv_data[csv_data['Time'] <= end_window]

    # Convert to seconds
    csv_data['Time'] = csv_data['Time'] * 60
    
    # Normalize to start at 0 (relative to your chosen start window)
    if not csv_data.empty:
        start_time = csv_data['Time'].min()
        csv_data['Time'] = csv_data['Time'] - start_time
    else:
        print("Warning: No data found in the specified time window.")
        return

    print(f"Processing {len(csv_data)} rows. First timestep: {csv_data['Time'].min()}")

    # ... rest of your code (extract mapping, root element, etc.) ...
    detector_mapping = extract_detector_mapping('12-15_detectors_gt.xml')
    root = ET.Element("fcd-export")

    for time, group in csv_data.groupby('Time'):
        timestep = ET.SubElement(root, "timestep", time=str(time))
        for _, row in group.iterrows():
            csv_id = str(row['Detector']).replace(".", "_")
            if csv_id in detector_mapping:
                lane_id, position = detector_mapping[csv_id]
                speed_ms = round(row['vPKW'] / 3.6, 2)
                
                if speed_ms > 0:
                    ET.SubElement(timestep, "vehicle", {
                        "id": str(row['Detector']),
                        "lane": lane_id,
                        "speed": str(speed_ms),
                        "pos": position,
                        "type": "PKW"
                    })

    # Save and Prettify
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(fcd_file, "w") as f:
        f.write(xml_str)

def run_sumo_od_estimation(sim_config,measured_data,tripinfo_output,
            fcd_output,det_interval,sensing_detectors,tracking_log):
    save_sim_to_rds_csv(measured_data, measurement_locations, output_filename="simulated_rds_data.csv")
    _ = od_estimation_large('simulated_rds_data.csv', plot=True, write_rou_xml=True)

    command = [SUMO_EXE, '-c', sim_config, 
               '--no-step-log', '--xml-validation', 'never', 
               '--lateral-resolution', '0.5']
    
    if tripinfo_output:
        command.extend(['--tripinfo-output', tripinfo_output])
    if fcd_output:
        command.extend(['--fcd-output', fcd_output])

    # 2. Start SUMO via TraCI
    traci.start(command)
    
    # Setup Logging (matching your PID CSV format)
    # This ensures your "Baseline" data looks exactly like your "PID" data
    f_log = open(tracking_log, 'w', newline='')
    writer = csv.writer(f_log)
    writer.writerow(["step", "time", "sensors", "target", "observed", "speed"])

    step = 0
    end_time = num_timesteps # Set your desired simulation end time in seconds
    
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            time = traci.simulation.getTime()
            
            # 3. Check if we reached the measurement interval
            if time % detector_interval == float(detector_interval - 1):
                for i, det_tuple in enumerate(sensing_detectors):
                    
                    # SUM the vehicle counts from every lane in the tuple
                    # This gives you the total flow for that cross-section
                    cumulative = sum(
                        traci.inductionloop.getLastIntervalVehicleNumber(d_id) 
                        for d_id in det_tuple
                    )
                    observed_flows = (cumulative / detector_interval) * 3600

                    
                    # Calculate target_flow from your synth_data to keep the CSV consistent
                    # Assuming target_idx aligns with your 30s buckets
                    target_idx = int(time // det_interval) - 1
                    
                    # We use the same lookup logic as your PID script
                    current_target = sum(
                        synth_data["volume"][measurement_locations.index(d)][target_idx]
                        for d in det_tuple
                    )

                    # Log the results
                    writer.writerow([step, time, str(det_tuple), current_target, observed_flows])
            
            step += 1
            if time > end_time:
                break
                
    finally:
        traci.close()
        f_log.close()

def save_sim_to_rds_fr(detector_data, measurement_locations, output_filename="fr_intermediate.csv", interval_seconds=30):
    """
    Converts detector_data dict into the specific CSV format required by od_estimation.
    """

    # Calculate fixed number of intervals for 24 hours
    # 24 hours * 3600 seconds / interval_seconds
    intervals = int(num_timesteps / interval_seconds) 
    
    num_detectors = detector_data['speed'].shape[0]
    actual_intervals = detector_data['speed'].shape[1]
    
    all_rows = []

    for t in range(intervals):
        
        for i in range(num_detectors):
            det_id = measurement_locations[i]
            
            
            # Use actual data if available, otherwise default to 0
            if t < actual_intervals:
                spd = detector_data["speed"][i, t]
                vol = detector_data["volume"][i, t]
                occ = detector_data["occupancy"][i, t]
            else:
                spd, vol, occ = 0.0, 0.0, 0.0 # Default padding
            
            all_rows.append({
                "Detector": det_id,
                "Time": t*0.5,
                "vPKW": spd,
                "qPKW": (vol / 60) * 0.5,
            })

    df_out = pd.DataFrame(all_rows)
    cwd = os.getcwd()
        
    df_out.to_csv(output_filename, sep = ";", index=False)

    sumo_home = os.environ.get('SUMO_HOME')
    if not sumo_home:
        print("Error: SUMO_HOME environment variable is not set.")
        return

    # 2. Construct the path to flowrouter.py
    # Using os.path.join ensures it works on both Windows and Linux
    script_path = os.path.join(sumo_home, 'tools', 'detector', 'flowrouter.py')
    
    # 3. Define the command arguments
    # Note: Ensure these files exist in your current working directory
    command = [
        sys.executable, script_path,
        '-n', 'i24b.net.xml',
        '-d', '12-15_detectors.xml',
        '-f', 'fr_intermediate.csv',
        '-o', 'routes.xml',
        '-e', 'flows.xml',
        '-i', '0.5'
    ]

    try:
        print(f"Running: {' '.join(command)}")
        # 4. Execute the command
        result = subprocess.run(
            command, 
            check=True,          # Raises CalledProcessError if the command fails
            capture_output=True, # Captures stdout and stderr
            text=True            # Returns output as string instead of bytes
        )
        
        print("Flowrouter completed successfully!")
        print("Output:", result.stdout)

    except subprocess.CalledProcessError as e:
        print("Error during flowrouter execution:")
        print(e.stderr)
    
    tree = ET.parse("routes.xml")
    root = tree.getroot()

    # Parse the flows file
    flow_tree = ET.parse("flows.xml")
    flow_root = flow_tree.getroot()

    vtype_str = """
    <vType id="hdv" length="4.3" carFollowModel="IDM" emergencyDecel="4.0" 
        laneChangeModel="SL2015" latAlignment="arbitrary" lcKeepRight="0.0" 
        lcOvertakeRight="0.0" maxSpeed="30.55" minGap="2.5" accel="1.5" 
        decel="2" tau="1.4" lcSublane="1.0" maxSpeedLat="1.4" 
        lcAccelLat="0.7" minGapLat="0.4" lcStrategic="10.0" 
        lcCooperative="1.0" lcPushy="0.4" lcImpatience="0.9" lcSpeedGain="1.5">
        <param key="has.fcd.device" value="true" />
    </vType>
    """

    # 3. Convert string to an element and insert at the VERY TOP of the root
    vtype_element = ET.fromstring(vtype_str)
    root.insert(0, vtype_element)

    # Append every flow/vehicle from the second file into the first
    for element in flow_root:
        root.append(element)

    # Write the combined result
    tree.write("i24_flowrouter.rou.xml", encoding="UTF-8", xml_declaration=True)
    print(f"Success! Created i24_flowrouter.rou.xml")
    print(f"Fixed 24h file saved. Total intervals: {intervals}")
    print(f"Time range: {df_out['Time'].iloc[0]} to {df_out['Time'].iloc[-1]}")


if __name__ == "__main__":
    measurement_locations = extract_detector_locations("../../data/RDS/detections_0360-0600.csv")
    print(measurement_locations)
    cmd_probs = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    ## SCRIPT CONFIGS ##
    RERUN_GT = False # whether to rerun the ground truth simulation and regenerate synthetic measurements (set to False to save time if already done)
    REAL_DATA = False
    method = "FLOWROUTER" # or "FLOWROUTER" "OD_ESTIMATION"

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
    if REAL_DATA:
        measured_output = reader.extract_rds_measurements(rds_file, measurement_locations, start_min=360, end_min=390)
        print("measured_output:", measured_output)
        if method == "FLOWROUTER":
            save_sim_to_rds_fr(measured_output, measurement_locations, output_filename="fr_intermediate.csv", interval_seconds=30)
        save_sim_to_rds_csv(measured_output, measurement_locations, output_filename="simulated_rds_data.csv")
        generate_fcd(rds_file, fcd_file="fcd_output/rds_fcd.xml", start_window=360, end_window=390)
        pid_flows = od_estimation_large("simulated_rds_data.csv", plot=True, write_rou_xml=True)
    else:
        if RERUN_GT:
            run_sumo(SCENARIO+'_gt.sumocfg', tripinfo_output="tripinfo_gt.xml", fcd_output="fcd_output/fcd_sim.xml")
        measured_output = reader.extract_sim_meas(measurement_locations, file_dir = "det/")
        save_sim_to_rds_csv(measured_output, measurement_locations, output_filename="simulated_rds_data.csv")
        if method == "FLOWROUTER":
            save_sim_to_rds_fr(measured_output, measurement_locations, output_filename="fr_intermediate.csv", interval_seconds=30)
        pid_flows = od_estimation_large('simulated_rds_data.csv', plot=True, write_rou_xml=True)


    synth_data = measured_output
    controlled_flows_route = [("1",), ("3",), ("4",), ("9",), 
                              ("10",), ("11",), ("12",), ("13",), ("19",),
                              ("20",), ("21",), ("23",), ("25",)]

    detectors_to_routes = {("565-westbound_0", "565-westbound_1", "565-westbound_2", "565-westbound_3"): ["25"],
                           ("564-westbound_0",): ["11", "12", "13", "23"],
                           ("560-westbound_0",): ["19", "20"],
                           ("558-westbound_0",): ["3", "4", "21", "9", "10"],
                           ("556-westbound_0",): ["1"]}
    
    map_keys = list(detectors_to_routes.keys())

    # The resulting list
    sensing_detectors = []

    for route_tuple in controlled_flows_route:
        route_id = route_tuple[0]
        matched_detector = None
        
        # Find which detector key contains this route ID
        for idx, key in enumerate(map_keys):
            if route_id in detectors_to_routes[key]:
                print(route_id, detectors_to_routes[key])
                matched_detector = key
                break
        
        sensing_detectors.append(matched_detector)

    print(sensing_detectors)
    
    num_lanes = [1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 4]
    
    

    debug_detectors = []
    pid_controllers_rectifier = [PID(0.8, 0.0, 0.0, setpoint=0.0), 
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0.0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0.0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0.0, 0, setpoint=0.0),]
    
    pid_controllers_feedback = [PID(1.2, 0.0, 0.0, setpoint=0.0), 
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0.0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0.0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0.0, 0, setpoint=0.0),]
    
    

    delays = probe_simulation_transfer_delays(SCENARIO+".sumocfg", controlled_flows_route, sensing_detectors, SUMO_EXE)
    no_delays = [1] * len(controlled_flows_route)
    #delays = [0] * len(controlled_flows_route)
    ## to do: add a configuration checking function here
    if method == "PID":
        log = "pid_log_sim_3hr.csv" if not REAL_DATA else "pid_log_rds.csv"
        run_PID_closed_loop_sumo(
            sim_config=SCENARIO + ".sumocfg", 
            controlled_flow_routes=controlled_flows_route, 
            sensing_detectors=sensing_detectors, 
            debug_detectors=debug_detectors, 
            pid_controllers_feedback=pid_controllers_feedback, 
            pid_controllers_rectifier=pid_controllers_rectifier,
            synth_data=synth_data, 
            fcd_output="fcd_output/fcd_pid_sim_3hr.xml",
            # --- New Parameters Added Below ---
            measurement_locations=measurement_locations, # List of detector IDs mapping synth_data
            detector_interval=detector_interval,         # e.g., 60 (seconds)
            num_timesteps=num_timesteps,                 # Total simulation duration
            step_length=step_length,                     # e.g., 1.0 or 0.5
            SUMO_EXE=SUMO_EXE,                           # Path to sumo or sumo-gui executable
            transfer_delays=no_delays,                          # The delay in seconds (adjust as needed)
            ma_windows= [30] * len(controlled_flows_route),             # Moving average window in seconds (adjust as needed)
            num_lanes = num_lanes,
            route_tracking_sensors=measurement_locations,
            routes=controlled_flows_route,
            pid_flows=pid_flows,
            detectors_to_routes=detectors_to_routes,
            pid_sensor_log=log)
    elif method == "FLOWROUTER":
        run_sumo_flowrouter(SCENARIO + "_fr.sumocfg", 
            tripinfo_output=None, 
            fcd_output="fcd_output/fcd_fr_sim_3hr.xml", 
            det_interval=30,
            sensing_detectors=sensing_detectors, 
            tracking_log = "fr_log.csv")
    elif method == "OD_ESTIMATION":
        run_sumo_od_estimation(
            sim_config = SCENARIO+"_od.sumocfg",
            measured_data = synth_data,
            tripinfo_output=None,
            fcd_output="fcd_output/fcd_od_sim_3hr.xml",
            det_interval=30,
            sensing_detectors=sensing_detectors,
            tracking_log="od_log.csv"
        )