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

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_data_read as reader
from utils_vis import od_estimation

# import utils_macro as macro
# import utils_vis as vis

# ================ CONFIGURATION ====================
with open('../config.json', 'r') as config_file:
    config = json.load(config_file)

computer_name = os.environ.get('HOSTNAME', 'Unknown')

if "CSI" in computer_name:
    SUMO_EXE = config['SUMO_PATH']["CSI"]
elif "VMS" in computer_name:
    SUMO_EXE = config['SUMO_PATH']["VMS"]
else: # run on SOL
    SUMO_EXE = config['SUMO_PATH']["SOL"]

SCENARIO = "i24"
EXP = "3b" # experiment label
N_TRIALS = 20000 # config["N_TRIALS"] # optimization trials
N_JOBS = 120 # config["N_JOBS"] # cores
RDS_DIR = config["i24"]["RDS_DIR"]
# RDS_DIR = os.path.join("../..", "data/RDS/I24_WB_52_60_11132023.csv")
# ================================================

# follows convention e.g., 56_7_0, milemarker 56.7, lane 1
measurement_locations = [
                         '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                        #  '54_1_0', '54_1_1', '54_1_2', '54_1_3'
                        ]

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

initial_guess = {key: DEFAULT_PARAMS[key] for key in param_names if key in DEFAULT_PARAMS}

num_timesteps = config["i24"]["SIMULATION_TIME"]
step_length = config["i24"]["STEP_LENGTH"]
detector_interval = config["i24"]["DETECTOR_INTERVAL"]

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
                             debug_detectors, pid_controllers_feedback, pid_controllers_rectifier, synth_data, 
                             measurement_locations, detector_interval, num_timesteps, 
                             step_length, SUMO_EXE, transfer_delays=None, 
                             ma_windows=None, num_lanes=None, # New argument: List of seconds for MA
                             route_tracking_sensors=None, routes=None, pid_flows=None,
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

    pid_sensor_log = "pid_log_sim_3hr.csv"
    header_pid = ['step', 'time', 'sensors', 'target', 'observed', 'smoothed_error', 'raw_control_signal', 'delayed_signal', 'new_total_injection']
    header_tracking = ['step', 'time', 'sensor_id', 'total_count'] + [f'route_{i}_prop' for i in range(len(controlled_flow_routes))]
    header_debug = ['step', 'time', 'sensors', 'target', 'observed']

    detector_sharing_map = {}
    for i, det_tuple in enumerate(sensing_detectors):
        for det_id in det_tuple:
            if det_id not in detector_sharing_map:
                detector_sharing_map[det_id] = [routes[i]]
            else:
                detector_sharing_map[det_id].append(routes[i])
    print(detector_sharing_map)

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
                        route_proportions[1] = pid_flows['r_1'][target_idx] / (pid_flows['r_1'][target_idx] + pid_flows['r_2'][target_idx])
                        route_proportions[2] = pid_flows['r_2'][target_idx] / (pid_flows['r_1'][target_idx] + pid_flows['r_2'][target_idx])
                        route_proportions[3] = pid_flows['r_3'][target_idx] / (pid_flows['r_3'][target_idx] + pid_flows['r_4'][target_idx])
                        route_proportions[4] = pid_flows['r_4'][target_idx] / (pid_flows['r_3'][target_idx] +pid_flows['r_4'][target_idx])
                    except:
                        route_proportions[1] = 0.5
                        route_proportions[2] = 0.5
                        route_proportions[3] = 0.5  
                        route_proportions[4] = 0.5

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
                        route_proportions[1] = pid_flows['r_1'][target_idx] / (pid_flows['r_1'][target_idx] + pid_flows['r_2'][target_idx])
                        route_proportions[2] = pid_flows['r_2'][target_idx] / (pid_flows['r_1'][target_idx] + pid_flows['r_2'][target_idx])
                        route_proportions[3] = pid_flows['r_3'][target_idx] / (pid_flows['r_3'][target_idx] + pid_flows['r_4'][target_idx])
                        route_proportions[4] = pid_flows['r_4'][target_idx] / (pid_flows['r_3'][target_idx] +pid_flows['r_4'][target_idx])
                    except:
                        route_proportions[1] = 0.5
                        route_proportions[2] = 0.5
                        route_proportions[3] = 0.5  
                        route_proportions[4] = 0.5
                    
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
                                step, time, det, target_flow, raw_observed, 
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
        "54_6_0": {"milemarker": 54.6, "link_name": "54_6_0", "lane": 1},
        "54_6_1": {"milemarker": 54.6, "link_name": "54_6_1", "lane": 2},
        "54_6_2": {"milemarker": 54.6, "link_name": "54_6_2", "lane": 3},
        "54_6_3": {"milemarker": 54.6, "link_name": "54_6_3", "lane": 4},
        "55_3_0": {"milemarker": 55.3, "link_name": " R3G-00I24-55.3W (259)", "lane": 1},
        "55_3_1": {"milemarker": 55.3, "link_name": " R3G-00I24-55.3W (259)", "lane": 2},
        "55_3_2": {"milemarker": 55.3, "link_name": " R3G-00I24-55.3W (259)", "lane": 3},
        "55_3_3": {"milemarker": 55.3, "link_name": " R3G-00I24-55.3W (259)", "lane": 4},
        "56_0_0": {"milemarker": 56.0, "link_name": " R3G-00I24-56.0W (262)", "lane": 1},
        "56_0_1": {"milemarker": 56.0, "link_name": " R3G-00I24-56.0W (262)", "lane": 2},
        "56_0_2": {"milemarker": 56.0, "link_name": " R3G-00I24-56.0W (262)", "lane": 3},
        "56_0_3": {"milemarker": 56.0, "link_name": " R3G-00I24-56.0W (262)", "lane": 4},
        "56_0_4": {"milemarker": 56.0, "link_name": " R3G-00I24-56.0W Off Ramp (262)", "lane": 5},
        "56_3_0": {"milemarker": 56.3, "link_name": "56_3_0", "lane": 1},
        "56_3_1": {"milemarker": 56.3, "link_name": "56_3_1", "lane": 2},
        "56_3_2": {"milemarker": 56.3, "link_name": "56_3_2", "lane": 3},
        "56_3_3": {"milemarker": 56.3, "link_name": "56_3_3", "lane": 4},
        "56_3_4": {"milemarker": 56.3, "link_name": "56_3_4", "lane": 5},
        "56_7_0": {"milemarker": 56.7, "link_name": " R3G-00I24-56.7W (267)", "lane": 1},
        "56_7_1": {"milemarker": 56.7, "link_name": " R3G-00I24-56.7W (267)", "lane": 2},
        "56_7_2": {"milemarker": 56.7, "link_name": " R3G-00I24-56.7W (267)", "lane": 3},
        "56_7_3": {"milemarker": 56.7, "link_name": " R3G-00I24-56.7W (267)", "lane": 4},
        "56_7_4": {"milemarker": 56.7, "link_name": " R3G-00I24-56.7W On Ramp (267)", "lane": 5},
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
        '-n', 'i24.net.xml',
        '-d', 'I24_RDS_gt.add.xml',
        '-f', 'fr_intermediate.csv',
        '-o', 'routes.xml',
        '-e', 'flows.xml',
        '-i', '5'
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


def generate_fcd(csv_file, fcd_file="fcd_output/rds_fcd.xml"):
    csv_data = pd.read_csv(csv_file, delimiter=';')

    csv_data['Time'] = csv_data['Time'] * 60
    
    # 2. Normalize to start at 0
    start_time = csv_data['Time'].min()
    csv_data['Time'] = csv_data['Time'] - start_time

    print(csv_data.head())

    
    # detector to lane mapping based on SUMO configuration
    detector_mapping = {
        # 56.7 RDS: 5 lanes
        "56_7_0": ("E1_5", "20"),
        "56_7_1": ("E1_4", "20"),
        "56_7_2": ("E1_3", "20"),
        "56_7_3": ("E1_2", "20"),
        "56_7_4": ("E1_1", "20"),

        # 56.3 RDS: 5 lanes
        "56_3_0": ("E3_4", "846"),
        "56_3_1": ("E3_3", "846"),
        "56_3_2": ("E3_2", "846"),
        "56_3_3": ("E3_1", "846"),
        "56_3_4": ("E3_0", "846"),

        # 56.0 RDS: 5 lanes
        "56_0_0": ("E3_4", "1329"),
        "56_0_1": ("E3_3", "1329"),
        "56_0_2": ("E3_2", "1329"),
        "56_0_3": ("E3_1", "1329"),
        "56_0_4": ("E3_0", "1329"),

        # 55.3 RDS: 4 lanes
        "55_3_0": ("E8_3", "130"),
        "55_3_1": ("E8_2", "130"),
        "55_3_2": ("E8_1", "130"),
        "55_3_3": ("E8_0", "130"),

        # 54.6 RDS: 4 lanes
        "54_6_0": ("E8_3", "1080"),
        "54_6_1": ("E8_2", "1080"),
        "54_6_2": ("E8_1", "1080"),
        "54_6_3": ("E8_0", "1080"),
    }
    # Create the root element
    root = ET.Element("fcd-export")

    # Group by Time (Timesteps)
    for time, group in csv_data.groupby('Time'):
        timestep = ET.SubElement(root, "timestep", time=str(time))
        
        for _, row in group.iterrows():
            # Normalize ID: Change '54.6_0' to '54_6_0' to match the dictionary
            csv_id = str(row['Detector']).replace(".", "_")
            
            if csv_id in detector_mapping:
                lane_id, position = detector_mapping[csv_id]
                speed_ms = round(row['vPKW'] / 3.6, 2)
                
                if speed_ms > 0:
                    ET.SubElement(timestep, "vehicle", {
                        "id": str(row['Detector']), # Keep original ID for the car
                        "lane": lane_id,
                        "speed": str(speed_ms),
                        "pos": position,  # Now uses the real pos from your XML (e.g., 1080)
                        "type": "PKW"
                    })

    # Save and Prettify
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(fcd_file, "w") as f:
        f.write(xml_str)

def run_sumo_od_estimation(sim_config,measured_data,tripinfo_output,
            fcd_output,det_interval,sensing_detectors,tracking_log):
    save_sim_to_rds_csv(measured_data, measurement_locations, output_filename="simulated_rds_data.csv")
    _ = od_estimation('simulated_rds_data.csv', plot=True, write_rou_xml=True)

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


if __name__ == "__main__":
    cmd_probs = [float(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    ## SCRIPT CONFIGS ##
    RERUN_GT = False # whether to rerun the ground truth simulation and regenerate synthetic measurements (set to False to save time if already done)
    REAL_DATA = False
    method = "OD_ESTIMATION" # or "FLOWROUTER" "OD_ESTIMATION"

    rds_file = "rds_file/mediumnet_0300-0480.csv" # only used if REAL_DATA = True
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
        measured_output = reader.extract_rds_measurements(rds_file, measurement_locations)
        if method == "FLOWROUTER":
            save_sim_to_rds_fr(measured_output, measurement_locations, output_filename="fr_intermediate.csv", interval_seconds=30)
        save_sim_to_rds_csv(measured_output, measurement_locations, output_filename="simulated_rds_data.csv")
        generate_fcd(rds_file)
        pid_flows = od_estimation('simulated_rds_data.csv', plot=True, write_rou_xml=True)
    else:
        if RERUN_GT:
            run_sumo(SCENARIO+'_gt.sumocfg', tripinfo_output="tripinfo_gt.xml", fcd_output="fcd_output/fcd_sim.xml")
        measured_output = reader.extract_sim_meas(measurement_locations)
        save_sim_to_rds_csv(measured_output, measurement_locations, output_filename="simulated_rds_data.csv")
        if method == "FLOWROUTER":
            save_sim_to_rds_fr(measured_output, measurement_locations, output_filename="fr_intermediate.csv", interval_seconds=30)
        pid_flows = od_estimation('simulated_rds_data.csv', plot=True, write_rou_xml=True)

    # ================================= run PID closed-loop calibration with synthetic measurements
    # 1. run cross correlation analysis
    controlled_flows_route = ["r_0","r_1", "r_2", "r_3", "r_4"]
    #route_correlations(SCENARIO+".sumocfg", controlled_flows_route, measurement_locations).to_csv("correlation_results.csv", index=False)


    synth_data = measured_output
    controlled_flows_route = [("r_0",), ("r_1",), ("r_2",), ("r_3",), ("r_4",)]
    debug_detectors = [
                         '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                        #  '54_1_0', '54_1_1', '54_1_2', '54_1_3'
                        ]
    sensing_detectors = [('56_7_0', '56_7_1', '56_7_2', '56_7_3'),
                         ("56_3_4",),
                         ("56_3_4",),
                         ("56_7_4",),
                         ("55_3_3",)]
    

    num_lanes = [5, 1, 1, 1, 1]
    pid_controllers_rectifier = [PID(0.8, 0.0, 0.0, setpoint=0.0), 
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.2, 0, 0, setpoint=0.0),
                       PID(0.4, 0, 0, setpoint=0.0),
                       PID(0.4, 0.0, 0, setpoint=0.0)]
    
    pid_controllers_feedback = [PID(1.2, 0, 0.0, setpoint=0.0), 
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0, 0, setpoint=0.0),
                       PID(1.2, 0.0, 0, setpoint=0.0)] # need for ablation study

    delays = probe_simulation_transfer_delays(SCENARIO+".sumocfg", controlled_flows_route, sensing_detectors, SUMO_EXE)
    no_delays = [1, 1, 1, 1, 1] # for ablation version with no delay
    #delays = [0] * len(controlled_flows_route)
    ## to do: add a configuration checking function here

    if method == "PID":
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
            ma_windows= [30, 30, 30, 30, 30],             # Moving average window in seconds (adjust as needed)
            num_lanes = num_lanes,
            route_tracking_sensors=measurement_locations,
            routes=['r_0', 'r_1', 'r_2', 'r_3', 'r_4'],
            pid_flows=pid_flows)
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