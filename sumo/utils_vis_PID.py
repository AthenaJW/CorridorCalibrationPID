import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lxml import etree
import xml.etree.ElementTree as ET
import os
import pandas as pd
import plotly.graph_objects as go
import sys
import json
import sumolib
import numpy as np
import seaborn as sns
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import shutil
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../')) # two levels up
sys.path.insert(0, main_path)

from utils_data_read import rds_to_matrix_i24b

# import utils_macro as macro
# import utils_vis as vis

# ================ on-ramp scenario setup ====================
SCENARIO = "onramp"
EXP = "1b"
N_TRIALS = 10000
SUMO_DIR = os.path.dirname(os.path.abspath(__file__)) # current script directory

with open('config.json', 'r') as config_file:
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
                            'downstream_0', 'downstream_1']

num_timesteps = config[SCENARIO]["SIMULATION_TIME"]
step_length = config[SCENARIO]["STEP_LENGTH"]
detector_interval = config[SCENARIO]["DETECTOR_INTERVAL"]
SIM_TIME = num_timesteps * step_length  # total simulation time in seconds
DETECTOR_FILE = config[SCENARIO]["DETECTOR_FILE"]
METHOD_TYPE = config[SCENARIO]["METHOD_TYPE"]
NET_FILE = config[SCENARIO]["NET_FILE"]
IS_RDS = config[SCENARIO]["IS_RDS"]

def parse_multiple_detector_files(lane_to_det, folder_path="onramp"):
    """
    Reads individual XML files for each detector and combines them.
    Expected filename format: det_[detector_id].out.xml
    """
    all_gt_data = []
    
    # Get all unique detector IDs from your mapping
    unique_detectors = {det for det_list in lane_to_det.values() for det in det_list}
    
    for det_id in unique_detectors:
        # Construct the filename based on your previous input
        file_path = os.path.join(folder_path, f"det_{det_id}.out.xml")
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping...")
            continue
            
        tree = etree.parse(file_path)
        for interval in tree.xpath('//interval'):
            all_gt_data.append({
                'begin': float(interval.get('begin')),
                'id': det_id, # Link data back to the detector ID
                'flow': float(interval.get('flow')),
                'speed': float(interval.get('speed'))
            })

    all_sim_data = []
    
    # Get all unique detector IDs from your mapping
    unique_detectors = {det for det_list in lane_to_det.values() for det in det_list}
    
    for det_id in unique_detectors:
        # Construct the filename based on your previous input
        file_path = os.path.join(folder_path, f"det_trial_{det_id}.out.xml")
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping...")
            continue
            
        tree = etree.parse(file_path)
        for interval in tree.xpath('//interval'):
            all_sim_data.append({
                'begin': float(interval.get('begin')),
                'id': det_id, # Link data back to the detector ID
                'flow': float(interval.get('flow')),
                'speed': float(interval.get('speed'))
            })

    all_sim_data = pd.DataFrame(all_sim_data)
    all_gt_data = pd.DataFrame(all_gt_data)
    comparison_df = pd.merge(
        all_sim_data, 
        all_gt_data, 
        on=['begin', 'id'], 
        suffixes=('_sim', '_gt')
    )

    # 2. Calculate the differences as new columns
    comparison_df['flow'] = comparison_df['flow_sim'] - comparison_df['flow_gt']
    comparison_df['speed'] = comparison_df['speed_sim'] - comparison_df['speed_gt']
            
    return {'sim': all_sim_data, 'gt': all_gt_data, 'diff': comparison_df}

def generate_multi_file_heatmap(output_video_dir, lane_shapes, lane_to_det, vid_suffix):
    # 1. Load data from multiple files
    data = parse_multiple_detector_files(lane_to_det, folder_path=SCENARIO)
    
    for key, df in data.items():
        if df.empty:
            print("No data loaded. Check filenames and paths.")
            return

        # 2. Setup Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_facecolor('#1a1a1a') # Darker theme
        ax.set_aspect('equal') # Keep meters-to-scale on Mac
        
        line_objects = {}
        cmap = plt.get_cmap('RdYlGn_r')
        max_flow = df['flow'].max()
        norm = plt.Normalize(0, max_flow+1) # Max flow threshold

        # Draw all roads (including E0)
        for lane_id, coords in lane_shapes.items():
            x_vals = [p[0] for p in coords]
            y_vals = [p[1] for p in coords]
            line, = ax.plot(x_vals, y_vals, lw=2.5, solid_capstyle='round', color='#444444')
            line_objects[lane_id] = line

        # Formatting
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label="Flow (vehs/h)")
        
        # Auto-zoom
        all_x = [p[0] for c in lane_shapes.values() for p in c]
        all_y = [p[1] for c in lane_shapes.values() for p in c]
        ax.set_xlim(min(all_x) - 50, max(all_x) + 50)
        ax.set_ylim(min(all_y) - 50, max(all_y) + 50)

        # 3. Animation Logic
        time_steps = sorted(df['begin'].unique())

        def update(time):
            current_data = df[df['begin'] == time]
            
            for lane_id, line in line_objects.items():
                detectors = lane_to_det.get(lane_id, [])
                if detectors:
                    det_id = detectors[0]
                    # Match the detector ID in the merged dataframe
                    val = current_data[current_data['id'] == det_id]['flow']
                    
                    if not val.empty:
                        line.set_color(cmap(norm(val.values[0])))
                    else:
                        line.set_color('#444444') # No data for this specific time
                else:
                    line.set_color('#444444') # No detector on this lane (E0)
            
            ax.set_title(f"Simulation Time: {time}s", color='white', fontsize=15)
            return list(line_objects.values())

        # 4. Save
        ani = animation.FuncAnimation(fig, update, frames=time_steps, blit=False)
        ani.save(output_video_dir+key+vid_suffix+".mp4", writer='ffmpeg', fps=4)
        print(f"Video saved as {output_video_dir+key+".mp4"}")
        plt.close()

def plot_stacked_3x2(plot_dir, csv_file):
    # Load data
    df = pd.read_csv(csv_file)
    unique_sensors = df['sensor_id'].unique()
    prop_cols = [c for c in df.columns if 'prop' in c]
    
    # Grid settings
    rows, cols = 3, 2
    sensors_per_page = rows * cols
    num_pages = math.ceil(len(unique_sensors) / sensors_per_page)

    for p in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True)
        axes = axes.flatten()  # Flatten to iterate easily
        
        for i in range(sensors_per_page):
            sensor_idx = p * sensors_per_page + i
            ax = axes[i]
            
            if sensor_idx < len(unique_sensors):
                s_id = unique_sensors[sensor_idx]
                subset = df[df['sensor_id'] == s_id].sort_values('time')
                
                # Plot Stacked Area
                time = subset['time']
                y = [subset[col] for col in prop_cols]
                labels = [c.replace('_prop', '').title() for c in prop_cols]
                
                ax.stackplot(time, *y, labels=labels, alpha=0.8)
                ax.set_title(f"Sensor: {s_id}", fontsize=10)
                ax.set_ylim(0, 1)
                
                # Only add legend to the first plot of each page to save space
                if i == 0:
                    ax.legend(loc='upper left', fontsize='small', frameon=True)
            else:
                # Hide empty subplots
                ax.axis('off')

        plt.suptitle(f"Route Proportions - Page {p+1}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_name = plot_dir + f"proportions_page_{p+1}.png"
        plt.savefig(output_name)
        print(f"Saved: {output_name}")
        plt.close()


def plot_detector_xml_flow_comparison(target_xml, observed_xml, detector_id="ramp_0"):
    def parse_sumo_xml(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = []
        for interval in root.findall('interval'):
            # Only grab data for the specific detector ID requested
            if interval.get('id') == detector_id:
                data.append({
                    'time': float(interval.get('begin')),
                    'flow': float(interval.get('flow'))
                })
        return pd.DataFrame(data)

    # Parse both files
    df_target = parse_sumo_xml(target_xml)
    df_observed = parse_sumo_xml(observed_xml)

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_target['time'], df_target['flow'], 
             label='Target Flow', color='#2ca02c', linestyle='--', marker='o', alpha=0.8)
    
    plt.plot(df_observed['time'], df_observed['flow'], 
             label='Observed Flow', color='#1f77b4', linewidth=2, marker='x')

    # Formatting
    plt.title(f'Flow Comparison for Detector: {detector_id}', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Flow (veh/h)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Clean up layout
    plt.tight_layout()
    plt.show()

def get_lane_shapes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Dictionary comprehension to find all lanes
    lane_dict = {
        lane.get('id'): lane.get('shape') 
        for lane in root.findall(".//lane")
    }
    
    return lane_dict

def get_formatted_shapes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    formatted_dict = {}

    for lane in root.findall(".//lane"):
        lane_id = lane.get('id')
        shape_str = lane.get('shape')
        
        # Converts "0.0,4.0 400.0,4.0" -> [(0.0, 4.0), (400.0, 4.0)]
        coords = [
            tuple(map(float, pair.split(','))) 
            for pair in shape_str.split(' ')
        ]
        formatted_dict[lane_id] = coords
        
    return formatted_dict

import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def generate_comparison_tracking_plot(base_dir, header_cols):
    # The three methods you want to compare
    methods = ["pid", "fr", "od"]
    ncols = len(methods)
    
    # We create a 2-row grid: Row 1 = Tracking, Row 2 = Error
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10), 
                             sharex=True, constrained_layout=True)

    # Global font adjustments for clarity
    plt.rcParams.update({'font.size': 12})

    for col_idx, m_name in enumerate(methods):
        # 1. Locate the file in the specific subdirectory
        folder_path = os.path.join(base_dir, m_name)
        log_files = glob.glob(os.path.join(folder_path, "*_log.csv"))
        
        if not log_files:
            print(f"Warning: No log file found in {folder_path}")
            continue
            
        # 2. Load and Process Data (Your logic)
        df = pd.read_csv(log_files[0], index_col=False)
        df.columns = header_cols
        
        # Calculate Metrics
        df['error'] = df['observed'] - df['target']
        mae_val = df['error'].abs().mean()
        bias_val = df['error'].mean()
        metrics_label = f"MAE: {mae_val:.2f} | Bias: {bias_val:.2f}"

        # --- ROW 1: Tracking (Target vs Observed) ---
        ax_top = axes[0, col_idx]
        ax_top.plot(df['step'], df['target'], label='Target', color='#636EFA', marker='o', markersize=2, alpha=0.6)
        ax_top.plot(df['step'], df['observed'], label='Observed', color='#EF553B', marker='o', markersize=2, alpha=0.6)
        
        ax_top.set_title(f"{m_name.upper()} CONTROL\n{metrics_label}", fontsize=16, fontweight='bold', pad=15)
        ax_top.grid(True, linestyle='--', alpha=0.5)
        
        if col_idx == 0:
            ax_top.set_ylabel('Flow (Vehicles / hour)', fontweight='bold')
        ax_top.legend(loc='upper right', fontsize='small')

        # --- ROW 2: Residual Error ---
        ax_bot = axes[1, col_idx]
        ax_bot.plot(df['step'], df['error'], color='black', linewidth=1, alpha=0.4)
        
        # Green/Red fill logic
        ax_bot.fill_between(df['step'], df['error'], 0, where=(df['error'] >= 0), color='green', alpha=0.2)
        ax_bot.fill_between(df['step'], df['error'], 0, where=(df['error'] < 0), color='red', alpha=0.2)
        
        ax_bot.axhline(0, color='black', linewidth=1.2)
        ax_bot.grid(True, linestyle=':', alpha=0.5)
        
        if col_idx == 0:
            ax_bot.set_ylabel('Error (Obs - Tar)', fontweight='bold')
        ax_bot.set_xlabel('Simulation Step', fontweight='bold')

    # Save the combined figure
    output_path = os.path.join(base_dir, 'method_comparison_tracking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Master plot saved to: {output_path}")

def create_interactive_plot(data, header_cols, output_fig_dir):
    df = pd.read_csv(data, index_col=False)

    # Force column names to match exactly what you expect
    df.columns = header_cols

    # Grouping by sensor as per original logic
    sensor_dfs = {name: group for name, group in df.groupby('sensors')}

    for sensor_name, s_df in sensor_dfs.items():
        # Calculate raw error (Observed - Target)
        # Note: This represents the residual R = f(x) - y
        s_df['error'] = s_df['observed'] - s_df['target']
        
        # --- 1. PLOTLY (Interactive Subplots) ---
        # Shared x-axis allows zooming in on both plots simultaneously
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1, 
            subplot_titles=(f'Flow Analysis: {sensor_name}', 'Residual Error (Bias Tracking)'),
            row_heights=[0.7, 0.3]
        )

        # Top Plot: Target vs Observed
        fig.add_trace(go.Scatter(
            x=s_df['step'], y=s_df['target'],
            mode='lines+markers', name='Target',
            line=dict(color='#636EFA'),
            hovertemplate='Step: %{x}<br>Target: %{y}<extra></extra>'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=s_df['step'], y=s_df['observed'],
            mode='lines+markers', name='Observed',
            line=dict(color='#EF553B'),
            hovertemplate='Step: %{x}<br>Observed: %{y}<extra></extra>'
        ), row=1, col=1)

        # Bottom Plot: Error (Not Absolute)
        fig.add_trace(go.Scatter(
            x=s_df['step'], y=s_df['error'],
            mode='lines', name='Error',
            fill='tozeroy', # Shades the area between error and zero
            line=dict(color='gray'),
            hovertemplate='Step: %{x}<br>Error: %{y}<extra></extra>'
        ), row=2, col=1)

        fig.update_layout(
            height=800,
            title_text=f"Traffic Flow Control: Sensor {sensor_name}",
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Flow (Veh/h)", row=1, col=1)
        fig.update_yaxes(title_text="Error (Obs - Tar)", row=2, col=1)
        
        fig.show()

        # --- 2. MATPLOTLIB (Static Subplots for Reports/LaTeX) ---
        fig_mtl, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                          gridspec_kw={'height_ratios': [2, 1]})

        # Main Plot
        ax1.plot(s_df['step'], s_df['target'], label='Target', color='#636EFA', marker='o', markersize=3, alpha=0.8)
        ax1.plot(s_df['step'], s_df['observed'], label='Observed', color='#EF553B', marker='o', markersize=3, alpha=0.8)
        ax1.set_ylabel('Flow (Vehicles / hour)')
        ax1.set_title(f'Target vs Observed Flow: {sensor_name}', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Error Plot (Residuals)
        ax2.plot(s_df['step'], s_df['error'], color='black', linewidth=1, alpha=0.5)
        # Green fill for over-prediction, Red for under-prediction
        ax2.fill_between(s_df['step'], s_df['error'], 0, where=(s_df['error'] >= 0), 
                         color='green', alpha=0.2, label='Over-predict')
        ax2.fill_between(s_df['step'], s_df['error'], 0, where=(s_df['error'] < 0), 
                         color='red', alpha=0.2, label='Under-predict')
        
        ax2.axhline(0, color='black', linewidth=1.5, zorder=1) # Baseline
        ax2.set_ylabel('Error')
        ax2.set_xlabel('Simulation Step')
        ax2.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout()
        
        # Save figure
        file_path = f"{output_fig_dir}analysis_{sensor_name}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved analysis plot for {sensor_name}")


# Example Call:
# headers = ['step', 'sensors', 'target', 'observed']
# generate_comparison_tracking_plot('./results', headers)

def get_lane_count_grid(net_file, mainline_edges, segment_length=100):
    """
    Returns a Series where the index is the space_bin and the value is num_lanes.
    """
    net = sumolib.net.readNet(net_file)
    
    # 1. Calculate the total length to define the grid size
    total_length = sum([net.getEdge(eid).getLength() for eid in mainline_edges])
    num_bins = int(total_length // segment_length) + 1
    
    # 2. Initialize an array to hold lane counts for every meter
    # (Easier to fill meter-by-meter then downsample to bins)
    lane_array = np.zeros(int(total_length) + 1)
    
    current_offset = 0
    for edge_id in mainline_edges:
        edge = net.getEdge(edge_id)
        length = int(edge.getLength())
        num_lanes = len(edge.getLanes())
        
        # Fill the range for this edge with its lane count
        lane_array[current_offset : current_offset + length] = num_lanes
        current_offset += length

    # 3. Downsample to your segment_length bins (using the average or max)
    bin_indices = np.arange(num_bins)
    lane_counts_per_bin = []
    
    for b in bin_indices:
        start_idx = b * segment_length
        end_idx = (b + 1) * segment_length
        # We take the mean in case a segment spans across two edges with different lane counts
        avg_lanes = np.mean(lane_array[start_idx : end_idx]) if end_idx <= len(lane_array) else lane_array[-1]
        lane_counts_per_bin.append(round(avg_lanes))

    return pd.Series(lane_counts_per_bin, name="num_lanes")

def parse_fcd_to_timespace(fcd_xml, net_file, mainline_edges, sim_time, segment_length=100, time_step=10, impute=False):
   
    # 1. Map the Mainline Backbone in order
    net = sumolib.net.readNet(net_file)
    edge_offsets = {}
    current_offset = 0.0
    mainline_set = set(mainline_edges) # For faster lookup
    
    # 1. Map the Mainline Backbone in order
    for eid in mainline_edges:
        try:
            edge = net.getEdge(eid)
            edge_offsets[eid] = current_offset
            current_offset += edge.getLength()
        except:
            continue 
    
    # 2. BETTER Internal Edge Mapping (The Recursive Stripe Killer)
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(':'):
            # We look ahead until we find a "Real" edge that is in our mainline
            search_queue = [edge]
            found = False
            visited = set()
            
            while search_queue and not found:
                curr = search_queue.pop(0)
                if curr.getID() in visited: continue
                visited.add(curr.getID())
                
                for out in curr.getOutgoing():
                    out_id = out.getID()
                    if out_id in edge_offsets: # Found a mainline edge!
                        edge_offsets[eid] = edge_offsets[out_id]
                        found = True
                        break
                    elif out_id.startswith(':'): # Keep digging through the junction
                        search_queue.append(out)
    
    
    total_mainline_length = current_offset
    print(edge_offsets)
    
    # 2. Parse FCD File (Iterative for performance)
    data = []
    context = ET.iterparse(fcd_xml, events=('start', 'end'))
    current_time = 0
    start_time = None
    MAX_REALISTIC_SPEED = 50.0 # m/s, ~90 mph, to filter out erroneous data

    
    for event, elem in context:
        if event == 'start' and elem.tag == 'timestep':
            if start_time is None:
                start_time = float(elem.get('time'))
            else:
                start_time = min(start_time, float(elem.get('time')))
            current_time = float(elem.get('time'))
        
        if event == 'end' and elem.tag == 'vehicle':
            lane_id = elem.get('lane')
            # Handle internal edges if necessary, but here we focus on mainline
            edge_id = lane_id.rsplit('_', 1)[0]

            if edge_id not in edge_offsets:
                #print(f"Dropped edge: {edge_id}") # Uncomment to see the 'black holes'
                continue

            if float(elem.get('speed')) > MAX_REALISTIC_SPEED:
                print(f"Dropped unrealistic speed: {elem.get('speed')} m/s for vehicle {elem.get('id')} at time {current_time}s")
                continue

            if edge_id in edge_offsets:
                abs_pos = edge_offsets[edge_id] + float(elem.get('pos'))
                data.append({
                    'time': current_time,
                    'veh_id': elem.get('id'),
                    'speed': float(elem.get('speed')),
                    'abs_pos': abs_pos
                })
            elem.clear()

    df = pd.DataFrame(data)

    # 3. Create Bins
    df['space_bin'] = (df['abs_pos'] // segment_length).astype(int)
    df['time_bin'] = (df['time'] // time_step * time_step).astype(int)

    # 4. Pivot instead of GroupBy (more robust for matrix formation)
    speed_matrix = df.pivot_table(index='space_bin', columns='time_bin', values='speed', aggfunc='mean')
    volume_matrix = df.pivot_table(index='space_bin', columns='time_bin', values='veh_id', aggfunc='nunique')
    
    # Normalize Volume to Hourly Flow
    volume_matrix = volume_matrix * (3600 / time_step)

    # 5. REINDEXING
    all_space_bins = np.arange(0, int(total_mainline_length // segment_length) + 1)
    all_time_bins = np.arange(start_time, start_time+int(sim_time // time_step * time_step) + time_step, time_step)

    speed_matrix = speed_matrix.reindex(index=all_space_bins, columns=all_time_bins)
    volume_matrix = volume_matrix.reindex(index=all_space_bins, columns=all_time_bins).fillna(0)

    # 6. ANTI-STRIPE LOGIC (Spatial Interpolation)
    # Even without 'impute=True', you need to fill the junction gaps 
    # so the plot doesn't have white lines.
    
    # Fill gaps between edges (horizontal stripes)
    speed_matrix = speed_matrix.interpolate(method='linear', axis=0, limit_direction='both')
    
    # Fill temporal flicker (random missing pixels)
    speed_matrix = speed_matrix.interpolate(method='linear', axis=1, limit_direction='both')

    # If you want to force 0 where there's still no data after interpolation:
    if impute:
        speed_matrix = speed_matrix.fillna(0)

    return speed_matrix, volume_matrix

def parse_detector_as_fcd(fcd_xml, net_file, mainline_edges, sim_time, segment_length=100, time_step=10):
    # 1. Use your EXISTING FCD parser to get the sparse matrices
    # This ensures identical shape and logic to your simulation results
    speed_matrix, volume_matrix = parse_fcd_to_timespace(
        fcd_xml, net_file, mainline_edges, sim_time, segment_length, time_step
    )

    # 2. Apply Spatial Imputation
    # Because the 'fake' FCD only has vehicles at detector locations,
    # we fill the space between them.
    #speed_imputed = speed_matrix.interpolate(method='linear', axis=0, limit_direction='both')
    
    # For flow/volume, we interpolate but usually fill edge cases with 0 
    # if the road starts/ends far from any detector.
    #volume_imputed = volume_matrix.interpolate(method='linear', axis=0, limit_direction='both')

    return speed_matrix, volume_matrix


def generate_detector_mapping(detector_file):
    """
    Parses a SUMO detector file and creates a mapping dictionary.
    Format: { "RDS_Lane": (lane_id, pos) }
    """
    tree = ET.parse(detector_file)
    root = tree.getroot()
    
    detector_mapping = {}

    # SUMO detectors are usually <inductionLoop> (E1) or <laneAreaDetector> (E2/E3)
    # We search for both tags
    for det_type in ['inductionLoop', 'laneAreaDetector']:
        for det in root.findall(f'.//{det_type}'):
            det_id = det.get('id')
            lane_id = det.get('lane')
            pos = det.get('pos')
            
            # Logic to construct your RDS-style key (e.g., "56_7_0")
            # This depends on how your lanes are named. 
            # If your lane_id is "E1_0", we can transform it.
            # Here we split the lane_id to try and match your naming convention:
            parts = lane_id.split('_')
            if len(parts) >= 2:
                # Example: "E1_0" becomes a key based on your specific RDS naming
                # Since RDS isn't in the XML, you might need a simple lookup 
                # or use the lane_id directly as the key.
                detector_mapping[lane_id] = (lane_id, pos)
            else:
                detector_mapping[det_id] = (lane_id, pos)

    return detector_mapping

def parse_detector_to_timespace(df_name, sim_time, total_mainline_length, segment_length=100, time_step=30):
    detector_mapping = generate_detector_mapping(DETECTOR_FILE) # Adjust path as needed
    
    df = pd.read_csv(df_name, delimiter=';')
    df.columns = df.columns.str.strip()
    
    # 1. Normalize IDs and Map to Absolute Position
    df['clean_id'] = df['Detector'].str.replace(".", "_")
    df['abs_pos'] = df['clean_id'].map(lambda x: float(detector_mapping[x][1]) if x in detector_mapping else np.nan)
    df = df.dropna(subset=['abs_pos'])

    # 2. Time Conversion (if timestamp is '00:05:00', convert to seconds)
    if df['Time'].dtype == object:
        df['Time'] = pd.to_timedelta(df['Time']).dt.total_seconds()

    # 3. Binning
    df['space_bin'] = (df['abs_pos'] // segment_length).astype(int)
    df['time_bin'] = (df['Time'] // time_step * time_step).astype(int)

    # 4. Create Pivots
    # For Speed: average of the speed column
    speed_pivot = df.pivot_table(index='space_bin', columns='time_bin', values='vPKW', aggfunc='mean')
    
    # For Flow: sum of qPKW (assuming qPKW is vehicles per time step)
    # Then multiply to get Vehicles per Hour: (count / time_step) * 3600
    flow_pivot = df.pivot_table(index='space_bin', columns='time_bin', values='qPKW', aggfunc='sum')
    flow_pivot = flow_pivot * (3600 / time_step)

    # 5. Create the Full Grid
    max_pos = max([float(v[1]) for v in detector_mapping.values()])
    all_space_bins = np.arange(0, int(total_mainline_length // segment_length) + 1)
    all_time_bins = np.arange(0, int(sim_time // time_step * time_step) + time_step, time_step)

    # 5. Reindex
    # speed_pivot only has rows for the bins where detectors exist.
    # Reindexing to 'all_space_bins' creates the empty rows for the rest of the road.
    speed_matrix = speed_pivot.reindex(index=all_space_bins, columns=all_time_bins)
    flow_matrix = flow_pivot.reindex(index=all_space_bins, columns=all_time_bins)

    # 6. Impute
    # IMPORTANT: Linear interpolation only fills BETWEEN data points. 
    # If your first detector is at bin 5 and the road starts at bin 0, 
    # limit_direction='both' will fill bin 0-4 with the value of bin 5.
    speed_matrix_imputed = speed_matrix.interpolate(method='linear', axis=0, limit_direction='both')
    flow_matrix_imputed = flow_matrix.interpolate(method='linear', axis=0, limit_direction='both')

    return speed_matrix_imputed, flow_matrix_imputed


def scipy_impute(df):
    # 1. Get coordinates of the grid
    # x = segments, y = time
    x_coords, y_coords = np.meshgrid(np.arange(df.shape[1]), np.arange(df.shape[0]))
    
    # 2. Identify where we have data and where we have NaNs
    mask = ~np.isnan(df.values)
    points = np.array([x_coords[mask], y_coords[mask]]).T
    values = df.values[mask]
    
    # 3. Interpolate
    # 'cubic' is smooth but can overshoot; 'linear' is safer for traffic data
    imputed_values = griddata(points, values, (x_coords, y_coords), method='linear')
    
    # 4. Fill edges that griddata might leave as NaN (extrapolation)
    # Griddata won't extrapolate outside the 'convex hull' of your data
    result_df = pd.DataFrame(imputed_values, index=df.index, columns=df.columns)
    return result_df.bfill(axis=1).ffill(axis=1).bfill(axis=0).ffill(axis=0)

def calculate_mape(sim_df, pid_df, eps=0.01):
    # 1. Align - ensure sim is the base
    sim_df, pid_df = sim_df.align(pid_df, join='inner')
    
    # 2. Mask
    mask = sim_df.notna() & pid_df.notna()
    s = sim_df.values[mask]
    p = pid_df.values[mask]
    
    # 3. Calculation - Only use 's' in the denominator!
    # If your current code has (s + p + eps), change it to this:
    abs_pct_error = np.abs(s - p) / (s + eps)
    
    return np.mean(abs_pct_error) * 100

def calculate_smape(sim_df, pid_df):
    # Align and mask
    sim_df, pid_df = pid_df.align(sim_df, join='inner')
    mask = sim_df.notna() & pid_df.notna()
    
    s = sim_df.values[mask]
    p = pid_df.values[mask]
    
    # Formula: 100 * |sim - pid| / ((|sim| + |pid|) / 2)
    # Adding a tiny epsilon (1e-5) prevents 0/0 if both are exactly zero
    numerator = np.abs(s - p)
    denominator = (np.abs(s) + np.abs(p)) / 2 + 1e-5
    
    smape_val = np.mean(numerator / denominator) * 100
    return smape_val

def calculate_mae(sim_df, pid_df):
    """Mean Absolute Error: Great for understanding error in physical units (vehicles)."""
    sim_df, pid_df = sim_df.align(pid_df, join='inner')
    mask = sim_df.notna() & pid_df.notna()
    return np.mean(np.abs(sim_df.values[mask] - pid_df.values[mask]))

def calculate_rmse(sim_df, pid_df):
    """Root Mean Squared Error: Penalizes large outliers/shocks more heavily."""
    sim_df, pid_df = sim_df.align(pid_df, join='inner')
    mask = sim_df.notna() & pid_df.notna()
    mse = np.mean((sim_df.values[mask] - pid_df.values[mask])**2)
    return np.sqrt(mse)

def calculate_nmape_shifted(sim_df, pid_df, shift=1.0):
    """
    Normalized MAPE (Shifted): Adds a constant to the denominator.
    Commonly used when data contains zeros to avoid division by zero
    and stabilize the metric.
    """
    sim_df, pid_df = sim_df.align(pid_df, join='inner')
    mask = sim_df.notna() & pid_df.notna()
    s = sim_df.values[mask]
    p = pid_df.values[mask]
    
    # Shifting values by 1 (or another constant) as described by your coworker
    abs_error = np.abs(s - p)
    normalized_pct_error = abs_error / (s + shift)
    
    return np.mean(normalized_pct_error) * 100

def main(plot_dir, data_dir):
    rerun_sim = False
    if METHOD_TYPE == "PID":
        method_log = "pid_log_sim_3hr.csv" #  if sim_data else "pid_log_rds.csv"
    elif METHOD_TYPE == "FR":
        method_log = "fr_log.csv"
    elif METHOD_TYPE == "OD":
        method_log = "od_log.csv"
    
    if METHOD_TYPE == "PID":
        method_fcd_file_name = "fcd_output/fcd_pid_sim_3hr.xml"
    elif METHOD_TYPE == "FR":
        method_fcd_file_name = "fcd_output/fcd_fr_sim_3hr.xml"
    elif METHOD_TYPE == "OD":
        method_fcd_file_name = "fcd_output/fcd_od_sim_3hr.xml"
    gt_fcd = 'fcd_output/fcd_sim.xml' if not IS_RDS else 'fcd_output/rds_fcd.xml'

    files_to_move = [method_fcd_file_name, method_log, gt_fcd]
    
    if not os.path.exists(plot_dir) or rerun_sim:
        print(f"Creating {plot_dir} and copying files from data to plot.")
        os.makedirs(plot_dir, exist_ok=True)
        src_base = data_dir
        dst_base = plot_dir
    else:
        print(f"{plot_dir} exists. Copying files from plot to data.")
        src_base = plot_dir
        dst_base = data_dir

    for file_path in files_to_move:
        # Construct full source and destination paths based on determined direction
        source = os.path.join(src_base, file_path)
        destination = os.path.join(dst_base, file_path)
        
        # Ensure the destination subdirectory exists (e.g., 'fcd_output/')
        dest_subdir = os.path.dirname(destination)
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)
            
        # Move the file
        if os.path.exists(source):
            # Using move instead of copy2 since you mentioned "move"
            shutil.copy2(source, destination)
            print(f"Copied: {file_path} to {dst_base}")
        else:
            print(f"Warning: {source} not found.")

    show_flow = False if IS_RDS else True

    try:
        with open(os.path.join(data_dir, 'lane_shapes.json'), 'r') as f:
            data = json.load(f)
            
        # Extract the dictionaries from the loaded JSON object
        lane_shapes = data["lane_shapes"]
        lane_to_det = data["lane_to_det"]
    except FileNotFoundError:
        print("Error: lane_shapes.json not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Check your file formatting.")

    header_pid = ['step', 'time', 'sensors', 'target', 'observed', 'smoothed_error', 'raw_control_signal', 'delayed_signal', 'new_total_injection']
    header_debug = ['step', 'time', 'sensors', 'target', 'observed']
    header_fr = ['step', 'time', 'sensors', 'target', 'observed', 'speed']
    header_od = ['step', 'time', 'sensors', 'target', 'observed', 'speed']
    if METHOD_TYPE == "PID":
        header_method = header_pid
    elif METHOD_TYPE == "FR":
        header_method = header_fr
    elif METHOD_TYPE == "OD":
        header_method = header_od


    sim_time = 1800
    create_interactive_plot(os.path.join(data_dir, method_log), header_method, plot_dir+"sensor_")
    
    '''
    generate_multi_file_heatmap(plot_dir, lane_shapes, lane_to_det, "150m_transfer")
    '''
    NET_FILE = os.path.join(data_dir, "onramp.net.xml")
    # Define your mainline order here
    MAINLINE = data["mainlane"]

    ## parameters for plotting time space plot
    segment_length = 100
    if not IS_RDS:
        time_bucket = 10 # seconds
        speed_df_sim, flow_df_sim = parse_fcd_to_timespace(os.path.join(data_dir, 'fcd_output/fcd_sim.xml'), NET_FILE, MAINLINE, sim_time, 
                                            segment_length=segment_length, time_step=time_bucket)
    else:
        time_bucket = 30 # seconds
        speed_df_sim, flow_df_sim = parse_detector_as_fcd(os.path.join(data_dir, 'fcd_output/rds_fcd.xml'), NET_FILE, MAINLINE, sim_time, 
                                            segment_length=segment_length, time_step=time_bucket)
        
    speed_df_pid, flow_df_pid = parse_fcd_to_timespace(os.path.join(data_dir, method_fcd_file_name), NET_FILE, MAINLINE, sim_time, 
                                           segment_length=segment_length, time_step=time_bucket,impute=False)
    
    data_to_save = {
        "speed_sim": speed_df_sim,
        "flow_sim": flow_df_sim,
        "speed_pid": speed_df_pid,
        "flow_pid": flow_df_pid
    }

    for name, df in data_to_save.items():
        # Convert to numpy and save
        file_path = os.path.join(plot_dir, f"{name}.npy")
        np.save(file_path, df.to_numpy())
        print(f"Saved {name} to {file_path}")

    
    # Impute along the 'segment' axis (axis=1) or 'time' axis (axis=0)
    #speed_df_pid = scipy_impute(speed_df_pid)
    #flow_df_pid = scipy_impute(flow_df_pid)

    #speed_df_sim = scipy_impute(speed_df_sim)
    #flow_df_sim = scipy_impute(flow_df_sim)
    
    # 1. Define the mask (using .values to ensure we are working with coordinates)
    
    # Calculate Metrics

    speed_mape = calculate_mape(speed_df_sim, speed_df_pid)
    flow_mape = calculate_mape(flow_df_sim, flow_df_pid) if show_flow else None

    speed_smape = calculate_smape(speed_df_sim, speed_df_pid)
    flow_smape = calculate_smape(flow_df_sim, flow_df_pid) if show_flow else None

    speed_nmape_shifted = calculate_nmape_shifted(speed_df_sim, speed_df_pid, shift=1.0)
    flow_nmape_shifted = calculate_nmape_shifted(flow_df_sim, flow_df_pid, shift=1.0) if show_flow else None

    speed_mae = calculate_mae(speed_df_sim, speed_df_pid)
    flow_mae = calculate_mae(flow_df_sim, flow_df_pid) if show_flow else None

    speed_rmse = calculate_rmse(speed_df_sim, speed_df_pid)
    flow_rmse = calculate_rmse(flow_df_sim, flow_df_pid) if show_flow else None


    # --- 2. Setup Plotting Grid ---
    # Choose 2 columns if show_flow is True, else 1 column
    ncols = 2 if show_flow else 1
    fig, axes = plt.subplots(2, ncols, figsize=(8*ncols, 10), sharex=True, sharey=True, layout='constrained')
    
    # Handle axes indexing if it's only 1 column (matplotlib flattens the array)
    if not show_flow:
        axes = np.expand_dims(axes, axis=1)

    # --- 3. Speed Plotting Parameters ---
    STABLE_MIN = 0.0
    STABLE_MAX = 30.0

    norm = mcolors.Normalize(vmin=STABLE_MIN, vmax=STABLE_MAX)

    # 3. Calculate your dynamic limit for the bar's length
    current_data_max = speed_df_sim.max().max() 

    speed_params = {
        'cmap': 'RdYlGn',
        'norm': norm,           # This fixes the color mapping
        'vmin': None,           # norm overrides vmin/vmax, so set to None
        'vmax': None,
        'cbar_kws': {
            'label': 'Speed (m/s)',
            'ticks': [0, 10, 20, 30, 40, 50] # You can fix the labels too
        }
    }

    # --- 4. Render Speed (Column 0) ---
    sns.heatmap(speed_df_sim.iloc[::-1], ax=axes[0, 0], **speed_params)
    #axes[0, 0].set_title(f'Simulation: Speed\n(Speed MAPE: {speed_mape:.2f}%, SMAPE: {speed_smape:.2f}%, NMAPE: {speed_nmape_shifted:.2f}%\n, MAE: {speed_mae:.2f}, RMSE: {speed_rmse:.2f})')
    axes[0, 0].set_title(f'Simulation: Speed')
 
    sns.heatmap(speed_df_pid.iloc[::-1], ax=axes[1, 0], **speed_params)
    axes[1, 0].set_title(f'{METHOD_TYPE} Control: Speed')

    # --- 5. Render Flow (Colimn 1) - Optional ---
    if show_flow:
        flow_min = min(flow_df_sim.min().min(), flow_df_pid.min().min())
        flow_max = max(flow_df_sim.max().max(), flow_df_pid.max().max())
        
        flow_params = {
            'cmap': 'viridis', 
            'vmin': flow_min, 
            'vmax': flow_max, 
            'cbar_kws': {'label': 'Flow (veh/h)'}
        }

        sns.heatmap(flow_df_sim.iloc[::-1], ax=axes[0, 1], **flow_params)
        #axes[0, 1].set_title(f'Similation: Flow\n(Flow MAPE: {flow_mape:.2f}%, SMAPE: {flow_smape:.2f}%, NMAPE: {flow_nmape_shifted:.2f}%\n, MAE: {flow_mae:.2f}, RMSE: {flow_rmse:.2f})')
        axes[0, 1].set_title(f'Simulation: Flow')

        sns.heatmap(flow_df_pid.iloc[::-1], ax=axes[1, 1], **flow_params)
        axes[1, 1].set_title('PID Control: Flow')

    # --- 6. Formatting ---
    for ax in axes[1, :]:
        ax.set_xlabel('Time Step')
    for ax in axes[:, 0]:
        ax.set_ylabel('Road Segment')

    plt.savefig(plot_dir + 'time_space_plots.png', dpi=300)
    plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter, MultipleLocator
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
def generate_master_stacked_plot(base_dir, is_rds=False):
    # Set global font sizes
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16})
    
    methods = ["pid", "fr"]
    show_flow = not is_rds
    ncols = 2 if show_flow else 1
    nrows = len(methods) + 1 
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 3.5 * nrows), 
                             sharex=True, constrained_layout=True)

    speed_norm = mcolors.Normalize(vmin=0, vmax=50)
    speed_cmap = 'RdYlGn'
    flow_cmap = 'viridis'
    
    # 1. Load Dimensions
    gt_path = os.path.join(base_dir, methods[0])
    speed_gt = np.load(os.path.join(gt_path, "speed_sim.npy"))
    num_segments, num_steps = speed_gt.shape
    
    # --- RDS TIME LOGIC ---
    # If RDS, 1 index step = 30 seconds.
    time_step_interval = 30 if is_rds else 10
    
    # We want to show a tick every 300 REAL seconds.
    # We find the array indices that correspond to [0, 300, 600, ...]
    # Formula: index = real_time / interval
    tick_spacing_seconds = 300
    
    # Array indices where we place the ticks
    x_tick_indices = np.arange(0, num_steps, tick_spacing_seconds / time_step_interval)
    # The actual labels to display (0, 300, 600...)
    x_tick_labels = [int(i * time_step_interval) for i in x_tick_indices]
    
    # Y-axis stays consistent with segments
    y_tick_indices = np.arange(0, num_segments, 5)

    if show_flow:
        flow_gt = np.load(os.path.join(gt_path, "flow_sim.npy"))
        flow_max = flow_gt.max()

    # 2. Plotting Loop
    for i in range(nrows):
        if i == 0:
            row_title, s_data = "GT", speed_gt
            f_data = flow_gt if show_flow else None
        else:
            m_name = methods[i-1]
            row_title, m_path = m_name.upper(), os.path.join(base_dir, m_name)
            s_data = np.load(os.path.join(m_path, f"speed_pid.npy"))
            f_data = np.load(os.path.join(m_path, f"flow_pid.npy")) if show_flow else None

        for col in range(ncols):
            ax = axes[i, col] if ncols > 1 else axes[i]
            data = s_data if col == 0 else f_data
            cmap = speed_cmap if col == 0 else flow_cmap
            norm = speed_norm if col == 0 else mcolors.Normalize(vmin=0, vmax=flow_max)
            
            sns.heatmap(data, ax=ax, cmap=cmap, norm=norm, cbar=False)
            
            if i == 0:
                ax.set_title("SPEED" if col == 0 else "FLOW", fontweight='bold', fontsize=20, pad=10)
            
            ax.invert_yaxis()

            # --- Y-Axis Formatting ---
            ax.set_ylabel(f"{row_title}\nSegments", fontweight='bold', multialignment='center')
            ax.set_yticks(y_tick_indices + 0.5)
            ax.set_yticklabels([str(y) for y in y_tick_indices], rotation=0, va='center', fontsize=12)

            # --- X-Axis Formatting (UPDATED FOR RDS) ---
            ax.set_xticks(x_tick_indices + 0.5)
            ax.set_xticklabels(x_tick_labels)
            ax.tick_params(left=True, bottom=True)

    # 3. Shared Colorbars (The part you asked for)
    speed_ax_target = axes[:, 0] if ncols > 1 else axes
    
    sm_s = plt.cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap)
    cbar_s = fig.colorbar(sm_s, ax=speed_ax_target, orientation='horizontal', 
                          location='bottom', fraction=0.04, pad=0.05)
    cbar_s.set_label('Speed (m/s)', fontweight='bold', labelpad=5)
    cbar_s.ax.set_title('Time (seconds)', fontweight='bold', fontsize=16, pad=30)

    if show_flow:
        sm_f = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=flow_max), cmap=flow_cmap)
        cbar_f = fig.colorbar(sm_f, ax=axes[:, 1], orientation='horizontal', 
                              location='bottom', fraction=0.04, pad=0.05)
        cbar_f.set_label('Flow (veh/h)', fontweight='bold', labelpad=5)
        cbar_f.ax.set_title('Time (seconds)', fontweight='bold', fontsize=16, pad=30)

    plt.savefig(os.path.join(base_dir, 'stacked_final_fixed_rotation.png'), bbox_inches='tight')
    plt.show()

def generate_comparison_tracking_plot(base_dir):
    # Mapping folders to their specific header structures
    header_map = {
        'pid': ['step', 'time', 'sensors', 'target', 'observed', 'smoothed_error', 'raw_control_signal', 'delayed_signal', 'new_total_injection'],
        'fr': ['step', 'time', 'sensors', 'target', 'observed', 'speed'],
        'od': ['step', 'time', 'sensors', 'target', 'observed', 'speed'],
        'debug': ['step', 'time', 'sensors', 'target', 'observed']
    }
    
    methods = ["pid", "fr", "od"]
    ncols = len(methods)
    
    # Grid: 2 rows (Tracking, Error) x 3 columns (PID, FR, OD)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10), 
                             sharex=True, constrained_layout=True)

    plt.rcParams.update({'font.size': 12})

    for col_idx, m_name in enumerate(methods):
        folder_path = os.path.join(base_dir, m_name)
        log_files = glob.glob(os.path.join(folder_path, "*_log.csv"))
        
        if not log_files:
            print(f"Skipping {m_name}: No log file found.")
            continue
            
        # Select the correct header for this specific folder
        current_headers = header_map.get(m_name)
        
        # Load data using the specific headers
        df = pd.read_csv(log_files[0], index_col=False)
        df.columns = current_headers
        
        # Calculate standardized metrics (Observed and Target exist in all)
        df['error'] = df['observed'] - df['target']
        mae_val = df['error'].abs().mean()
        bias_val = df['error'].mean()
        metrics_label = f"MAE: {mae_val:.2f} | Bias: {bias_val:.2f}"

        # --- ROW 1: Tracking ---
        ax_top = axes[0, col_idx]
        ax_top.plot(df['step'], df['target'], label='Target', color='#636EFA', alpha=0.7)
        ax_top.plot(df['step'], df['observed'], label='Observed', color='#EF553B', alpha=0.7)
        
        ax_top.set_title(f"{m_name.upper()}\n{metrics_label}", fontsize=16, fontweight='bold')
        ax_top.grid(True, linestyle='--', alpha=0.5)
        
        if col_idx == 0:
            ax_top.set_ylabel('Flow (Veh/h)', fontweight='bold')
        ax_top.legend(loc='upper right')

        # --- ROW 2: Error ---
        ax_bot = axes[1, col_idx]
        ax_bot.plot(df['step'], df['error'], color='black', linewidth=1, alpha=0.4)
        ax_bot.fill_between(df['step'], df['error'], 0, where=(df['error'] >= 0), color='green', alpha=0.2)
        ax_bot.fill_between(df['step'], df['error'], 0, where=(df['error'] < 0), color='red', alpha=0.2)
        
        ax_bot.axhline(0, color='black', linewidth=1.2)
        ax_bot.grid(True, linestyle=':', alpha=0.5)
        
        if col_idx == 0:
            ax_bot.set_ylabel('Error (Obs - Tar)', fontweight='bold')
        ax_bot.set_xlabel('Simulation Step', fontweight='bold')

    output_path = os.path.join(base_dir, 'comparison_fixed_headers.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Master plot saved to: {output_path}")

# Call with the root directory containing the pid, fr, and od folders
# generate_comparison_tracking_plot('./scenario_1_results')

# Usage
# generate_horizontal_summary_plots('your_base_dir_here')

if __name__ == "__main__":

    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="Run on-ramp simulation and save plots.")

    # 2. Add the argument as a keyword (using the -- flag)
    parser.add_argument(
        "--plot_dir", 
        type=str, 
        default="onramp/figures/temp/",
        help="Directory where figures will be saved"
    )

    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="onramp/data/",
        help="Directory where data files are located"
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Access the value using args.plot_dir
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
        print(f"Created new directory: {args.plot_dir}")


    main(plot_dir=args.plot_dir, data_dir=args.data_dir)
    #generate_master_stacked_plot(base_dir = args.data_dir, is_rds=True)
