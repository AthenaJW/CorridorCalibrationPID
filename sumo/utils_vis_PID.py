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

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)

# import utils_macro as macro
# import utils_vis as vis

# ================ on-ramp scenario setup ====================
SCENARIO = "i24"
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

def parse_fcd_to_timespace(fcd_xml, net_file, mainline_edges, sim_time, segment_length=100, time_step=10):
    # 1. Automatically get offsets and total mainline length
    net = sumolib.net.readNet(net_file)
    edge_offsets = {}
    current_offset = 0.0
    for edge_id in mainline_edges:
        edge = net.getEdge(edge_id)
        edge_offsets[edge_id] = current_offset
        current_offset += edge.getLength()
    
    total_mainline_length = current_offset
    
    # 2. Parse FCD File
    data = []
    max_sim_time = 0
    context = ET.iterparse(fcd_xml, events=('start', 'end'))
    
    for event, elem in context:
        if event == 'start' and elem.tag == 'timestep':
            current_time = float(elem.get('time'))
            if current_time > max_sim_time:
                max_sim_time = current_time
        
        if event == 'end' and elem.tag == 'vehicle':
            lane_id = elem.get('lane')
            edge_id = lane_id.rsplit('_', 1)[0]
            
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

    # 4. Aggregate
    speed_matrix = df.groupby(['space_bin', 'time_bin'])['speed'].mean().unstack()
    volume_matrix = df.groupby(['space_bin', 'time_bin'])['veh_id'].nunique().unstack() * (3600 / time_step)

    # 5. REINDEXING: Fill the gaps
    # Define full range of bins
    all_space_bins = np.arange(0, int(total_mainline_length // segment_length) + 1)
    all_time_bins = np.arange(0, int(sim_time // time_step * time_step) + time_step, time_step)

    # Apply reindexing to both axes
    def fill_matrix(mtx):
        return mtx.reindex(index=all_space_bins, columns=all_time_bins)

    speed_matrix = fill_matrix(speed_matrix)
    volume_matrix = fill_matrix(volume_matrix)

    # Optional: Fill volume NaNs with 0 (since no vehicles = 0 flow)
    # Note: We usually keep speed as NaN where no vehicles exist to avoid showing 0 km/h 
    volume_matrix = volume_matrix.fillna(0)

    return speed_matrix, volume_matrix


def main(plot_dir, data_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

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
    
    create_interactive_plot(os.path.join(data_dir, "pid_log.csv"), header_pid, plot_dir+"sensor_")
    if os.path.exists(os.path.join(data_dir, "debug_log.csv")):
        create_interactive_plot(os.path.join(data_dir, "debug_log.csv"), header_debug, plot_dir+"debug_")
    else:
        print(f"File not found: {os.path.join(data_dir, 'debug_log.csv')}, skipping analysis")
   
    plot_stacked_3x2(plot_dir, os.path.join(data_dir, 'route_proportions.csv'))
    '''
    generate_multi_file_heatmap(plot_dir, lane_shapes, lane_to_det, "150m_transfer")
    '''
    # Time space plot absolute position calculation
    NET_FILE = os.path.join(data_dir, 'i24.net.xml')
    DET_FILE = os.path.join(data_dir, 'i24_RDS_gt.add.xml')
    # Define your mainline order here
    MAINLINE = data["mainlane"]

    ## parameters for plotting time space plot
    segment_length = 100
    time_bucket = 10 # seconds

    speed_df_sim, flow_df_sim = parse_fcd_to_timespace(os.path.join(data_dir, 'fcd_output/fcd_sim.xml'), NET_FILE, MAINLINE, num_timesteps, 
                                           segment_length=segment_length, time_step=time_bucket)
    speed_df_pid, flow_df_pid = parse_fcd_to_timespace(os.path.join(data_dir, 'fcd_output/fcd_pid.xml'), NET_FILE, MAINLINE, num_timesteps, 
                                           segment_length=segment_length, time_step=time_bucket)
    
    # 1. Define the mask (using .values to ensure we are working with coordinates)
    valid_mask = (flow_df_pid != 0).values & flow_df_pid.notna().values & flow_df_sim.notna().values

    # 2. Initialize error array with NaNs
    error = np.full(flow_df_sim.shape, np.nan)

    # 3. Calculate the percentage error for valid indices only
    # We flatten the result to 1D to match the boolean indexing requirements
    numerator = (flow_df_sim.values[valid_mask] - flow_df_pid.values[valid_mask])
    denominator = flow_df_pid.values[valid_mask]

    error[valid_mask] = np.abs(numerator / denominator)

    # 4. Calculate MAPE
    mape = np.nanmean(error) * 100
    print(f"Mean Absolute Percentage Error (MAPE) between Simulation and PID Flow: {mape:.2f}%")
    
    speed_min = min(speed_df_sim.min().min(), speed_df_pid.min().min())
    speed_max = max(speed_df_sim.max().max(), speed_df_pid.max().max())

    # 2. Calculate global min/max for Flow
    flow_min = min(flow_df_sim.min().min(), flow_df_pid.min().min())
    flow_max = max(flow_df_sim.max().max(), flow_df_pid.max().max())

    lane_count_series = get_lane_count_grid(NET_FILE, MAINLINE, segment_length=segment_length)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True, layout='constrained')
    
    # Define plotting parameters to keep it clean
    plot_params = {
        'cmap': 'RdYlGn', 
        'vmin': speed_min, 
        'vmax': speed_max, 
        'cbar_kws': {'label': 'Speed (m/s)'}
    }

    flow_params = {
        'cmap': 'viridis', 
        'vmin': flow_min, 
        'vmax': flow_max, 
        'cbar_kws': {'label': 'Flow (veh/h)'}
    }

    # --- TOP ROW: Simulation ---
    mape_str = f" (MAPE: {mape:.2f}%)"
    sns.heatmap(speed_df_sim.iloc[::-1], ax=axes[0, 0], **plot_params)
    axes[0, 0].set_title('Simulation: Speed')
    
    sns.heatmap(flow_df_sim.iloc[::-1], ax=axes[0, 1], **flow_params)
    axes[0, 1].set_title('Simulation: Flow')

    # --- BOTTOM ROW: PID ---
    sns.heatmap(speed_df_pid.iloc[::-1], ax=axes[1, 0], **plot_params)
    axes[1, 0].set_title('PID Control: Speed')
    
    sns.heatmap(flow_df_pid.iloc[::-1], ax=axes[1, 1], **flow_params)
    axes[1, 1].set_title('PID Control: Flow')

    plt.suptitle(mape_str, fontsize=16, fontweight='bold')

    # Formatting Labels
    for ax in axes[1, :]:
        ax.set_xlabel('Time Step')
    for ax in axes[:, 0]:
        ax.set_ylabel('Road Segment')

    #plt.tight_layout()
    plt.savefig(plot_dir + 'time_space_plots.png', dpi=300)
    plt.show()


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