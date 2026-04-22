'''
generate flows in .rou.xml given detector measurements
quick and easy, not precise
'''
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import csv
from collections import defaultdict
import pandas as pd

# --- Configuration Constants ---
# DETECTOR_LOCATIONS_FILE = '12-15_detectors.xml'
# ROUTES_FILE = 'i24b.rou.xml'
# DETECTOR_MEASUREMENTS_FILE = 'measurements.csv'
# START_MINUTE = 60  # e.g., start at 1:00 AM
# END_MINUTE = 120   # e.g., end at 2:00 AM

DETECTOR_LOCATIONS_FILE = '12-15_detectors_gt.xml'
ROUTES_FILE = 'i24b_gt.rou.xml'
OUTPUT_ROUTES_FILE = 'i24b_gt2.rou.xml'
DETECTOR_MEASUREMENTS_FILE = 'data/RDS/detections_0360-0600.csv'
START_MINUTE = 360  # e.g., start at 1:00 AM
END_MINUTE = 600   # e.g., end at 2:00 AM


def get_flow_data_list(detector_measurements_file, start_minute, end_minute):
    """
    Parses detectors and measurements to return a list of dictionaries
    representing traffic flows per route per minute.
    """
    start_time_seconds = start_minute * 60

    # Step 1: Parse detector locations (mapping detector ID to road edge)
    detector_edges = {}
    try:
        det_tree = ET.parse(DETECTOR_LOCATIONS_FILE)
        for il in det_tree.findall('inductionLoop'):
            detector_id = il.get('id')
            lane = il.get('lane')
            if lane:
                edge = lane.split('_')[0]
                detector_edges[detector_id] = edge
    except FileNotFoundError:
        print(f"Error: {DETECTOR_LOCATIONS_FILE} not found.")
        return []
    

    # Step 2: Parse routes (mapping route ID to list of edges)
    routes = {}
    try:
        rou_tree = ET.parse(ROUTES_FILE)
        for route in rou_tree.findall('route'):
            route_id = route.get('id')
            edges = route.get('edges').split()
            routes[route_id] = edges
    except FileNotFoundError:
        print(f"Error: {ROUTES_FILE} not found.")
        return []

    # Step 3: Initialize data structure for aggregation
    # interval -> route_id -> metrics
    time_interval_data = defaultdict(
        lambda: defaultdict(lambda: {'total_q': 0, 'total_v': 0.0, 'count': 0})
    )

    # Step 4: Process detector measurements from CSV
    try:
        with open(detector_measurements_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                try:
                    time_min = float(rds_timestamp_to_minutes(row['timestamp']))
                    # Multiply by 2 to get 30-second buckets
                    interval_30s = int(time_min * 2) 
                except (ValueError, KeyError):
                    continue
                
                # Adjust filter for 30s intervals
                if not (start_minute * 2 <= interval_30s < end_minute * 2):
                    continue
                detector_id = row['Detector']
                edge = detector_edges.get(detector_id)
                
                if not (edge and detector_id):
                    continue
                for route_id, edges in routes.items():
                    if edge in edges:
                        q = float(row.get('qPKW', 0))
                        v = float(row.get('vPKW', 0.0))
                        
                        if q > 0:
                            data = time_interval_data[interval_30s][route_id]
                            data['total_q'] += q
                            data['total_v'] += v * 0.3048
                            data['count'] += 1
    except FileNotFoundError:
        print(f"Error: {DETECTOR_MEASUREMENTS_FILE} not found.")
        return []

    # Step 5: Construct the list of flow dictionaries
    flows_list = {name: [] for name in routes.keys()}
    
    for interval_30s in range(start_minute * 2, end_minute * 2):
        interval_data = time_interval_data.get(interval_30s, {})
        
        # --- Filtered Imputation Prep ---
        # Calculate avg flow using only routes that ARE NOT '25'
        secondary_flows = [
            d['total_q'] * 120 
            for r_id, d in interval_data.items() 
            if d['count'] > 0 and r_id != '25'
        ]
        
        avg_secondary_flow = sum(secondary_flows) / len(secondary_flows) if secondary_flows else 0.0

        for route_id in routes.keys():
            data = interval_data.get(route_id)
            
            if data and data['count'] > 0:
                # Use actual measured data if available
                flow_value = float(data['total_q'] * 120)
            else:
                # Impute missing data
                if route_id == '25':
                    # If the mainlane is missing data, you might want it to be 0 
                    # or use a different logic, as secondary roads shouldn't dictate highway flow.
                    flow_value = 0.0 
                else:
                    # Use the average of the other secondary routes
                    flow_value = float(avg_secondary_flow)
            
            flows_list[route_id].append(flow_value)

    return flows_list

def rds_timestamp_to_minutes(timestamp):
    """Convert RDS timestamp (HH:MM:SS) to minutes since midnight."""
    h, m, s = map(int, timestamp.split(':'))
    return h * 60 + m + s / 60.0

def shift_flow_times(input_file, output_file, shift_amount=1800):
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find all flow elements
    for flow in root.findall('flow'):
        begin = float(flow.get('begin'))
        end = float(flow.get('end'))

        # Check if the flow exists within or after the 1800 window
        if end <= shift_amount:
            # If the entire flow happens before 1800, we remove it
            root.remove(flow)
        else:
            # Shift the times. We use max(0, ...) to ensure no negative start times
            new_begin = max(0.0, begin - shift_amount)
            new_end = max(0.0, end - shift_amount)
            
            flow.set('begin', str(round(new_begin, 2)))
            flow.set('end', str(round(new_end, 2)))

    # Write the modified XML back to a file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Successfully shifted times and saved to {output_file}")


def main():
    # Convert minutes to seconds for SUMO
    start_time = START_MINUTE * 60
    end_time = END_MINUTE * 60

    # Step 1: Parse detector locations
    detector_edges = {}
    det_tree = ET.parse(DETECTOR_LOCATIONS_FILE)
    for il in det_tree.findall('inductionLoop'):
        detector_id = il.get('id')
        lane = il.get('lane')
        edge = lane.split('_')[0]
        detector_edges[detector_id] = edge
    

    # Step 2: Parse routes
    routes = {}
    rou_tree = ET.parse(ROUTES_FILE)
    for route in rou_tree.findall('route'):
        route_id = route.get('id')
        edges = route.get('edges').split()
        routes[route_id] = edges

    # Step 3: Initialize data structure for time-varying flows
    time_interval_data = defaultdict(
        lambda: defaultdict(lambda: {'total_q': 0, 'total_v': 0.0, 'count': 0})
    )

    # Step 4: Process detector measurements
    with open(DETECTOR_MEASUREMENTS_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            route_count = 0
            try:
                time_min = float(row['Time'])
                interval = int(time_min)  # 1-minute interval
            except ValueError:
                continue
            
            # Filter data within our time window
            if not (START_MINUTE <= interval < END_MINUTE):
                continue


            detector_id = row['Detector']
            edge = detector_edges.get(detector_id)
            if not edge:
                continue

            # Find routes containing this edge
            for route_id, edges in routes.items():
                if edge in edges:
                    q = int(row['qPKW']) # every 30 sec
                    v = float(row['vPKW']) # TODO v=0 if no vehicle
                    
                    # Aggregate data per route per time interval
                    if q>0:
                        data = time_interval_data[interval][route_id]
                        data['total_q'] += q
                        route_count += 1
                        data['total_v'] += v*0.3048
                        data['count'] += 1

    # Step 5: Create a NEW root to ensure we don't just append to old data
    new_root = ET.Element('routes')
    
    # Optional: Re-add the original route definitions if needed
    # (Since flows need route IDs, we usually keep the <route> elements)
    for route_id, edges in routes.items():
        route_elem = ET.SubElement(new_root, 'route', {
            'id': route_id,
            'edges': " ".join(edges)
        })

    # Create time-varying flows
    for interval in range(START_MINUTE, END_MINUTE):
        interval_data = time_interval_data.get(interval, {})

        for route_id, data in interval_data.items():
            if data['count'] == 0:
                continue
            
            total_q = data['total_q']
            # Correcting the math: 60 if interval is 1 minute
            vehs_per_hour = total_q * 60 
            
            begin = interval * 60
            end = (interval + 1) * 60
            
            if route_id != '25':
                flow_elem = ET.SubElement(new_root, 'flow', {
                    'id': f'flow_{route_id}_{interval}',
                    'route': route_id,
                    'begin': str(begin - start_time),
                    'end': str(end - start_time),
                    'vehsPerHour': str(vehs_per_hour//30),
                    'departSpeed': "desired",
                    'type': 'hdv'
                })
            else:
                flow_elem = ET.SubElement(new_root, 'flow', {
                    'id': f'flow_{route_id}_{interval}',
                    'route': route_id,
                    'begin': str(begin - start_time),
                    'end': str(end - start_time),
                    'vehsPerHour': str(vehs_per_hour//5),
                    'departSpeed': "desired",
                    'type': 'hdv'
                })

    # Step 6: Save (The 'w' flag in open() already overwrites the file)
    rough_string = ET.tostring(new_root, encoding='UTF-8')
    parsed = minidom.parseString(rough_string)
    with open(OUTPUT_ROUTES_FILE, 'w', encoding='UTF-8') as f:
        # toprettyxml will now write the CLEAN list to the file
        f.write(parsed.toprettyxml(indent="    "))

    print(f"Success! Generated flow data and saved to {OUTPUT_ROUTES_FILE}")



if __name__ == '__main__':
    main()
    shift_flow_times(OUTPUT_ROUTES_FILE, OUTPUT_ROUTES_FILE, shift_amount=1800) # optionally shift times if needed
