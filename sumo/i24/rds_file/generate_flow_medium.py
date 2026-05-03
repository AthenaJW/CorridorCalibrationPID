import pandas as pd
import numpy as np

'''
generate flows in .rou.xml given detector measurements
quick and easy, not precise
'''
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import csv
from collections import defaultdict

# Configuration - Update these paths to match your files
DETECTOR_LOCATIONS_FILE = "i24_RDS_gt.add.xml"
ROUTES_FILE = "i24_gt.rou.xml"
DETECTOR_MEASUREMENTS_FILE = "rds_file/mediumnet_scenario2.csv"
OUTPUT_ROUTES_FILE = "i24_gt2.rou.xml"
START_MINUTE = 300  # Start of measurement window (minutes)
END_MINUTE = 480    # End of measurement window (minutes)

def main():
    # Convert minutes to seconds for SUMO
    start_time = START_MINUTE * 60
    end_time = END_MINUTE * 60

    # Step 1: Parse detector locations
    detector_edges = {}
    det_tree = ET.parse(DETECTOR_LOCATIONS_FILE)
    for il in det_tree.findall('e1Detector'):
        detector_id = il.get('id')
        lane = il.get('lane')
        edge = lane.split('_')[0]
        detector_edges[detector_id] = edge
    print(detector_edges)

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
    print(DETECTOR_MEASUREMENTS_FILE)
    # Step 4: Process detector measurements
    with open(DETECTOR_MEASUREMENTS_FILE, 'r') as f:
        print(f"Processing measurements from {DETECTOR_MEASUREMENTS_FILE}...")
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
            detector_id = detector_id.replace('.', '_')  # Match the format in the XML
            
            edge = detector_edges.get(detector_id)
            if not edge:
                continue

            # Find routes containing this edge
            for route_id, edges in routes.items():
                if edge in edges:
                    q = int(float(row['qPKW'])) # every 30 sec
                    v = float(row['vPKW']) # TODO v=0 if no vehicle
                    
                    # Aggregate data per route per time interval
                    if q>0:
                        data = time_interval_data[interval][route_id]
                        data['total_q'] += q
                        data['total_v'] += v*0.3048
                        data['count'] += 1

    # Step 5: Create flow elements
    root = rou_tree.getroot()
    
    # Create time-varying flows for each interval and route
    # 1. Pre-aggregate the data into 30-minute buckets
    thirty_min_buckets = {}

    for interval in range(START_MINUTE, END_MINUTE):
        bucket_index = interval // 30  # Group minutes 0-29, 30-59, etc.
        interval_data = time_interval_data.get(interval, {})
        
        if bucket_index not in thirty_min_buckets:
            thirty_min_buckets[bucket_index] = {}
            
        for route_id, data in interval_data.items():
            if route_id not in thirty_min_buckets[bucket_index]:
                thirty_min_buckets[bucket_index][route_id] = {'sum_q': 0.0, 'count': 0}
            
            thirty_min_buckets[bucket_index][route_id]['sum_q'] += data['total_q']
            thirty_min_buckets[bucket_index][route_id]['count'] += 1

    # 2. Generate XML elements based on the averaged buckets
    # This sequence has 6 steps, repeating twice over 12 buckets (6 * 30 mins = 180 mins)
    wave_pattern = [1/2, 1, 1/3, 4/5, 1, 1/3] 

    for bucket_index, routes in thirty_min_buckets.items():
        # Calculate the 30-minute bounds
        base_begin = bucket_index * 1800
        
        # We use the bucket_index to pick the multiplier from the pattern
        # Using modulo (%) allows the pattern to repeat if you have more than 6 buckets
        current_wave_mult = wave_pattern[bucket_index % len(wave_pattern)]
        
        for route_id, agg_data in routes.items():
            if agg_data['count'] == 0:
                continue
                
            avg_q = agg_data['sum_q'] / agg_data['count']
            
            # Your original route-specific multipliers
            multipliers = {'r_4': 12, 'r_3': 4, 'r_0': 16}
            base_mult = multipliers.get(route_id, 2)
            
            # Apply the wave multiplier to the base multiplier
            final_vehs_per_hour = avg_q * base_mult * current_wave_mult
            
            flow_elem = ET.Element('flow', {
                'id': f'flow_{route_id}_b{bucket_index}',
                'route': route_id,
                'begin': str(max(0, base_begin - start_time)),
                'end': str(base_begin + 1800 - start_time),
                'vehsPerHour': str(round(final_vehs_per_hour, 2)),
                'departSpeed': "desired",
                'departLane': "best",
                'type': 'hdv'
            })
            root.append(flow_elem)

    # Step 6: Save modified route file
    rough_string = ET.tostring(root, encoding='UTF-8', xml_declaration=True)
    parsed = minidom.parseString(rough_string)
    with open(OUTPUT_ROUTES_FILE, 'w', encoding='UTF-8') as f:
        f.write(parsed.toprettyxml(indent="    "))  # Use 4 spaces for indentation
    # rou_tree.write(OUTPUT_ROUTES_FILE, encoding='UTF-8', xml_declaration=True)
    print(f"Generated time-varying route file saved to {OUTPUT_ROUTES_FILE}")


def generate_synthetic_rds(filename, duration_mins=180, interval_sec=30):
    # 1. Setup Detectors (Locations based on your IDs)
    # Grouping by highway kilometer markers
    detector_stations = [
                         '56.7_0', '56.7_1', '56.7_2', '56.7_3', '56.7_4', 
                         '56.3_0', '56.3_1', '56.3_2', '56.3_3', '56.3_4',
                         '56.0_0', '56.0_1', '56.0_2', '56.0_3', '56.0_4',
                         '55.3_0', '55.3_1', '55.3_2', '55.3_3',
                         '54.6_0', '54.6_1', '54.6_2', '54.6_3',
                        ]
    lanes_per_station = [1]
    
    # 2. Setup Time (30-second steps)
    timesteps = np.arange(300.0, 300.0 + duration_mins, interval_sec / 60.0)
    
    data = []

    for t in timesteps:
        for station in detector_stations:
            # --- WAVE LOGIC ---
            # Wave 1: Hits station 56.3 at T=305, moves to 54.6 by T=310
            # Wave 2: Hits station 56.3 at T=320, moves to 54.6 by T=325
            km_str, lane = station.split('_')[0], station.split('_')[1]
            km = float(km_str)
            is_wave_1 = (305 + (56.3 - km) * 2 < t < 308 + (56.3 - km) * 2)
            is_wave_2 = (320 + (56.3 - km) * 2 < t < 323 + (56.3 - km) * 2)
            
            
            if is_wave_1 or is_wave_2:
                # Congested Flow: Low speed, Higher relative flow (but limited by capacity)
                vPKW = np.random.uniform(20.0, 45.0)
                qPKW = np.random.uniform(10.0, 20.0)
            else:
                # Free Flow: High speed, Variable flow
                vPKW = np.random.uniform(110.0, 140.0)
                qPKW = np.random.uniform(2.0, 10.0)
            
            # Special case: Lane 3 (leftmost) often has 0 flow/speed in quiet times
            if lane == 3 and not (is_wave_1 or is_wave_2) and np.random.rand() > 0.7:
                vPKW, qPKW = 0.0, 0.0

            data.append({
                "Detector": station,
                "Time": round(t, 2),
                "qPKW": round(qPKW, 2),
                "vPKW": round(vPKW, 2)
            })

    # 3. Create DataFrame and Save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep=';')
    print(f"Successfully generated {filename} with two traffic waves.")

if __name__ == "__main__":
    generate_synthetic_rds("mediumnet_scenario2.csv")
    main()
    
