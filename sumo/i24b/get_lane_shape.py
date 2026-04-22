import xml.etree.ElementTree as ET
import json
def parse_sumo_network_with_detectors(net_file, det_file, output_json):
    # 1. Parse the Network File
    net_tree = ET.parse(net_file)
    net_root = net_tree.getroot()
    
    lane_shapes = {}
    mainlane_edges = []
    
    for edge in net_root.findall('edge'):
        edge_id = edge.get('id')
        
        # Skip internal edges
        if edge_id.startswith(':'): 
            continue
            
        # FILTER LOGIC:
        # 1. Must have 3 or more lanes
        # 2. Must NOT contain "Ramp" or "AddedOffRamp" or "AddedOnRamp" in the ID
        lanes = edge.findall('lane')
        is_ramp = "Ramp" in edge_id or "Added" in edge_id
        
        if len(lanes) >= 3 and not is_ramp:
            mainlane_edges.append(edge_id)
            
            # Save shapes only for mainline lanes
            for lane in lanes:
                lane_id = lane.get('id')
                shape_str = lane.get('shape')
                if shape_str:
                    coords = [[float(p.split(',')[0]), float(p.split(',')[1])] 
                              for p in shape_str.split(' ')]
                    lane_shapes[lane_id] = coords

    # 2. Parse the Detector File
    lane_to_det = {}
    try:
        det_tree = ET.parse(det_file)
        det_root = det_tree.getroot()
        for detector in det_root.findall('.//inductionLoop') + det_root.findall('.//laneAreaDetector'):
            det_id = detector.get('id')
            l_id = detector.get('lane')
            if l_id:
                lane_to_det[l_id] = det_id
    except FileNotFoundError:
        print(f"Warning: Detector file {det_file} not found.")

    # 3. Combine and Save
    # CRITICAL: We use 'mainlane_edges' here, NOT a new findall()
    data = {
        "lane_shapes": lane_shapes,
        "lane_to_det": lane_to_det,
        "mainlane": mainlane_edges 
    }

    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(mainlane_edges)} mainline edges to {output_json}")

if __name__ == "__main__":
    # Replace with your actual filename
    parse_sumo_network_with_detectors('i24b.net.xml', '12-15_detectors_gt.xml', 'lane_shapes.json')