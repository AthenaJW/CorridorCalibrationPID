import re

input_file = 'mediumnet_0300-0480.csv'
output_file = 'mediumnet_0300-0480_flowrouter.csv'
START_TIME_MINUTES=300.5

import xml.etree.ElementTree as ET

# Change these to your actual filenames
routes_in = "../routes.xml"
flows_in = "../flows.xml"
combined_out = "../i24_flowrouter.rou.xml"

def combine_sumo_files():
    # Start with the routes file as the base
    tree = ET.parse(routes_in)
    root = tree.getroot()

    # Parse the flows file
    flow_tree = ET.parse(flows_in)
    flow_root = flow_tree.getroot()

    # Append every flow/vehicle from the second file into the first
    for element in flow_root:
        root.append(element)

    # Write the combined result
    tree.write(combined_out, encoding="UTF-8", xml_declaration=True)
    print(f"Success! Created {combined_out}")



def format_detector_name(line):
    # Regex logic: 
    # Finds a sequence of digits, a dot, then digits, followed by an underscore.
    # It replaces the dot with an underscore only in that specific pattern.
    return re.sub(r'(\d+)\.(\d+)_', r'\1_\2_', line)

if __name__ == "__main__":
    combine_sumo_files()

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Write the header (adjust column names if dfrouter needs specific titles)
        header = f_in.readline()
        f_out.write(header)
        
        
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            header = f_in.readline()
            f_out.write(header)
            
            for line in f_in:
                if not line.strip(): continue # Skip empty lines
                
                # 1. First, fix the name (dot to underscore) in the raw string
                fixed_name_line = format_detector_name(line)
                
                # 2. Split the ALREADY FIXED line to get the time
                parts = fixed_name_line.strip().split(";")
                
                try:
                    # 3. Shift the time
                    original_time = float(parts[1])
                    shifted_time = original_time - START_TIME_MINUTES
                    parts[1] = str(round(shifted_time, 2))
                    
                    # 4. Reconstruct the line
                    # This now uses the fixed name and the shifted time
                    final_line = ";".join(parts) + "\n"
                    
                    # 5. Write it ONLY ONCE
                    f_out.write(final_line)
                    
                except (ValueError, IndexError):
                    continue # Skip lines with bad data

    print(f"Done! Formatted data saved to {output_file}")

