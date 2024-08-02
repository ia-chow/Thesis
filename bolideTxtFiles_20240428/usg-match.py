import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to parse the file and extract required information
def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract date of the event
    date_match = re.search(r'At (\d{2}:\d{2}:\d{2} UT on \d{2} \w+ \d{4})', content)
    date = date_match.group(1) if date_match else 'Unknown Date'

    # Extract latitude and longitude
    lat_match = re.search(r'Lat ([\d\.\+\-]+) °N', content)
    lon_match = re.search(r'Lon ([\d\.\+\-]+) °W', content)
    latitude = lat_match.group(1) if lat_match else 'Unknown Latitude'
    longitude = lon_match.group(1) if lon_match else 'Unknown Longitude'

    # Extract peak brightness
    brightness_match = re.search(r'peak brightness of the flash\s*was\s*determined\s*to\s*be\s*approximately\s*([\d\.e\+\-]+)\s*watts/steradian', content)
    peak_brightness = brightness_match.group(1) if brightness_match else 'Unknown Brightness'

    # Extract total radiated energy
    energy_match = re.search(r'total radiated flash\s*energy\s*([\d\.e\+\-]+)\s*joules', content)
    total_energy = float(energy_match.group(1)) if energy_match else 'Unknown Energy'

    # Extract Intensity array
    intensity_match = re.search(r'Intensity=\[(.*?)\]', content, re.DOTALL)
    intensity_data = intensity_match.group(1).strip() if intensity_match else ''
    
    # Parse the intensity data into a list of tuples
    intensity_tuples = eval(f'[{intensity_data}]')

    return date, latitude, longitude, peak_brightness, total_energy, intensity_tuples

# Function to compute the integrated intensity
def compute_integrated_intensity(intensity_data, start_time, end_time):
    # Convert intensity data to a DataFrame
    intensity_df = pd.DataFrame(intensity_data, columns=['Time', 'Intensity'])
    
    # Filter data between start_time and end_time
    filtered_df = intensity_df[(intensity_df['Time'] >= start_time) & (intensity_df['Time'] <= end_time)]
    
    # Compute the area under the curve using the trapezoidal rule
    integrated_intensity = np.trapz(filtered_df['Intensity'], filtered_df['Time'])
    
    return integrated_intensity

# Directory path
home_directory = os.path.expanduser("D:\\oldE\\fireballs\\Satellite Data\\bolide-lcs\\USG-CNEOS\\bolideTxtFiles_20220513\\")

# List to store results
results = []

# Loop through all files in the home directory
for filename in os.listdir(home_directory):
    # Check if the file has a .txt extension
    if filename.endswith('.txt'):
        file_path = os.path.join(home_directory, filename)
        
        # Parse the file
        date, latitude, longitude, peak_brightness, total_energy, intensity_data = parse_file(file_path)

      # Parse the file
        date, latitude, longitude, peak_brightness, total_energy, intensity_data = parse_file(file_path)

   # Initialize start and end times for integration
        initial_start_time = -5
        initial_end_time = 5
        end_time = initial_end_time
        perc = float('inf')  # Initialize perc to a high value
        
        # Loop to adjust start and end times until perc is below 10%
        while perc >= 10 and end_time > initial_start_time:
            start_time = initial_start_time
            while perc >= 10 and start_time < end_time:
                # Compute the integrated intensity
                integrated_intensity = compute_integrated_intensity(intensity_data, start_time, end_time)

                # Compute the total energy
                computed_total_energy = 4 * np.pi * integrated_intensity

                # Compute the energy difference
                diff = computed_total_energy - total_energy
                perc = abs(diff / total_energy * 100)
                print(f'{filename} Energy Difference: {diff:.3e} and percentage: {perc:.1f}%, start={start_time:5.2f}, end={end_time:5.2f}')
                # If percentage difference is below 20%, break the loop
                if perc < 20:
                    break

                # Increment start_time
                start_time += 0.1

            # Decrement end_time
            end_time -= 0.1

        # Store the result
        results.append((filename, diff, perc, start_time, end_time))

# Sort results by the absolute value of percentage difference in descending order
results.sort(key=lambda x: abs(x[2]), reverse=True)

# Open output file and write sorted results
with open("energy_comp.dat", 'a+') as file_out:
    for filename, diff, perc, start_time, end_time in results:
        print(f'{filename} Energy Difference: {diff:.3e} and percentage: {perc:.1f}%, start_time: {start_time:5.2f}, end time: {end_time:5.2f}')
        file_out.write(f"{filename}, {diff:.3e}, {perc:.1f}, {start_time:5.2f}, {end_time:5.2f} \n")

