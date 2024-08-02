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

# File path
file_path = 'D:\\oldE\\fireballs\\Satellite Data\\bolide-lcs\\USG-CNEOS\\bolideTxtFiles_20220513\\2021.363.031535.txt'

# Parse the file
date, latitude, longitude, peak_brightness, total_energy, intensity_data = parse_file(file_path)

# Define start and end times for integration
start_time = -0.1
end_time = 1

# Compute the integrated intensity
integrated_intensity = compute_integrated_intensity(intensity_data, start_time, end_time)

# Compute the total energy
computed_total_energy = 4 * np.pi * integrated_intensity

# Debug: Print the energy difference
diff = computed_total_energy - total_energy
perc = diff / total_energy *100
print(f'Energy Difference: {diff:.3e} and percentage: {perc:.1f}%, Int_start {start_time} and Int_end {end_time}')

# Convert intensity data to a DataFrame
intensity_df = pd.DataFrame(intensity_data, columns=['Time', 'Intensity'])

# Compute Bolometric Magnitude
intensity_df['M_bol'] = 6 - 2.5 * np.log10(intensity_df['Intensity'].replace(0, np.nan))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

# Plot Intensity vs. Time on the first subplot
ax1.plot(intensity_df['Time'], intensity_df['Intensity'], linestyle='-', color='b')
ax1.axvline(x=start_time, color='red', linestyle='--', label='Start Time')
ax1.axvline(x=end_time, color='red', linestyle='--', label='End Time')
ax1.set_ylabel('Intensity (w/sr)')
ax1.set_title(f'Intensity vs. Time\nEvent Date: {date}, Location: Lat {latitude}°N, Lon {longitude}°W\nPeak Brightness: {peak_brightness} watts/steradian, Total Radiated Energy: {total_energy:.3e} joules\nComputed Total Energy: {computed_total_energy:.3e} joules')
ax1.grid(True)
ax1.legend()

# Plot Bolometric Magnitude vs. Time on the second subplot
ax2.plot(intensity_df['Time'], intensity_df['M_bol'], linestyle='-', color='g')
ax2.axvline(x=start_time, color='red', linestyle='--', label='Start Time')
ax2.axvline(x=end_time, color='red', linestyle='--', label='End Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Bolometric Magnitude (M_bol)')
ax2.set_title('Bolometric Magnitude vs. Time')
ax2.grid(True)

# Set y-axis limits and invert the y-axis
ax2.set_ylim(-22, -15)
ax2.invert_yaxis()

# Show the plot
plt.tight_layout()
plt.show()