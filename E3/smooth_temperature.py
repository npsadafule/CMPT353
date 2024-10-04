import sys 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess # From lecture slides
from pykalman import KalmanFilter                             # From lecture slides 
import numpy as np

# Get the command-line argument (CSV filename)
filename1 = sys.argv[1]

# Read CSV file into a DataFrame (cpu_data), parsing the timestamp column
cpu_data = pd.read_csv(filename1, parse_dates=['timestamp'])  # Parsing timestamp to datetime
cpu_data['temperature'] = cpu_data['temperature'].astype('float')  # Ensuring temperature is float
cpu_data = cpu_data.dropna()  # Drop any rows with NaN values

# LOESS Smoothing
# Adjusted the frac value to find the right balance between noise reduction and responsiveness
loess_smoothed = lowess(endog=cpu_data['temperature'], exog=cpu_data['timestamp'], frac=0.04) 

# Kalman Smoothing
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

# Initial Kalman Filter settings
initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([0.8] * kalman_data.shape[1]) ** 2  
transition_covariance = np.diag([0.045] * kalman_data.shape[1]) ** 2  
transition = [[0.94, 0.5, 0.2, -0.001], [0.1, 0.4, 2.1, 0], [0, 0, 0.94, 0], [0, 0, 0, 1]] 

# If fan_rpm is not present, reduce the transition matrix accordingly
if kalman_data.shape[1] == 3:
    transition = [[0.94, 0.5, 0.2], [0.1, 0.4, 2.1], [0, 0, 0.94]]

# Apply Kalman filter
kf = KalmanFilter(
    initial_state_mean=initial_state,
    initial_state_covariance=observation_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
)

kalman_smoothed, _ = kf.smooth(kalman_data)

# Plotting the data
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label='Raw Data')  # Plot raw data
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-', label='LOESS Smoothed')  # LOESS-smoothed line
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label='Kalman Smoothed')  # Kalman-smoothed line

# Add legend, labels, and title
plt.legend(['Raw Data', 'LOESS Smoothed', 'Kalman Smoothed'])
plt.xlabel('Timestamp')
plt.ylabel('Temperature (°C)')
plt.title('CPU Temperature Noise Reduction')

# Save the plot as cpu.svg
plt.savefig('cpu.svg')
