import sys 
import pandas as pd 
import numpy as np
from pykalman import KalmanFilter
from xml.dom.minidom import parse, getDOMImplementation
from math import cos, asin, sqrt, pi

# Function to output GPX file from DataFrame
# ref: https://www.geeksforgeeks.org/parse-xml-using-minidom-in-python/
def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

# Function to parse GPX data and return DataFrame
# ref: https://www.geeksforgeeks.org/parsing-xml-with-dom-apis-in-python/?ref=ml_lbp
def get_data(input_gpx):
    """
    Extracts latitude, longitude, and time from the GPX file and returns a DataFrame.
    """
    file = parse(input_gpx)
    trkpt_tag = file.getElementsByTagName('trkpt')

    lat_list, lon_list, time_list = [], [], []
    for i in trkpt_tag:
        lat_list.append(i.getAttribute('lat'))
        lon_list.append(i.getAttribute('lon'))
        time_list.append(i.getElementsByTagName('time')[0].firstChild.data)

    data_frame = pd.DataFrame({'datetime': time_list, 'lat': lat_list, 'lon': lon_list})
    data_frame['datetime'] = pd.to_datetime(data_frame['datetime'], utc=True)
    data_frame['lat'] = data_frame['lat'].astype(float)
    data_frame['lon'] = data_frame['lon'].astype(float)
    return data_frame

# Haversine formula to calculate distance between lat/lon points
# Source: https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
def haversine_formula(lat1, lon1, lat2, lon2):
    """
    Haversine formula to calculate distance between two latitude/longitude points.
    """
    r = 6371 * 1000  # meters
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))

# Function to calculate the total distance using Haversine formula
# ref: https://www.geeksforgeeks.org/apply-function-to-every-row-in-a-pandas-dataframe/
def distance(data_file):
    """
    Calculates the total distance from a DataFrame containing latitude and longitude.
    """
    data_file['lat2'] = data_file['lat'].shift(-1)
    data_file['lon2'] = data_file['lon'].shift(-1)

    distance_meter = data_file.apply(lambda row: haversine_formula(row['lat'], row['lon'], row['lat2'], row['lon2']), axis=1)
    return distance_meter.sum()

# Kalman filter smoothing function
# Took help from smooth_temperature.py from Assignment 2
def smooth(points):
    """
    Applies Kalman Filter to smooth latitude and longitude data.
    """
    initial_state = points[['lat', 'lon']].iloc[0]
    observation_covariance = np.diag([0.71, 0.71]) ** 2  # Adjusted based on GPS noise level
    transition_covariance = np.diag([0.89, 0.89]) ** 2  # Adjusted based on expected motion
    transition = [[1, 0], [0, 1]]  # Constant velocity model

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )

    kalman_smoothed, _ = kf.smooth(points[['lat', 'lon']])
    kalman_data_frame = pd.DataFrame(kalman_smoothed, columns=['lat', 'lon'])
    return kalman_data_frame

# Main function to execute the distance calculation and filtering
def main():
    input_gpx = sys.argv[1]
    input_csv = sys.argv[2]

    # Parse GPX data
    points = get_data(input_gpx).set_index('datetime')

    # Read sensor data from CSV and combine
    sensor_data = pd.read_csv(input_csv, parse_dates=['datetime']).set_index('datetime')
    points['Bx'] = sensor_data['Bx']
    points['By'] = sensor_data['By']

    # Calculate unfiltered distance
    dist = distance(points)
    print(f'Unfiltered distance: {dist:.2f}')

    # Apply Kalman smoothing
    smoothed_points = smooth(points)

    # Calculate filtered distance
    smoothed_dist = distance(smoothed_points)
    print(f'Filtered distance: {smoothed_dist:.2f}')

    # Output the smoothed points as a GPX file
    output_gpx(smoothed_points, 'out.gpx')

if __name__ == '__main__':
    main()
