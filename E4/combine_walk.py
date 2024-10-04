import os
import pathlib
import sys
import numpy as np
import pandas as pd
from xml.dom.minidom import parse, getDOMImplementation


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    xmlns = 'http://www.topografix.com/GPX/1/0'

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.10f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.10f' % (pt['lon']))
        time = doc.createElement('time')
        time.appendChild(doc.createTextNode(pt['datetime'].strftime("%Y-%m-%dT%H:%M:%SZ")))
        trkpt.appendChild(time)
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    doc.documentElement.setAttribute('xmlns', xmlns)

    with open(output_filename, 'w') as fh:
        fh.write(doc.toprettyxml(indent='  '))


def get_data(input_gpx):
    """
    Parse the GPX file to get timestamp, latitude, and longitude.
    """
    with open(input_gpx, 'r') as f:
        file = parse(f)
    trkpt_tag = file.getElementsByTagName('trkpt')

    lat_list = []
    lon_list = []
    time_list = []
    for i in trkpt_tag:
        lat_list.append(i.getAttribute('lat'))
        lon_list.append(i.getAttribute('lon'))
        time_list.append(i.getElementsByTagName('time')[0].firstChild.data)

    data_frame = pd.DataFrame({'timestamp': time_list, 'lat': lat_list, 'lon': lon_list})
    data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'], utc=True)
    data_frame['lat'] = data_frame['lat'].astype(float)
    data_frame['lon'] = data_frame['lon'].astype(float)
    return data_frame


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 combine_walk.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_directory = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])

    # Load accelerometer, GPS, and phone sensor data
    accl = pd.read_json(input_directory / 'accl.ndjson.gz', lines=True, convert_dates=['timestamp'])[['timestamp', 'x']]
    gps = get_data(input_directory / 'gopro.gpx')
    phone = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']]

    # Use the first timestamp from the accelerometer data to synchronize
    first_time = accl['timestamp'].min()

    # Round the timestamps once before entering the loop
    accl['timestamp'] = accl['timestamp'].round('4S')
    gps['timestamp'] = gps['timestamp'].round('4S')

    # Group accelerometer and GPS data once before the loop
    accl_grouped = accl.groupby(['timestamp']).mean().reset_index()
    gps_grouped = gps.groupby(['timestamp']).mean().reset_index()

    # Find the best time offset using cross-correlation
    highest_cross_correlation = 0
    best_offset = 0
    for offset in np.linspace(-5.0, 5.0, 101):
        # Apply the offset to the phone data timestamps
        phone['timestamp'] = first_time + pd.to_timedelta(phone['time'] + offset, unit='sec')
        phone['timestamp'] = phone['timestamp'].round('4S')

        # Group the phone data by the rounded timestamps
        phone_grouped = phone.groupby(['timestamp']).mean().reset_index()

        # Merge the dataframes on the timestamp
        combined_0 = pd.merge(accl_grouped, gps_grouped, on='timestamp', how='inner')
        combined = pd.merge(combined_0, phone_grouped, on='timestamp', how='inner')

        # Calculate the cross-correlation between gFx (phone) and x (accelerometer)
        dot_product = combined['gFx'].dot(combined['x'])
        cross_correlation = dot_product

        # Keep track of the best offset with the highest cross-correlation
        if cross_correlation > highest_cross_correlation:
            highest_cross_correlation = cross_correlation
            best_offset = offset

    print(f'Best time offset: {best_offset:.1f}')

    # Apply the best offset and save the final combined data
    phone['timestamp'] = first_time + pd.to_timedelta(phone['time'] + best_offset, unit='sec')
    phone['timestamp'] = phone['timestamp'].round('4S')
    phone_grouped = phone.groupby(['timestamp']).mean().reset_index()

    combined_0 = pd.merge(accl_grouped, gps_grouped, on='timestamp', how='inner')
    combined = pd.merge(combined_0, phone_grouped, on='timestamp', how='inner')

    # Rename the 'timestamp' column to 'datetime' for output
    combined.rename(columns={"timestamp": "datetime"}, inplace=True)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Output the data to the specified files
    output_gpx(combined[['datetime', 'lat', 'lon']], output_directory / 'walk.gpx')
    combined[['datetime', 'Bx', 'By']].to_csv(output_directory / 'walk.csv', index=False)


if __name__ == '__main__':
    main()
