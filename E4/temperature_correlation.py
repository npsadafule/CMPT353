import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos, asin, sqrt, pi

# Haversine formula to calculate the distance between two lat/lon points
def haversine_formula(lat1, lon1, lat2, lon2):
    r = 6371 * 1000  # Earth radius in meters
    p = pi / 180

    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))

# Function to calculate the distance between a city and all stations
def calculate_distances(city, stations):
    city_lat = city['latitude']
    city_lon = city['longitude']
    distances = stations.apply(lambda station: haversine_formula(city_lat, city_lon, station['latitude'], station['longitude']), axis=1)
    return distances

# Function to find the closest station's avg_tmax for a city
def closest_station_tmax(city, stations):
    distances = calculate_distances(city, stations)
    closest_station_index = np.argmin(distances)
    return stations.at[closest_station_index, 'avg_tmax']

# Function to load and preprocess station data
def load_and_preprocess_stations(stations_file):
    stations = pd.read_json(stations_file, lines=True)
    stations['avg_tmax'] = stations['avg_tmax'] / 10  # Convert to °C
    return stations

# Function to load and preprocess city data
def load_and_preprocess_cities(city_file):
    cities = pd.read_csv(city_file).dropna()
    cities['area'] = cities['area'] / 1_000_000  # Convert m² to km²
    cities = cities[cities['area'] <= 10_000]    # Filter cities with an area larger than 10,000 km²
    cities['density'] = cities['population'] / cities['area']
    return cities

# Function to create the scatter plot
def plot_temperature_vs_density(cities, output_file):
    plt.scatter(cities['avg_tmax'], cities['density'], alpha=0.4)
    plt.ylabel('Population Density (people/km\u00b2)')
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.title('Temperature vs Population Density')
    plt.savefig(output_file)
    #plt.show()  # Uncomment if you want to show the plot

# Main function to tie everything together
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 temperature_correlation.py <stations.json.gz> <city_data.csv> <output.svg>")
        sys.exit(1)

    stations_file = sys.argv[1]
    city_file = sys.argv[2]
    output_file = sys.argv[3]

    # Load and preprocess data
    stations = load_and_preprocess_stations(stations_file)
    cities = load_and_preprocess_cities(city_file)

    # Calculate the closest station's avg_tmax for each city
    cities['avg_tmax'] = cities.apply(closest_station_tmax, stations=stations, axis=1)

    # Create and save the scatter plot
    plot_temperature_vs_density(cities, output_file)

if __name__ == '__main__':
    main()
