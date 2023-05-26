import os
os.environ['USE_PYGEOS'] = '0'

from osmnx_road_network import get_road_network, select_points_on_road_network, get_features_on_points
from datetime import timedelta
from time import time
import sys

if __name__ == '__main__':
    args = sys.argv

    city = args[1] # City to analyse
    access_token = args[2] # Access token for mapillary
    path = args[3] if len(args) > 3 else ""

    dir_path = os.path.join(path, "results", city, "points")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Get the initial time
    start_time = time()
    road = get_road_network(city)
    points = select_points_on_road_network(road)
    features = get_features_on_points(points, access_token)
    # Get the final time
    end_time = time()

    file_path = os.path.join(path, "results", city, "points", f"points.gpkg")
    features.to_file(file_path, driver="GPKG")

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time as "hh:mm:ss"
    formatted_time = str(timedelta(seconds=elapsed_time))
    print(f"Running time to get the points gpkg file: {formatted_time}")