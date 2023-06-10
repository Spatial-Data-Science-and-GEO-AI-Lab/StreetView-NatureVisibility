import os
os.environ['USE_PYGEOS'] = '0'

from process_data import download_images_for_points, prepare_folders
from osmnx_road_network import get_road_network, select_points_on_road_network, get_features_on_points

import geopandas as gpd
from datetime import timedelta
from time import time
import sys


if __name__ == "__main__":
    # When running the code from the terminal, this is the order in which the parameters should be entered
    args = sys.argv
    city = args[1] # Name of the city to analyse (e.g. Amsterdam, Netherlands)
    distance = int(args[2])  # Distance between the sample points in meters
    access_token = args[3] # Access token for mapillary (e.g. MLY|)
    file_name = args[4] # Name of the csv file in which the points with the GVI value are going to be stored
    max_workers = int(args[5]) # Number of threads that are going to be used, a good starting point could be the number of cores of the computer
    begin = int(args[6]) if len(args) > 6 else None
    end = int(args[7]) if len(args) > 7 else None
    
    prepare_folders(city)

    file_path_features = os.path.join("results", city, "points", "points.gpkg")  
    file_path_road = os.path.join("results", city, "roads", "roads.gpkg")    

    if not os.path.exists(file_path_features):
        # Get the sample points and the features assigned to each point
        road = get_road_network(city)

        # Save road in gpkg file
        road["index"] = road.index
        road["index"] = road["index"].astype(str)
        road["highway"] = road["highway"].astype(str)
        road["length"] = road["length"].astype(float)
        road[["index", "geometry", "length", "highway"]].to_file(file_path_road, driver="GPKG", crs=road.crs)
        road["geometry"].to_file(file_path_road, driver="GPKG", crs=road.crs)
        
        points = select_points_on_road_network(road, distance)
        features = get_features_on_points(points, access_token, distance)
        features.to_file(file_path_features, driver="GPKG")
    else:
        # If the points file already exists, then we use it to continue with the analysis
        features = gpd.read_file(file_path_features, layer="points")
    
    features = features.sort_values(by='id')

    # If we include a begin and end value, then the dataframe is splitted and we are going to analyse just that points
    if begin != None and end != None:   
        features = features.iloc[begin:end]

    # Get the initial time
    start_time = time()
    
    results = download_images_for_points(features, access_token, max_workers, city, file_name)
    
    # Get the final time
    end_time = time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time as "hh:mm:ss"
    formatted_time = str(timedelta(seconds=elapsed_time))

    print(f"Running time: {formatted_time}")