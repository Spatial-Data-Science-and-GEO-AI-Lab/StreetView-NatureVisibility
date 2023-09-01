import os
os.environ['USE_PYGEOS'] = '0'

import modules.process_data as process_data
import modules.osmnx_road_network as road_network

import geopandas as gpd
from datetime import timedelta
from time import time
import random
import sys


if __name__ == "__main__":
    # When running the code from the terminal, this is the order in which the parameters should be entered
    args = sys.argv
    city = args[1] # Name of the city to analyse (e.g. Amsterdam, Netherlands)
    distance = int(args[2])  # Distance between the sample points in meters
    cut_by_road_centres = int(args[3])  # Determine if panoramic images are going to be cropped using the road centres
    access_token = args[4] # Access token for mapillary (e.g. MLY|)
    file_name = args[5] # Name of the csv file in which the points with the GVI value are going to be stored
    max_workers = int(args[6]) # Number of threads that are going to be used, a good starting point could be the number of cores of the computer
    num_sample_images = int(args[7])
    begin = int(args[8]) if len(args) > 8 else None
    end = int(args[9]) if len(args) > 9 else None
    
    process_data.prepare_folders(city)

    file_path_features = os.path.join("results", city, "points", "points.gpkg")  
    file_path_road = os.path.join("results", city, "roads", "roads.gpkg")    

    if not os.path.exists(file_path_features):
        # Get the sample points and the features assigned to each point
        road = road_network.get_road_network(city)

        # Save road in gpkg file
        road["index"] = road.index
        road["index"] = road["index"].astype(str)
        road["highway"] = road["highway"].astype(str)
        road["length"] = road["length"].astype(float)
        road[["index", "geometry", "length", "highway"]].to_file(file_path_road, driver="GPKG", crs=road.crs)
        
        points = road_network.select_points_on_road_network(road, distance)
        features = road_network.get_features_on_points(points, access_token, distance)
        features.to_file(file_path_features, driver="GPKG")
    else:
        # If the points file already exists, then we use it to continue with the analysis
        features = gpd.read_file(file_path_features, layer="points")
    
    features = features.sort_values(by='id')

    # If we include a begin and end value, then the dataframe is splitted and we are going to analyse just that points
    if begin != None and end != None:   
        features = features.iloc[begin:end]
    
    # Get a list of n random row indices
    sample_indices = random.sample(range(len(features)), num_sample_images)
    # Create a new column 'random_flag' and set it to False for all rows
    features["save_sample"] = False

    # Set True for the randomly selected rows
    features.loc[sample_indices, "save_sample"] = True

    # Get the initial time
    start_time = time()
    
    results = process_data.download_images_for_points(features, access_token, max_workers, cut_by_road_centres, city, file_name)
    
    # Get the final time
    end_time = time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time as "hh:mm:ss"
    formatted_time = str(timedelta(seconds=elapsed_time))

    print(f"Running time: {formatted_time}")