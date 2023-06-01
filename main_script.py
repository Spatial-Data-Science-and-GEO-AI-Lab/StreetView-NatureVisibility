import os
os.environ['USE_PYGEOS'] = '0'

from process_data import process_data, get_models, prepare_folders
from osmnx_road_network import get_road_network, select_points_on_road_network, get_features_on_points
import multiprocessing as mp
import geopandas as gpd
import numpy as np
from datetime import timedelta
from time import time
import threading
import sys


def download_images_for_points(gdf, access_token, max_workers, city, file_name):
    processor, model = get_models()
    
    lock = threading.Lock()
    
    images_results = []

    # Split the dataset into parts
    num_processes = mp.cpu_count() # Get the number of CPU cores
    data_parts = np.array_split(gdf, num_processes) # Split the dataset
    
    with mp.get_context("spawn").Pool(processes=num_processes) as pool:
        # Apply the function to each part of the dataset using multiprocessing
        results = pool.starmap(process_data, [(index, data_part, processor, model, access_token, max_workers, lock, city, file_name) for index, data_part in enumerate(data_parts)])

        # Combine the results from all parts
        images_results = [result for part_result in results for result in part_result]

        # Close the pool to release resources
        pool.close()
        pool.join()

    return images_results


if __name__ == "__main__":
    args = sys.argv

    city = args[1] # City to analyse
    access_token = args[2] # Access token for mapillary
    file_name = args[3]
    path = args[4] # Path to save the results
    max_workers = int(args[5])
    begin = int(args[6]) if len(args) > 6 else None
    end = int(args[7]) if len(args) > 7 else None
    
    prepare_folders(city, path)

    file_path = os.path.join(path, "results", city, "points", "points.gpkg")    

    if not os.path.exists(file_path):
        road = get_road_network(city)
        points = select_points_on_road_network(road)
        features = get_features_on_points(points, access_token)
        features.to_file(file_path, driver="GPKG")
    else:
        features = gpd.read_file(file_path, layer="points")
    
    features = features.sort_values(by='id')

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