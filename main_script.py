import os
os.environ['USE_PYGEOS'] = '0'

from process_data import process_data, get_models
from osmnx_road_network import get_road_network, select_points_on_road_network, get_features_on_points
import multiprocessing as mp
import geopandas as gpd
import numpy as np
from datetime import timedelta
from time import time
import sys


def download_images_for_points(gdf, city, access_token):
    processor, model = get_models()
    #prepare_folders(city)
    
    images_results = []

    # Split the dataset into parts
    num_processes = mp.cpu_count() # Get the number of CPU cores
    data_parts = np.array_split(gdf, num_processes) # Split the dataset
    
    with mp.get_context("spawn").Pool(processes=num_processes) as pool:
        # Apply the function to each part of the dataset using multiprocessing
        results = pool.starmap(process_data, [(index, data_part, processor, model, city, access_token) for index, data_part in enumerate(data_parts)])

        # Combine the results from all parts
        images_results = [result for part_result in results for result in part_result]

        # Close the pool to release resources
        pool.close()
        pool.join()

    return gpd.GeoDataFrame(images_results, columns=["geometry", "GVI", "is_panoramic", "missing", "error"])


if __name__ == "__main__":
    city = sys.argv[1] # City to analyse
    access_token = sys.argv[2] # Access token for mapillary
    
    road = get_road_network(city)
    points = select_points_on_road_network(road)
    features = get_features_on_points(points, access_token)

    file_path = os.path.join("results", city, "points.gpkg")
    features.to_file(file_path, driver="GPKG")

    features = features.head(10) # I'm using this line for testing

    # Get the initial time
    start_time = time()
    
    results = download_images_for_points(features, city, access_token)
    # Get the final time
    end_time = time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time as "hh:mm:ss"
    formatted_time = str(timedelta(seconds=elapsed_time))

    print(f"Running time: {formatted_time}")

    file_path = os.path.join("results", city, "GVI.gpkg")
    results.to_file(file_path, driver="GPKG")