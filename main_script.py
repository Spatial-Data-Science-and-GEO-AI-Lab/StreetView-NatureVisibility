import os
os.environ['USE_PYGEOS'] = '0'

from process_data import process_data, prepare_folders, get_models
import multiprocessing as mp
import geopandas as gpd
import numpy as np
from datetime import timedelta
from time import time
import sys


def download_images_for_points(gdf, city, access_token, path=""):
    processor, model = get_models()
    prepare_folders(path, city)
    
    images_results = []

    # Split the dataset into parts
    num_processes = mp.cpu_count() # Get the number of CPU cores
    data_parts = np.array_split(gdf, num_processes) # Split the dataset
    
    with mp.get_context("spawn").Pool(processes=num_processes) as pool:
        # Apply the function to each part of the dataset using multiprocessing
        results = pool.starmap(process_data, [(index, data_part, processor, model, city, access_token, path) for index, data_part in enumerate(data_parts)])

        # Combine the results from all parts
        images_results = [result for part_result in results for result in part_result]

        # Close the pool to release resources
        pool.close()
        pool.join()

    return gpd.GeoDataFrame(images_results, columns=["geometry", "GVI", "is_panoramic", "missing", "error"])


if __name__ == "__main__":
    city = sys.argv[1] # City to analyse
    access_token = sys.argv[2] # Access token for mapillary
    file_name = sys.argv[3] # Name of the file where the results are gonna be stored
    path = sys.argv[4] # Path to store the results
    path_to_file = sys.argv[5] # Path to the points file

    gdf_features = gpd.read_file(path_to_file)
    gdf_features = gdf_features.head(10) # I'm using this line for testing

    # Get the initial time
    start_time = time()
    
    results = download_images_for_points(gdf_features, city, access_token)
    # Get the final time
    end_time = time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time as "hh:mm:ss"
    formatted_time = str(timedelta(seconds=elapsed_time))

    print(f"Running time: {formatted_time}")

    file_path = os.path.join(path, "results", city, f"{file_name}.gpkg")
    results.to_file(file_path, driver="GPKG")