import os
os.environ['USE_PYGEOS'] = '0'

from process_data import process_data, get_models, get_gvi_per_buffer
from osmnx_road_network import get_road_network, select_points_on_road_network, get_features_on_points, get_road_network_with_points, select_points_within_buffers
import multiprocessing as mp
import geopandas as gpd
import numpy as np
from datetime import timedelta
from time import time
import sys


def download_images_for_points(gdf, access_token):
    processor, model = get_models()
    #prepare_folders(city)
    
    images_results = []

    # Split the dataset into parts
    num_processes = mp.cpu_count() # Get the number of CPU cores
    data_parts = np.array_split(gdf, num_processes) # Split the dataset
    
    with mp.get_context("spawn").Pool(processes=num_processes) as pool:
        # Apply the function to each part of the dataset using multiprocessing
        results = pool.starmap(process_data, [(index, data_part, processor, model, access_token) for index, data_part in enumerate(data_parts)])

        # Combine the results from all parts
        images_results = [result for part_result in results for result in part_result]

        # Close the pool to release resources
        pool.close()
        pool.join()

    return gpd.GeoDataFrame(images_results, columns=["geometry", "GVI", "is_panoramic", "missing", "error"], crs='EPSG:4326')



if __name__ == "__main__":
    args = sys.argv

    if len(sys.argv) == 3:
        city = args[1] # City to analyse
        access_token = args[2] # Access token for mapillary

        dir_path = os.path.join("results", city)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        road = get_road_network(city)
        points = select_points_on_road_network(road)
        features = get_features_on_points(points, access_token)

        file_path = os.path.join("results", city, "points.gpkg")
        features.to_file(file_path, driver="GPKG")

        features = features.iloc[750:800] # I'm using this line for testing

        # Get the initial time
        start_time = time()
    
        results = download_images_for_points(features, access_token)
        # Get the final time
        end_time = time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Format the elapsed time as "hh:mm:ss"
        formatted_time = str(timedelta(seconds=elapsed_time))

        print(f"Running time: {formatted_time}")

        file_path = os.path.join("results", city, "GVI.gpkg")
        results.to_file(file_path, driver="GPKG")
    
    else:
        file = args[1] # File with address points
        layer_name = args[2]
        buffer_distance = int(args[3]) # in decimal meters
        access_token = args[4] # Access token for mapillary

        # Read the file as a GeoDataFrame
        gdf_points = gpd.read_file(file, layer=layer_name)
        
        # Get the buffer for each point
        gdf_points['buffer'] = gdf_points['geometry'].buffer(buffer_distance)

        # Reproject the GeoDataFrame to WGS84 (EPSG:4326)
        gdf_points.to_crs(epsg=4326, inplace=True)

        # Reproject the 'buffer' column to EPSG 4326
        gdf_points['buffer'] = gdf_points['buffer'].to_crs(epsg=4326)

        # Get the sample points
        road = get_road_network_with_points(gdf_points['buffer'])

        # Reproject the GeoDataFrame to WGS84 (EPSG:4326)
        road.to_crs(epsg=4326, inplace=True)

        # Get the list of points of the road network that fall within any buffer
        road_points = select_points_on_road_network(road)
        points = select_points_within_buffers(gdf_points, road_points)

        # Get features for each point
        features = get_features_on_points(points, access_token)

        file_path = os.path.join("results", "points.gpkg")
        features.to_file(file_path, driver="GPKG")

        # Get the initial time
        start_time = time()
    
        gvi_per_point = download_images_for_points(features, access_token)
        # Get the final time
        end_time = time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Format the elapsed time as "hh:mm:ss"
        formatted_time = str(timedelta(seconds=elapsed_time))

        print(f"Running time: {formatted_time}")

        # Get the GVI per buffer
        results = get_gvi_per_buffer(gdf_points, gvi_per_point)

        file_path = os.path.join("results", "GVI-points.gpkg")
        gvi_per_point.to_file(file_path, driver="GPKG")

        file_path = os.path.join("results", "GVI-buffer.gpkg")
        results.to_file(file_path, driver="GPKG")