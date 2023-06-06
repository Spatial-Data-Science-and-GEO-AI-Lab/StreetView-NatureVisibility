import os
os.environ['USE_PYGEOS'] = '0'

from process_data import download_image, get_models, prepare_folders
from osmnx_road_network import get_road_network, select_points_on_road_network, get_features_on_points
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd
from datetime import timedelta
from time import time
from tqdm import tqdm
import threading
import csv
import sys



def download_images_for_points(gdf, access_token, max_workers, city, file_name):
    # Get image processing models
    processor, model = get_models()

    # Prepare CSV file path
    csv_file = f"gvi-points-{file_name}.csv"
    csv_path = os.path.join("results", city, "gvi", csv_file)

    # Check if the CSV file exists and chose the correct editing mode
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'

    # Create a lock object for thread safety
    results = []
    lock = threading.Lock()
    
    # Open the CSV file in append mode with newline=''
    with open(csv_path, mode, newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(["id", "x", "y", "GVI", "is_panoramic", "missing", "error"])
        
        # Create a ThreadPoolExecutor to process images concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Iterate over the rows in the GeoDataFrame
            for _, row in gdf.iterrows():
                try:
                    # Submit a download_image task to the executor
                    futures.append(executor.submit(download_image, row["id"], row["geometry"], row["image_id"], row["is_panoramic"], access_token, processor, model))
                except Exception as e:
                    print(f"Exception occurred for row {row['id']}: {str(e)}")
            
            # Process the completed futures using tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading images"):
                # Retrieve the result of the completed future
                image_result = future.result()

                # Acquire the lock before appending to results and writing to the CSV file
                with lock:
                    results.append(image_result)
                    writer.writerow(image_result)

    # Return the processed image results
    return results


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