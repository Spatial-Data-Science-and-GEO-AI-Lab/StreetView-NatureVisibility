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
    processor, model = get_models()

    csv_file = f"gvi-points-{file_name}.csv"
    csv_path = os.path.join("results", city, "gvi", csv_file)

    # Check if the CSV file exists
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'

    # Create a lock object
    results = []
    lock = threading.Lock()
    
    # Open the CSV file in append mode with newline=''
    with open(csv_path, mode, newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(["id", "x", "y", "GVI", "is_panoramic", "missing", "error"])
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for _, row in gdf.iterrows():
                try:
                    futures.append(executor.submit(download_image, row["id"], row["geometry"], row["image_id"], row["is_panoramic"], access_token, processor, model))
                except Exception as e:
                    print(f"Exception occurred for row {row['id']}: {str(e)}")
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading images"):
                image_result = future.result()

                # Acquire the lock before appending to results
                with lock:
                    results.append(image_result)
                    writer.writerow(image_result)

    return results


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

    file_path_features = os.path.join(path, "results", city, "points", "points.gpkg")  
    file_path_road = os.path.join(path, "results", city, "roads", "roads.gpkg")    

    if not os.path.exists(file_path_features):
        road = get_road_network(city)
        road["geometry"].to_file(file_path_road, driver="GPKG", crs=road.crs)
        points = select_points_on_road_network(road)
        features = get_features_on_points(points, access_token)
        features.to_file(file_path_features, driver="GPKG")
    else:
        features = gpd.read_file(file_path_features, layer="points")
    
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