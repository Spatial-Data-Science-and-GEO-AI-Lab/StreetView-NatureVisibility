import os
os.environ['USE_PYGEOS'] = '0'

from datetime import timedelta
import geopandas as gpd
from time import time
from PIL import Image
import requests
import json
import csv
import sys


def download_image(geometry, image_metadata, access_token, dir_path):
    header = {'Authorization': 'OAuth {}'.format(access_token)}

    image_id = image_metadata["properties"]["id"]
    is_panoramic = image_metadata["properties"]["is_pano"]
    
    url = 'https://graph.mapillary.com/{}?fields=thumb_original_url'.format(image_id)
    response = requests.get(url, headers=header)
    data = response.json()
    image_url = data["thumb_original_url"]

    try:
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Save original image 
        img_path = os.path.join(dir_path, f"{image_id}_{is_panoramic}.jpg")
        image.save(img_path)

        # Update csv file here
        return [geometry, image_id, is_panoramic, False]
    except:
        # Update csv file here
        return [geometry, image_id, is_panoramic, True]

if __name__ == '__main__':
    args = sys.argv

    city = args[1] # City to analyse
    access_token = args[2] # Access token for mapillary
    path = args[3] if len(args) > 3 else ""
    
    begin = args[4] if len(args) > 4 else None
    end = args[5] if len(args) > 5 else None

    # Read the file as a GeoDataFrame
    file = os.path.join(path, "results", city, "points/points.gpkg")
    gdf_points = gpd.read_file(file, layer="points")

    if begin and end:
        gdf_points = gdf_points.iloc[begin:end]

    dir_path = os.path.join(path, "results", city, "images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    csv_file = 'image_information.csv'
    csv_path = os.path.join(path, "results", city, csv_file)

    # Check if the CSV file exists
    file_exists = os.path.exists(csv_path)

    # Get the initial time
    start_time = time()
    
    # Open the CSV file in append mode with newline=''
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(["Geometry", "image_id", "is_panoramic", "error"])

        for _, point in gdf_points.iterrows():
            feature = point["feature"]
            geometry = point["geometry"]
            feature = json.loads(feature)
            newrow = download_image(geometry, feature, access_token, dir_path)
            
            writer.writerow(newrow)
    
    # Get the final time
    end_time = time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Format the elapsed time as "hh:mm:ss"
    formatted_time = str(timedelta(seconds=elapsed_time))
    print(f"Running time to download the images: {formatted_time}")