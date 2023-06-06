import os
os.environ['USE_PYGEOS'] = '0'

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.signal import find_peaks

import geopandas as gpd
import pandas as pd
import numpy as np
import threading
import csv
import torch

from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point
from scipy.spatial import cKDTree
from geopandas.tools import sjoin
import osmnx as ox
import mercantile

from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
import requests


def prepare_folders(path=""):
    dir_path = os.path.join(path, "results")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



""" CODE TO GET ROAD NETWORKS AND SAMPLE POINTS """
def get_road_network_with_points(buffered_points):
    # Get the bounds (bounding box) of the buffered points
    bounding_box = buffered_points.bounds

    # Unpack the bounds into separate variables
    min_x = bounding_box['minx'].min()
    min_y = bounding_box['miny'].min()
    max_x = bounding_box['maxx'].max()
    max_y = bounding_box['maxy'].max()

    # Get the road network within the bounding box
    G = ox.graph_from_bbox(max_y, min_y, max_x, min_x, network_type='drive', simplify=True)

    # Create a set to store unique road identifiers
    unique_roads = set()
    # Create a new graph to store the simplified road network
    G_simplified = G.copy()

    # Iterate over each road segment
    for u, v, key, data in G.edges(keys=True, data=True):
        # Check if the road segment is a duplicate
        if (v, u) in unique_roads:
            # Remove the duplicate road segment
            G_simplified.remove_edge(u, v, key)
        else:
            # Add the road segment to the set of unique roads
            unique_roads.add((u, v))
    
    # Update the graph with the simplified road network
    G = G_simplified
    
    #Project the graph from latitude-longitude coordinates to a local projection (in meters)
    G_proj = ox.project_graph(G)

    # Convert the projected graph to a GeoDataFrame
    _, edges = ox.graph_to_gdfs(G_proj)

    return edges


# Get a list of points over the road map with a N distance between them
def select_points_on_road_network(roads, N=50):
    points = []
    # Iterate over each road
    
    for row in roads.itertuples(index=True, name='Road'):
        # Get the LineString object from the geometry
        linestring = row.geometry

        # Calculate the distance along the linestring and create points every 50 meters
        for distance in range(0, int(linestring.length), N):
            # Get the point on the road at the current position
            point = linestring.interpolate(distance)

            # Add the curent point to the list of points
            points.append(point)
    
    # Convert the list of points to a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(geometry=points)

    # Set the same CRS as the road dataframes for the points dataframe
    gdf_points.set_crs(roads.crs, inplace=True)

    # Drop duplicate rows based on the geometry column
    gdf_points = gdf_points.drop_duplicates(subset=['geometry'])
    gdf_points = gdf_points.reset_index(drop=True)

    return gdf_points


def get_features_for_tile(tile, access_token):
    tile_url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}"
    response = requests.get(tile_url)
    result = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer="image")
    return [tile, result]


def get_features_on_points(points, access_token, max_distance=50, zoom=14):
    # Add a new column to gdf_points that contains the tile coordinates for each point
    points["tile"] = [mercantile.tile(x, y, zoom) for x, y in zip(points.geometry.x, points.geometry.y)]

    # Group the points by their corresponding tiles
    groups = points.groupby("tile")

    # Download the tiles and extract the features for each group
    features = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        for tile, _ in groups:
            futures.append(executor.submit(get_features_for_tile, tile, access_token))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading tiles"):
            result = future.result()
            features.append(result)

    pd_features = pd.DataFrame(features, columns=["tile", "features"])

    # Compute distances between each feature and all the points in gdf_points
    feature_points = gpd.GeoDataFrame(
        [(Point(f["geometry"]["coordinates"]), f) for row in pd_features["features"] for f in row["features"]],
        columns=["geometry", "feature"],
        geometry="geometry",
        crs=4326
    )

    # Transform from EPSG:4326 (world °) to EPSG:32662 (world meters)
    feature_points.to_crs(epsg=32634, inplace=True)
    points.to_crs(epsg=32634, inplace=True)

    feature_tree = cKDTree(feature_points["geometry"].apply(lambda p: [p.x, p.y]).tolist())
    distances, indices = feature_tree.query(points["geometry"].apply(lambda p: [p.x, p.y]).tolist(), k=1, distance_upper_bound=max_distance)

    # Create a list to store the closest features and distances
    closest_features = [feature_points.loc[i, "feature"] if np.isfinite(distances[idx]) else None for idx, i in enumerate(indices)]
    closest_distances = [distances[idx] if np.isfinite(distances[idx]) else None for idx in range(len(distances))]

    # Store the closest feature for each point in the "feature" column of the points DataFrame
    points["feature"] = closest_features

    # Store the distances as a new column in points
    points["distance"] = closest_distances

    # Store image id and is panoramic information as part of the dataframe
    points["image_id"] = points.apply(lambda row: str(row["feature"]["properties"]["id"]) if row["feature"] else "", axis=1)
    points["image_id"] = points["image_id"].astype(str)
    
    points["is_panoramic"] = points.apply(lambda row: bool(row["feature"]["properties"]["is_pano"]) if row["feature"] else None, axis=1)
    points["is_panoramic"] = points["is_panoramic"].astype(bool)

    # Convert results to geodataframe
    points["tile"] = points["tile"].astype(str)

    # Save the current index as a column
    points["id"] = points.index

    # Reset the index
    points = points.reset_index(drop=True)

    # Transform the coordinate reference system to EPSG 4326
    points.to_crs(epsg=4326, inplace=True)
    
    return points


def select_points_within_buffers(buffered_points, road_points):
    buffered_points.to_crs(epsg=4326, inplace=True)
    road_points.to_crs(epsg=4326, inplace=True)
    
    points_within_buffers = sjoin(road_points, buffered_points.set_geometry('buffer'), how='inner', predicate='within')

    # Get the unique points that fall within any buffer
    unique_points = points_within_buffers['geometry_left'].unique()

    # Create a new GeoDataFrame with the points that fall within any buffer
    return gpd.GeoDataFrame(geometry=[Point(p.x, p.y) for p in unique_points], crs=buffered_points.crs)



""" CODE TO PROCESS IMAGES """
def get_models():
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    model = model.to(device)
    return processor, model



def segment_images(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = model(**inputs)
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].to('cpu')
        else:
            outputs = model(**inputs)
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            
    return segmentation


# Based on Matthew Danish code (https://github.com/mrd/vsvi_filter/tree/master)
def run_length_encoding(in_array):
    image_array = np.asarray(in_array)
    length = len(image_array)
    if length == 0: 
        return (None, None, None)
    else:
        pairwise_unequal = image_array[1:] != image_array[:-1]
        change_points = np.append(np.where(pairwise_unequal), length - 1)   # must include last element posi
        run_lengths = np.diff(np.append(-1, change_points))       # run lengths
        return(run_lengths, image_array[change_points])

def get_road_pixels_per_column(prediction):
    road_pixels = prediction == 0.0 # The label for the roads is 0
    road_pixels_per_col = np.zeros(road_pixels.shape[1])
    
    for i in range(road_pixels.shape[1]):
        run_lengths, values = run_length_encoding(road_pixels[:,i])
        road_pixels_per_col[i] = run_lengths[values.nonzero()].max(initial=0)
    return road_pixels_per_col

def get_road_centres(prediction, distance=2000, prominence=100):
    road_pixels_per_col = get_road_pixels_per_column(prediction)
    peaks, _ = find_peaks(road_pixels_per_col, distance=distance, prominence=prominence)
    
    return peaks


def find_road_centre(segmentation):
	distance = int(2000 * segmentation.shape[1] // 5760)
	prominence = int(100 * segmentation.shape[0] // 2880)
	
	centres = get_road_centres(segmentation, distance=distance, prominence=prominence)
	
	return centres


def crop_panoramic_images(original_width, image, segmentation, road_centre):
    width, height = image.size

    # Find duplicated centres
    duplicated_centres = [centre - original_width for centre in road_centre if centre >= original_width]
            
    # Drop the duplicated centres
    road_centre = [centre for centre in road_centre if centre not in duplicated_centres]

    # Calculate dimensions and offsets
    w4 = int(width / 4) # 
    h4 = int(height / 4)
    hFor43 = int(w4 * 3 / 4)
    w98 = width + (w4 / 2)
    xrapneeded = int(width * 7 / 8)

    pickles = []
    # Crop the panoramic image
    for centre in road_centre:
        # Wrapped all the way around
        if centre >= w98:
            xlo = int(centre - w4/2)
            cropped_image = image.crop((xlo, h4,  xlo+w4, h4 + hFor43))
            cropped_segmentation = segmentation[h4:h4+hFor43, xlo:xlo+w4]
        
        # Image requires assembly of two sides
        elif centre > xrapneeded:
            xlo = int(centre - (w4/2)) # horizontal_offset
            w4_p1 = width - xlo
            w4_p2 = w4 - w4_p1
            cropped_image_1 = image.crop((xlo, h4, xlo + w4_p1, h4 + hFor43))
            cropped_image_2 = image.crop((0, h4, w4_p2, h4 + hFor43))

            cropped_image = Image.new(image.mode, (w4, hFor43))
            cropped_image.paste(cropped_image_1, (0, 0))
            cropped_image.paste(cropped_image_2, (w4_p1, 0))

            cropped_segmentation_1 = segmentation[h4:h4+hFor43, xlo:xlo+w4_p1]
            cropped_segmentation_2 = segmentation[h4:h4+hFor43, 0:w4_p2]
            cropped_segmentation = torch.cat((cropped_segmentation_1, cropped_segmentation_2), dim=1)
        
        # Must paste together the two sides of the image
        elif centre < (w4 / 2):
            w4_p1 = int((w4 / 2) - centre)
            xhi = width - w4_p1
            w4_p2 = w4 - w4_p1

            cropped_image_1 = image.crop((xhi, h4, xhi + w4_p1, h4 + hFor43))
            cropped_image_2 = image.crop((0, h4, w4_p2, h4 + hFor43))

            cropped_image = Image.new(image.mode, (w4, hFor43))
            cropped_image.paste(cropped_image_1, (0, 0))
            cropped_image.paste(cropped_image_2, (w4_p1, 0))

            cropped_segmentation_1 = segmentation[h4:h4+hFor43, xhi:xhi+w4_p1]
            cropped_segmentation_2 = segmentation[h4:h4+hFor43, 0:w4_p2]
            cropped_segmentation = torch.cat((cropped_segmentation_1, cropped_segmentation_2), dim=1)
            
        # Straightforward crop
        else:
            xlo = int(centre - w4/2)
            cropped_image = image.crop((xlo, h4, xlo + w4, h4 + hFor43))
            cropped_segmentation = segmentation[h4:h4+hFor43, xlo:xlo+w4]

        pickles.append(cropped_segmentation)

    return pickles


def get_GVI(segmentations):
    green_percentage = 0
    for segment in segmentations:
        total_pixels = segment.numel()
        vegetation_pixels = (segment == 8).sum().item()
        green_percentage += vegetation_pixels / total_pixels
    
    return green_percentage / len(segmentations)


def process_images(image_url, is_panoramic, processor, model):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)

        if is_panoramic:
            # Get the size of the image
            width, height = image.size

            # Crop the bottom 20% of the image to cut the band on the bottom of the panoramic image
            bottom_crop = int(height * 0.2)
            image = image.crop((0, 0, width, height - bottom_crop))

        # Image segmentation
        segmentation = segment_images(image, processor, model)

        if is_panoramic:
            # Create a widened panorama by wrapping the first 25% of the image onto the right edge
            width, height = image.size
            w4 = int(0.25 * width)

            segmentation_25 = segmentation[:, :w4]
            # Concatenate the tensors along the first dimension (rows)
            segmentation_road = torch.cat((segmentation, segmentation_25), dim=1)
        else:
            segmentation_road = segmentation
        
        # Find roads to determine if the image is suitable for the analysis or not AND crop the panoramic images
        road_centre = find_road_centre(segmentation_road)

        if len(road_centre) > 0:
            if is_panoramic:
                pickles = crop_panoramic_images(width, image, segmentation_road, road_centre)
            else:
                pickles = [segmentation]
        
            # Now we can get the Green View Index
            GVI = get_GVI(pickles)
            return [GVI, is_panoramic, False, False]
        else:
            # There are not road centres, so the image is unusable
            return [None, None, True, False]
    except:
        return [None, None, True, True]
    

def download_image(id, geometry, image_id, is_panoramic, access_token, processor, model):
    if image_id:
        try:
            header = {'Authorization': 'OAuth {}'.format(access_token)}
        
            url = 'https://graph.mapillary.com/{}?fields=thumb_original_url'.format(image_id)
            response = requests.get(url, headers=header)
            data = response.json()
            image_url = data["thumb_original_url"]

            result = process_images(image_url, is_panoramic, processor, model)
        except:
            # There was an error during the downloading of the image
            result = [None, None, True, True]
    else:
        # The point doesn't have an image, then we set the missing value to true
        result = [None, None, True, False]
    
    result.insert(0, geometry)
    result.insert(0, id)

    return result


def download_images_for_points(gdf, access_token, max_workers=4):
    processor, model = get_models()
    
    # We will use this csv file to store the results every time a process finishes. If the script stops working for whatever reason, the results obtained until that point will be saved
    csv_file = f"gvi-points.csv"
    csv_path = os.path.join("results", csv_file)

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
        
    # Combine the results from all parts
    final_results = gpd.GeoDataFrame(results, columns=["id", "geometry", "GVI", "is_panoramic", "missing", "error"], crs=4326)
    
    return final_results


def get_gvi_per_buffer(buffered_points, gvi_per_point):
    joined = gpd.sjoin(gvi_per_point, buffered_points.set_geometry('buffer'), how='inner', predicate='within')

    # Group the points by buffer
    grouped = joined.groupby('index_right', group_keys=True)

    # Convert 'grouped' to a DataFrame
    grouped_df = grouped.apply(lambda x: x.reset_index(drop=True))
    grouped_df = grouped_df[["geometry_left", "GVI", "is_panoramic", "missing"]].reset_index()
    # Convert grouped_df to a GeoDataFrame
    grouped_gdf = gpd.GeoDataFrame(grouped_df, geometry='geometry_left')

    # Calculate the average 'gvi' for each group
    avg_gvi = grouped['GVI'].mean().reset_index()
    point_count = grouped['GVI'].count().reset_index(name='Point_Count')

    # Merge with the buffered_points dataframe to get the buffer geometries
    result = avg_gvi.merge(buffered_points, left_on='index_right', right_index=True)
    result = result.merge(point_count, on='index_right')
    # Convert the result to a GeoDataFrame
    result = gpd.GeoDataFrame(result[['geometry', 'GVI', 'Point_Count']])

    return result, grouped_gdf