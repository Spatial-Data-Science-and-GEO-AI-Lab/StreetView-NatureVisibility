# Libraries for working with maps and geospatial data
from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point, MultiPoint
from scipy.spatial import cKDTree
import geopandas as gpd
from geopandas.tools import sjoin
import osmnx as ox
import mercantile

# Libraries for working with concurrency and file manipulation
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import requests

def get_road_network(city):
    # Get the road network graph using OpenStreetMap data
    # 'network_type' argument is set to 'drive' to get the road network suitable for driving
    # 'simplify' argument is set to 'True' to simplify the road network
    G = ox.graph_from_place(city, network_type="drive", simplify=True)

    # Project the graph from latitude-longitude coordinates to a local projection (in meters)
    G_proj = ox.project_graph(G)

    # Convert the projected graph to a GeoDataFrame
    _, edges = ox.graph_to_gdfs(G_proj)    

    return edges


# Get a list of points over the road map with a N distance between them
def select_points_on_road_network(roads, distance=50):
    # Initialize a list to store the points
    points = []

    # Loop through each road in the road network graph
    for road in roads.geometry:
        # Calculate the total length of the road
        road_length = road.length

        # Start at the beginning of the road
        current_position = 0

        # Loop through the road, adding points every 50 meters
        while current_position < road_length:
            # Get the point on the road at the current position
            current_point = road.interpolate(current_position)

            # Add the curent point to the list of points
            points.append(current_point)

            # Increment the position by the desired distance
            current_position += distance
    
    # Convert the list of points to a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(geometry=points)

    # Set the same CRS as the road dataframes for the points dataframe
    gdf_points.set_crs(roads.crs, inplace=True)

    # Drop duplicate rows based on the geometry column
    gdf_points = gdf_points.drop_duplicates(subset=['geometry'])

    return gdf_points


# This function extracts the features for a given tile
def get_features_for_tile(tile, access_token):
    tile_url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}"
    response = requests.get(tile_url)
    result = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer="image")
    return [tile, result]


def get_features_on_points(points, access_token, zoom=14):
    # Transform the coordinate reference system to EPSG 4326
    points.to_crs(epsg=4326, inplace=True)

    # Add a new column to gdf_points that contains the tile coordinates for each point
    points['tile'] = [mercantile.tile(x, y, zoom) for x, y in zip(points.geometry.x, points.geometry.y)]

    # Group the points by their corresponding tiles
    groups = points.groupby('tile')

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
    feature_points = pd.DataFrame(
        [(Point(f["geometry"]["coordinates"]), f) for row in pd_features["features"] for f in row["features"]],
        columns=["geometry", "feature"]
    )
    feature_tree = cKDTree(feature_points["geometry"].apply(lambda p: [p.x, p.y]).tolist())
    _, indices = feature_tree.query(points["geometry"].apply(lambda p: [p.x, p.y]).tolist())

    # Select the closest feature for each point
    points["feature"] = feature_points.loc[indices, "feature"].tolist()

    # Convert results to geodataframe
    points['tile'] = points['tile'].astype(str)
    
    return points