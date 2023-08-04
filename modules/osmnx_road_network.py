# Libraries for working with maps and geospatial data
from vt2geojson.tools import vt_bytes_to_geojson
from shapely.geometry import Point
from scipy.spatial import cKDTree
import geopandas as gpd
import osmnx as ox
import mercantile

# Libraries for working with concurrency and file manipulation
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests

def get_road_network(city):
    # Get the road network graph using OpenStreetMap data
    # 'network_type' argument is set to 'drive' to get the road network suitable for driving
    # 'simplify' argument is set to 'True' to simplify the road network
    G = ox.graph_from_place(city, network_type="drive", simplify=True)

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
    
    # Project the graph from latitude-longitude coordinates to a local projection (in meters)
    G_proj = ox.project_graph(G)

    # Convert the projected graph to a GeoDataFrame
    # This function projects the graph to the UTM CRS for the UTM zone in which the graph's centroid lies
    _, edges = ox.graph_to_gdfs(G_proj) 

    return edges


# Get a list of points over the road map with a N distance between them
def select_points_on_road_network(roads, N=50):
    points = []
    # Iterate over each road
    
    for row in roads.itertuples(index=True, name='Road'):
        # Get the LineString object from the geometry
        linestring = row.geometry
        index = row.Index

        # Calculate the distance along the linestring and create points every 50 meters
        for distance in range(0, int(linestring.length), N):
            # Get the point on the road at the current position
            point = linestring.interpolate(distance)

            # Add the curent point to the list of points
            points.append([point, index])
    
    # Convert the list of points to a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(points, columns=["geometry", "road_index"], geometry="geometry")

    # Set the same CRS as the road dataframes for the points dataframe
    gdf_points.set_crs(roads.crs, inplace=True)

    # Drop duplicate rows based on the geometry column
    gdf_points = gdf_points.drop_duplicates(subset=['geometry'])
    gdf_points = gdf_points.reset_index(drop=True)

    return gdf_points


# This function extracts the features for a given tile
def get_features_for_tile(tile, access_token):
    # This URL retrieves all the features within the tile. These features are then going to be assigned to each sample point depending on the distance.
    tile_url = f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}"
    response = requests.get(tile_url)
    result = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z, layer="image")
    return [tile, result]


def get_features_on_points(points, access_token, max_distance=50, zoom=14):
    # Store the local crs in meters that was assigned by osmnx previously so we can use it to calculate the distances between features and points
    local_crs = points.crs

    # Set the CRS to 4326 because it is used by Mapillary
    points.to_crs(crs=4326, inplace=True)
    
    # Add a new column to gdf_points that contains the tile coordinates for each point
    points["tile"] = [mercantile.tile(x, y, zoom) for x, y in zip(points.geometry.x, points.geometry.y)]

    # Group the points by their corresponding tiles
    groups = points.groupby("tile")

    # Download the tiles and extract the features for each group
    features = []
    
    # To make the process faster the tiles are downloaded using threads
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

    # Transform from EPSG:4326 (world °) to the local crs in meters that we got when we projected the roads graph in the previous step
    feature_points.to_crs(local_crs, inplace=True)
    points.to_crs(local_crs, inplace=True)

    # Create a KDTree (k-dimensional tree) from the "geometry" coordinates of feature_points
    feature_tree = cKDTree(feature_points["geometry"].apply(lambda p: [p.x, p.y]).tolist())
    # Use the KDTree to query the nearest neighbors of the points in the "geometry" column of points DataFrame
    # The query returns the distances and indices of the nearest neighbors
    # The parameter "k=1" specifies that we want to find the nearest neighbor
    # The parameter "distance_upper_bound=max_distance" sets a maximum distance for the nearest neighbors
    distances, indices = feature_tree.query(points["geometry"].apply(lambda p: [p.x, p.y]).tolist(), k=1, distance_upper_bound=max_distance/2)

    # Create a list to store the closest features and distances to each point. If there are no images close then set the value of both to None
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
    points = points.reset_index(drop=True)

    # Transform the coordinate reference system to EPSG 4326
    points.to_crs(epsg=4326, inplace=True)
    
    return points