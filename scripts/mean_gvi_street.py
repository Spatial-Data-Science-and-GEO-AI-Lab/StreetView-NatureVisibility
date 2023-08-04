import geopandas as gpd
import sys
import os

"""
The main purpose of the script is to compute statistical measures such as the mean GVI, the number of missing points, and the total number of points per road segment. These statistics provide valuable insights into the visibility and quality of roads within a given city.

To use the script, the user needs to provide the name of the city to be analyzed as a command-line argument. The script then retrieves the necessary data files from the corresponding directories and performs the statistical calculations.
"""


if __name__ == "__main__":
    args = sys.argv

    city = args[1] # City to analyse

    dir_path = os.path.join("results", city)

    # Load roads layer
    roads_path = os.path.join(dir_path, "roads", "roads.gpkg")
    roads = gpd.read_file(roads_path, layer="roads")

    # Load points with gvi layer
    points_path = os.path.join(dir_path, "ndvi", "calculated_missing_values_linreg.gpkg")
    points = gpd.read_file(points_path, layer="calculated_missing_values_linreg", crs=4326)
    points.to_crs(crs=roads.crs, inplace=True)

    # Load points with roads layer
    points_road_path = os.path.join(dir_path, "points", "points.gpkg")
    points_road = gpd.read_file(points_road_path, layer="points", crs=4326)
    points_road.to_crs(crs=roads.crs, inplace=True)

    # Merge the dataframe containing the GVI value with the dataframe containing the roads ids
    points_road = points.merge(points_road, on="id")

    # Merge the previous dataframe with the roads dataframe
    intersection = points_road.merge(roads, left_on="road_index", right_on="index")

    # Get statistics per road (mean GVI value, number of null points, number of total points)
    gvi_per_road = intersection.groupby("road_index").agg(
        {'GVI': ['mean', lambda x: x.isnull().sum(), 'size']}
    ).reset_index()

    gvi_per_road.columns = ['road_index', 'avg_GVI', 'null_points_count', 'total_points']

    # Merge the results back into the road layer
    roads_with_avg_gvi = roads.merge(gvi_per_road, left_on="index", right_on="road_index", how='left')

    # Save results to GPKG
    path_to_file="results/{}/gvi/gvi-streets.gpkg".format(city)
    roads_with_avg_gvi.to_file(path_to_file, driver="GPKG", crs=roads.crs)
