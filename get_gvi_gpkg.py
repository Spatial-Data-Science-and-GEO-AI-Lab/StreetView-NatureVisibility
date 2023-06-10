import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import glob
import sys

""" 
This script converts CSV files generated with the main_script into a GeoPackage (gpkg) file and a GeoJSON file. It processes the CSV files for a specific city, performs data cleaning and validation, and saves the resulting files in the specified path.

The script just takes the city name as a command-line argument
"""

if __name__ == "__main__":
    args = sys.argv

    city = args[1] # City to analyse
    
    # Path to the CSV files
    csv_files = glob.glob(f"results/{city}/gvi/*.csv")

    # Create an empty list to store individual DataFrames
    dfs = []

    # Loop through the CSV files, read each file using pandas, and append the resulting DataFrame to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    # Concatenate all the DataFrames in the list along the rows
    merged_df = pd.concat(dfs, ignore_index=True)

    # Iterate over the DataFrame rows
    for index, row in merged_df.iterrows():
        try:
            # Attempt to convert the "x" and "y" values to floats
            float(row["x"])
            float(row["y"])
        except ValueError:
            # If a ValueError occurs, drop the row from the DataFrame
            merged_df.drop(index, inplace=True)

    # Drop duplicate rows based on the id column
    merged_df = merged_df.drop_duplicates(subset=['id'])

    merged_df.to_csv(f"results/{city}/gvi/gvi-points.csv", index=False)
    
    # Convert the 'geometry' column to valid Point objects
    merged_df['geometry'] = merged_df.apply(lambda row: Point(float(row["x"]), float(row["y"])), axis=1)
    merged_df["id"] = merged_df["id"].astype(int)

    # Convert the merged DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=4326)

    path_to_file="results/{}/gvi/gvi-points.gpkg".format(city, city)
    gdf.to_file(path_to_file, driver="GPKG", crs=4326)

    path_to_file="results/{}/gvi/gvi-points.geojson".format(city, city)
    gdf.to_file(path_to_file, driver="GeoJSON", crs=4326)
