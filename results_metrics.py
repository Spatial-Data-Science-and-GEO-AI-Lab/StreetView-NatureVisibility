import geopandas as gpd
import numpy as np
import pandas as pd
import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt


def plot_unavailable_images(df, city):
    grouped = df.groupby(
        ["highway", "city"]
        ).agg(
            {"total_null": "sum", 
             "proportion_null": "sum"})
    
    grouped2 = grouped.groupby("highway").agg({"total_null": "sum"})

    # Sort the grouped DataFrame by 'total_null' column in descending order and select the top 5 rows
    top_5_highways = list(grouped2.nlargest(5, 'total_null').index)

    grouped = grouped.loc[top_5_highways]

    # Reset the index for proper sorting and grouping
    grouped = grouped.reset_index()
    
    grouped = grouped.sort_values(by="proportion_null", ascending=False)
    
    custom_palette = ["#D53E4F", "#FC8D59", "#FEE08B", "#FFFFBF", "#E6F598", "#99D594", "#3288BD"]

    # Create a bar plot for the top 5 highway types
    bar1 = sns.barplot(data=grouped, x="proportion_null", y="highway", hue="city", palette=custom_palette)

    # Create custom legend handles and labels for bar1
    handles1, labels1 = bar1.get_legend_handles_labels()
    # Add the legends outside of the plot
    plt.legend(handles1, labels1, title='City', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set labels and title
    plt.title('Top 5 Highway Types with Most Missing Images')
    
    plt.xlabel('Highway Types')
    plt.ylabel('Proportion of Missing Images by Highway Type')
    
    # Set the maximum value for the y-axis
    plt.xlim(0, 1)

    plt.show()
    
    # Set the figure size
    bar1.figure.set_size_inches(8, 6)  # Adjust the width and height as needed

    bar1.figure.savefig(f'results/{city}/plot_missing_images_{city}.svg', format='svg', bbox_inches='tight')

    return grouped


def get_unavailable_images(intersection, city):
    grouped = intersection.groupby(['road_index_x', 'highway']).agg({
        'image_id': lambda x: (~x.isnull()).sum(), # Count number of points with missing images
    }).reset_index()

    # Rename the columns of the grouped dataframe
    grouped.columns = ['road_index', 'highway', 'total_null']
    
    # Count the number of missing values per road type
    count = grouped.groupby('highway').agg({
        'total_null': 'sum',
    }).sort_values('total_null', ascending=False)

    count['city'] = city
    count['proportion_null'] = count["total_null"] / len(intersection)
    
    return count


def get_road_unavailable_images(city):
    dir_path = os.path.join("results", city)

    # Load roads layer
    roads_path = os.path.join(dir_path, "gvi", "gvi-streets.gpkg")
    roads = gpd.read_file(roads_path, layer="gvi-streets")

    # Load points with gvi layer
    points_path = os.path.join(dir_path, "gvi", "gvi-points.gpkg")
    points = gpd.read_file(points_path, layer="gvi-points", crs=4326)
    points.to_crs(crs=roads.crs, inplace=True)

    # Load points with roads layer
    points_road_path = os.path.join(dir_path, "points", "points.gpkg")
    points_road = gpd.read_file(points_road_path, layer="points", crs=4326)
    points_road.to_crs(crs=roads.crs, inplace=True)

    points_road = points_road.merge(points, on="id")

    # Merge the previous dataframe with the roads dataframe
    intersection = points_road.merge(roads, left_on="road_index", right_on="index")

    intersection = intersection[["id", "image_id", "distance", "is_panoramic_x", "road_index_x", "geometry_x", "GVI", "length", "highway"]]

    count = get_unavailable_images(intersection, city)

    return intersection, count


def get_missing_images(df):
    unavailable = df[df["image_id"] == ""].count()[0]
    unsuitable = df[(df["GVI"].isnull()) & (df["image_id"]!="")].count()[0]
    total_null = df[df["GVI"].isnull()].count()[0]
    total = df.count()[0]
    percentage_null = total_null / total

    result_table = [unavailable, unsuitable, total_null, percentage_null, total]
    return  pd.DataFrame([result_table], columns=['Unavailable', 'Unsuitable', 'Total', 'Proportion', 'Total Sample Points'])


# Create an empty DataFrame to store the results
def get_panoramic_images(df):
    is_panoramic = df[df["is_panoramic_x"]].count()[0]
    total = df[df["image_id"] != ""].count()[0]

    result_table = [is_panoramic, total, is_panoramic/total]
    return pd.DataFrame([result_table], columns=['Panoramic Images', 'Total Images', "Proportion"])


def get_availability_score(df):
    gvi_points = df[df["image_id"]!=""].count()[0]
    road_length = df["length"].sum() / 1000
    total = df.count()[0]

    result_table = [gvi_points, road_length, total, gvi_points/total, (gvi_points  * np.log(road_length))/total]
    return pd.DataFrame([result_table], columns=['GVI Points', 'Road Length', 'Total Sample', 'Availability Score', 'Adjusted Availability Score'])


def get_usability_score(df):
    gvi_points = df[(~df["GVI"].isnull()) & (df["image_id"]!="")].count()[0]
    road_length = df["length"].sum() / 1000
    total = df[df["image_id"]!=""].count()[0]

    result_table = [gvi_points, road_length, total, gvi_points/total, (gvi_points  * np.log(road_length))/total]

    return pd.DataFrame([result_table], columns=['GVI Points', 'Road Length', 'Total Sample', 'Usability Score', 'Adjusted Usability Score'])


def get_metrics(city):
    intersection, count = get_road_unavailable_images(city)
    
    print(f"Unavailable images per road type for {city}")
    print(plot_unavailable_images(count, city))
    
    print(f"\nMissing images for {city}")
    print(get_missing_images(intersection))

    print(f"\nPanoramic images for {city}")
    print(get_panoramic_images(intersection))

    print(f"\nImage Availability Score and Adjusted Image Availability Score for {city}")
    print(get_availability_score(intersection))

    print(f"\nImage Usability Score and Ajdusted Image Usability Score  for {city}")
    print(get_usability_score(intersection))


if __name__ == "__main__":
    # Read command-line arguments
    args = sys.argv

    # Extract city, CRS, and distance from the command-line arguments
    city = args[1] # City to analyze
    
    get_metrics(city)
    
    

