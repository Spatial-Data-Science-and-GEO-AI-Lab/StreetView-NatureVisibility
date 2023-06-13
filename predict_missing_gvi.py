from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from pygam import LinearGAM, s 
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import sys
import os

# Function to calculate mean NDVI taken from Yúri Grings' GitHub repository
# https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/GreenEx_Py
from availability import get_mean_NDVI

def calculate_ndvi(gvi, ndvi, N, city, crs):
    ndvi_folder = os.path.join("results", city, "ndvi")
    
    mean_ndvi = get_mean_NDVI(  point_of_interest_file=gvi,
                                ndvi_raster_file = ndvi,
                                buffer_type="euclidean",
                                buffer_dist=N,
                                crs_epsg=crs,
                                write_to_file=False,
                                save_ndvi=False)

    # Save the calculated NDVI values to a file
    mean_ndvi.to_crs(crs=4326, inplace=True)
    path_to_file = os.path.join(ndvi_folder, "calculated_ndvi_values.gpkg")
    mean_ndvi.to_file(path_to_file, driver="GPKG", crs=4326)

    return path_to_file


def linear_regression(city):
    ndvi_folder = os.path.join("results", city, "ndvi")

    # Load ndvi layer
    ndvi_file = os.path.join(ndvi_folder, "calculated_ndvi_values.gpkg")
    ndvi_df = gpd.read_file(ndvi_file, layer="calculated_ndvi_values", crs=4326)

    # Separate data into known and missing GVI values
    known_df = ndvi_df[ndvi_df['missing'] == False].copy()
    missing_df = ndvi_df[ndvi_df['missing'] == True].copy()

    # Split known data into features (NDVI) and target (GVI)
    X_train = known_df[['mean_NDVI']]
    y_train = known_df['GVI']

    # Prepare missing data for prediction
    X_test = missing_df[['mean_NDVI']]

    # Perform linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    predicted_GVI = lin_reg.predict(X_test)

    # Assign the predicted values to the missing GVI values in the DataFrame
    missing_df['GVI'] = predicted_GVI

    # Concatenate the updated missing values with the known values
    updated_df = pd.concat([known_df, missing_df])

    path_to_file = os.path.join(ndvi_folder, "calculated_missing_values_linreg.gpkg")
    updated_df.to_file(path_to_file, driver="GPKG", crs=4326)

    # Compute RMSE using cross-validation
    rmse_scores = np.sqrt(-cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5))
    avg_rmse = np.mean(rmse_scores)

    print(f"RMSE for Linear Regression with cross-validation: {avg_rmse}")

    return avg_rmse, updated_df


def gam_regression(city):
    ndvi_folder = os.path.join("results", city, "ndvi")

    # Load ndvi layer
    ndvi_file = os.path.join(ndvi_folder, "calculated_ndvi_values.gpkg")
    ndvi_df = gpd.read_file(ndvi_file, layer="calculated_ndvi_values", crs=4326)

    # Separate data into known and missing GVI values
    known_df = ndvi_df[ndvi_df['missing'] == False].copy()
    missing_df = ndvi_df[ndvi_df['missing'] == True].copy()

    # Split known data into features (NDVI) and target (GVI)
    X_train = known_df[['mean_NDVI']]
    y_train = known_df['GVI']

    # Prepare missing data for prediction
    X_test = missing_df[['mean_NDVI']]

    n_features = 1 # number of features used in the model
    lams = np.logspace(-5, 5, 20) * n_features
    splines = 25

    # Train a Generalized Additive Model (GAM)
    gam = LinearGAM(
        s(0, n_splines=splines)).gridsearch(
            X_train.values,
            y_train.values,
            lam=lams
        )

    predicted_GVI = gam.predict(X_test.values)

    # Assign the predicted values to the missing GVI values in the DataFrame
    missing_df['GVI'] = predicted_GVI

    # Concatenate the updated missing values with the known values
    updated_df = pd.concat([known_df, missing_df])

    path_to_file= os.path.join(ndvi_folder, "calculated_missing_values_gam.gpkg")
    updated_df.to_file(path_to_file, driver="GPKG", crs=4326)

    # Compute RMSE using cross-validation
    rmse_scores = np.sqrt(-cross_val_score(gam, X_train, y_train, scoring='neg_mean_squared_error', cv=5))
    avg_rmse = np.mean(rmse_scores)

    print(f"RMSE for GAM with cross-validation: {avg_rmse}")

    return avg_rmse, updated_df


def clean_points(city, crs):
    # Cleans the GVI points data by dropping points outside the extent of the NDVI file.

    # File paths for the GVI points and NDVI files
    # The NDVI file has to be stored in results/city/ndvi folder and has to be named ndvi.tif
    gvi = os.path.join("results", city, "gvi", "gvi-points.gpkg")
    ndvi = os.path.join("results", city, "ndvi", f"ndvi.tif")

    gvi_df = gpd.read_file(gvi, layer="gvi-points", crs=4326)
    gvi_df.to_crs(epsg=crs, inplace=True)

    # Get the extent of the NDVI file
    with rasterio.open(ndvi) as src:
        extent = src.bounds

        # Filter the GVI points to include only those within the extent of the NDVI file
        filtered_gvi = gvi_df.cx[extent[0]:extent[2], extent[1]:extent[3]]
    
    # Save the filtered GVI points to a new file to preserve the original data
    filtered_gvi_path = os.path.join("results", city, "ndvi", "filtered-points.gpkg")
    filtered_gvi.to_file(filtered_gvi_path, driver="GPKG", crs=crs)

    return filtered_gvi_path, ndvi


if __name__ == "__main__":
    # Read command-line arguments
    args = sys.argv

    # Extract city, CRS, and distance from the command-line arguments
    city = args[1] # City to analyze
    crs = int(args[2]) # CRS in meters, suitable for the area in which we are working
    # For example, we can use the same CRS as the roads.gpkg file
    # IMPORTANT: The NDVI image should be in this CRS
    distance = int(args[3]) # The distance used to generate the sample points
    
    # Step 1: Clean the GVI points by filtering points outside the extent of the NDVI file
    gvi, ndvi = clean_points(city, crs)

    # Step 2: Calculate the mean NDVI values from the filtered GVI points
    ndvi_path = calculate_ndvi(gvi, ndvi, distance//2, city, crs)

    # Step 3: Train a Linear Regression model to predict missing GVI values
    linreg_rmse, linreg = linear_regression(city)