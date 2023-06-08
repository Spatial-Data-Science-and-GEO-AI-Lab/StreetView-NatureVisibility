import os
os.environ['USE_PYGEOS'] = '0'

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.signal import find_peaks
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import csv


from PIL import Image, ImageFile
import numpy as np
import requests

ImageFile.LOAD_TRUNCATED_IMAGES = True

def prepare_folders(city):
    # Create folder for storing GVI results, sample points and road network if they don't exist yet
    dir_path = os.path.join("results", city, "gvi")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    dir_path = os.path.join("results", city, "points")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join("results", city, "roads")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    

def get_models():
    # Load the pretrained AutoImageProcessor from the "facebook/mask2former-swin-large-cityscapes-semantic" model
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pretrained Mask2FormerForUniversalSegmentation model from "facebook/mask2former-swin-large-cityscapes-semantic"
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    # Move the model to the specified device (GPU or CPU)
    model = model.to(device)
    # Return the processor and model as a tuple
    return processor, model


def segment_images(image, processor, model):
    # Preprocess the image using the image processor
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform a forward pass through the model to obtain the segmentation
    with torch.no_grad():
        # Check if a GPU is available
        if torch.cuda.is_available():
            # Move the inputs to the GPU
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            # Perform the forward pass through the model
            outputs = model(**inputs)
            # Post-process the semantic segmentation outputs using the processor and move the results to CPU
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0].to('cpu')
        else:
            # Perform the forward pass through the model
            outputs = model(**inputs)
            # Post-process the semantic segmentation outputs using the processor
            segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            
    return segmentation


# Based on Matthew Danish code (https://github.com/mrd/vsvi_filter/tree/master)
def run_length_encoding(in_array):
    # Convert input array to a NumPy array
    image_array = np.asarray(in_array)
    length = len(image_array)
    if length == 0: 
        # Return None values if the array is empty
        return (None, None, None)
    else:
        # Calculate run lengths and change points in the array
        pairwise_unequal = image_array[1:] != image_array[:-1]
        change_points = np.append(np.where(pairwise_unequal), length - 1)   # must include last element posi
        run_lengths = np.diff(np.append(-1, change_points))       # run lengths
        return(run_lengths, image_array[change_points])

def get_road_pixels_per_column(prediction):
    # Check which pixels in the prediction array correspond to roads (label 0)
    road_pixels = prediction == 0.0
    road_pixels_per_col = np.zeros(road_pixels.shape[1])
    
    for i in range(road_pixels.shape[1]):
        # Encode the road pixels in each column and calculate the maximum run length
        run_lengths, values = run_length_encoding(road_pixels[:,i])
        road_pixels_per_col[i] = run_lengths[values.nonzero()].max(initial=0)
    return road_pixels_per_col

def get_road_centres(prediction, distance=2000, prominence=100):
    # Get the road pixels per column in the prediction
    road_pixels_per_col = get_road_pixels_per_column(prediction)

    # Find peaks in the road_pixels_per_col array based on distance and prominence criteria
    peaks, _ = find_peaks(road_pixels_per_col, distance=distance, prominence=prominence)
    
    return peaks


def find_road_centre(segmentation):
    # Calculate distance and prominence thresholds based on the segmentation shape
	distance = int(2000 * segmentation.shape[1] // 5760)
	prominence = int(100 * segmentation.shape[0] // 2880)
	
    # Find road centers based on the segmentation, distance, and prominence thresholds
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

    images = []
    pickles = []

    # Crop the panoramic image based on road centers
    for centre in road_centre:
        # Wrapped all the way around
        if centre >= w98:
            xlo = int((width - centre) - w4/2)
            cropped_image = image.crop((xlo, h4, xlo + w4, h4 + hFor43))
            cropped_segmentation = segmentation[h4:h4+hFor43, xlo:xlo+w4]
        
        # Image requires assembly of two sides
        elif centre > xrapneeded:
            xlo = int(centre - (w4/2)) # horizontal_offset
            w4_p1 = width - xlo
            w4_p2 = w4 - w4_p1

            # Crop and concatenate image and segmentation
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

            # Crop and concatenate image and segmentation
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
        
        images.append(cropped_image)
        pickles.append(cropped_segmentation)

    return images, pickles


def get_GVI(segmentations):
    green_percentage = 0
    for segment in segmentations:
        # Calculate the total number of pixels in the segmentation
        total_pixels = segment.numel()
        # Filter the pixels that represent vegetation (label 8) and count them
        vegetation_pixels = (segment == 8).sum().item()
        # Calculate the percentage of green pixels in the segmentation
        green_percentage += vegetation_pixels / total_pixels
    
    # Calculate the average green percentage across all segmentations
    return green_percentage / len(segmentations)


def process_images(image_url, is_panoramic, processor, model):
    try:
        # Fetch and process the image
        image = Image.open(requests.get(image_url, stream=True).raw)

        if is_panoramic:
            # Get the size of the image
            width, height = image.size

            # Crop the bottom 20% of the image to remove the band at the bottom of the panoramic image
            bottom_crop = int(height * 0.2)
            image = image.crop((0, 0, width, height - bottom_crop))

        # Apply the semantic segmentation to the image
        segmentation = segment_images(image, processor, model)

        if is_panoramic:
            # Create a widened panorama by wrapping the first 25% of the image onto the right edge
            width, height = image.size
            w4 = int(0.25 * width)

            segmentation_25 = segmentation[:, :w4]
            # Concatenate the tensors along the first dimension (rows) to create the widened panorama with the segmentations
            segmentation_road = torch.cat((segmentation, segmentation_25), dim=1)
        else:
            # If the image is not panoramic, use the segmentation as it is
            segmentation_road = segmentation
        
        # Find the road centers to determine if the image is suitable for analysis
        road_centre = find_road_centre(segmentation_road)

        if len(road_centre) > 0:
            # The image is suitable for analysis
            if is_panoramic:
                # If it's panoramic, crop the image and its segmentation based on the previously found road centers
                images, pickles = crop_panoramic_images(width, image, segmentation_road, road_centre)
            else:
                # If it's not panoramic, use the segmentation without any modification
                images = [image]
                pickles = [segmentation]
        
            # Calculate the Green View Index (GVI) for the cropped segmentations
            GVI = get_GVI(pickles)
            return images, pickles, [GVI, is_panoramic, False, False]
        else:
            # There are no road centers, so the image is not suitable for analysis
            return [image], [segmentation], [None, None, True, False]
    except:
        # If there was an error while processing the image, set the "error" flag to true and continue with other images
        return None, None, [None, None, True, True]


# Download images
def download_image(id, geometry, image_id, is_panoramic, access_token, processor, model):
    # Check if the image id exists
    if image_id:
        try:
            # Create the authorization header for the Mapillary API request
            header = {'Authorization': 'OAuth {}'.format(access_token)}

            # Build the URL to fetch the image thumbnail's original URL
            url = 'https://graph.mapillary.com/{}?fields=thumb_original_url'.format(image_id)
            
            # Send a GET request to the Mapillary API to obtain the image URL
            response = requests.get(url, headers=header)
            data = response.json()
            
            # Extract the image URL from the response data
            image_url = data["thumb_original_url"]

            # Process the downloaded image using the provided image URL, is_panoramic flag, processor, and model
            _, _, result = process_images(image_url, is_panoramic, processor, model)
        except:
            # An error occurred during the downloading of the image
            result = [None, None, True, True]
    else:
        # The point doesn't have an associated image, so we set the missing value flags
        result = [None, None, True, False]

    # Insert the coordinates (x and y) and the point ID at the beginning of the result list
    # This helps us associate the values in the result list with their corresponding point
    result.insert(0, geometry.y)
    result.insert(0, geometry.x)
    result.insert(0, id)

    return result


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