import os
os.environ['USE_PYGEOS'] = '0'

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image, ImageDraw
import torch

from tqdm import tqdm
import numpy as np
import requests

import json

from scipy.signal import find_peaks
import pickle

# color palette to map each class to a RGB value
color_palette = [
    [128, 64, 128],  # 0: road - maroon
    [244, 35, 232],  # 1: sidewalk - pink
    [70, 70, 70],  # 2: building - dark gray
    [102, 102, 156],  # 3: wall - purple
    [190, 153, 153],  # 4: fence - light brown
    [153, 153, 153],  # 5: pole - gray
    [250, 170, 30],  # 6: traffic light - orange
    [220, 220, 0],  # 7: traffic sign - yellow
    [107, 142, 35],  # 8: vegetation - dark green
    [152, 251, 152],  # 9: terrain - light green
    [70, 130, 180],  # 10: sky - blue
    [220, 20, 60],  # 11: person - red
    [255, 0, 0],  # 12: rider - bright red
    [0, 0, 142],  # 13: car - dark blue
    [0, 0, 70],  # 14: truck - navy blue
    [0, 60, 100],  # 15: bus - dark teal
    [0, 80, 100],  # 16: train - dark green
    [0, 0, 230],  # 17: motorcycle - blue
    [119, 11, 32]  # 18: bicycle - dark red
]


def prepare_folders(path, city):
    dir_path = os.path.join(path, "results", city, "images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    dir_path = os.path.join(path, "results", city, "final_images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    dir_path = os.path.join(path, "results", city, "segments")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    dir_path = os.path.join(path, "results", city, "pickles")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join(path, "results", city, "final_pickles")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_models():
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
    return processor, model


def segment_images(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # You can pass them to processor for postprocessing
    segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    return segmentation


def save_files(image_id, image, segmentation, images, pickles, city, path=""):
    # Save original image 
    dir_path = os.path.join(path, "results", city, "images")
    img_path = os.path.join(dir_path, "{}.jpg".format(image_id))
    image.save(img_path)

    color_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(color_palette)
    for label, color in enumerate(palette):
        color_seg[segmentation == label, :] = color
    
    # Show image + mask
    img = np.array(image) * 0.4 + color_seg * 0.6
    img = img.astype(np.uint8)

    # Save final images
    dir_path = os.path.join(path, "results", city, "final_images")
    for index, image in enumerate(images):
        img_path = os.path.join(dir_path, "{}_{}.jpg".format(image_id, index))
        image.save(img_path)
    
    # Convert numpy array to PIL Image and save masked image
    pil_img = Image.fromarray(img)
    dir_path = os.path.join(path, "results", city, "segments")
    img_path = os.path.join(dir_path, "{}.png".format(image_id))
    pil_img.save(img_path)

    # Save segmentation array as a pickle file
    dir_path = os.path.join(path, "results", city, "pickles")
    pickle_path = os.path.join(dir_path, "{}.pkl".format(image_id))
    with open(pickle_path, 'wb') as f:
        pickle.dump(segmentation, f)
    
    # Save final segmentation arrays as a pickle file
    dir_path = os.path.join(path, "results", city, "final_pickles")
    for index, pick in enumerate(pickles):
        pickle_path = os.path.join(dir_path, "{}_{}.pkl".format(image_id, index))
        with open(pickle_path, 'wb') as f:
            pickle.dump(pick, f)
    
    return pickle_path


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

    images = []
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
        
        images.append(cropped_image)
        pickles.append(cropped_segmentation)

    return images, pickles


def get_GVI(segmentations):
    green_percentage = 0
    for segment in segmentations:
        total_pixels = segment.numel()
        vegetation_pixels = (segment == 8).sum().item()
        green_percentage += vegetation_pixels / total_pixels
    
    return green_percentage / len(segmentations)


def process_images(image_id, image_url, is_panoramic, processor, model, city, path):
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
                images, pickles = crop_panoramic_images(width, image, segmentation_road, road_centre)
            else:
                images = [image]
                pickles = [segmentation]
        
            # Now we can get the Green View Index
            GVI = get_GVI(pickles)

            if path: save_files(image_id, image, segmentation, images, pickles, city, path)
            return [GVI, is_panoramic, False, False]
        else:
            if path: save_files(image_id, image, segmentation, [], [], city, path)
        
            # There are not road centres, so the image is unusable
            return [0, None, True, False]
    except:
        return [0, None, True, True]


# Download images
def download_image(geometry, image_metadata, city, access_token, processor, model, path):
    if path: prepare_folders(path, city)
    header = {'Authorization': 'OAuth {}'.format(access_token)}

    image_id = image_metadata["properties"]["id"]
    is_panoramic = image_metadata["properties"]["is_pano"]
    
    url = 'https://graph.mapillary.com/{}?fields=thumb_original_url'.format(image_id)
    response = requests.get(url, headers=header)
    data = response.json()
    image_url = data["thumb_original_url"]

    result = process_images(image_id, image_url, is_panoramic, processor, model, city, path)
    result.insert(0, geometry)

    return result


def process_data(index, data_part, processor, model, city, access_token, path):
    results = []
    max_workers = 5 # We can adjutst this value
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in data_part.iterrows():
            feature = row["feature"]
            geometry = row["geometry"]
            feature = json.loads(feature)
            futures.append(executor.submit(download_image, geometry, feature, city, access_token, processor, model, path))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading images (Process {index})"):
            image_result = future.result()
            results.append(image_result)
    
    return results