import os
os.environ['USE_PYGEOS'] = '0'

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.signal import find_peaks
import torch

from PIL import Image
import numpy as np
import requests



def prepare_folders(city, path):
    dir_path = os.path.join(path, "results", city, "gvi")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    dir_path = os.path.join(path, "results", city, "points")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join(path, "results", city, "roads")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    

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


# Download images
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

    result.insert(0, geometry.y)
    result.insert(0, geometry.x)
    result.insert(0, id)

    return result