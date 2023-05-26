from process_data import get_models, segment_images, find_road_centre, crop_panoramic_images, get_GVI

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from PIL import Image
import subprocess
import torch
import time
import csv
import sys
import os

processor, model = get_models()

def process_images(image_path, processor, model):
    try:
        image_name = os.path.basename(image_path)

        # Remove the file extension from the image name
        image_name = os.path.splitext(image_name)[0]
        
        # Split the file name into two parts: image ID and ispanoramic value
        image_id, ispanoramic = image_name.split("_")

        # Remove the file extension (.jpg) from the ispanoramic value
        is_panoramic = bool(ispanoramic.split(".")[0])
        
        image = Image.open(image_path)

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
                _, pickles = crop_panoramic_images(width, image, segmentation_road, road_centre)
            else:
                images = [image]
                pickles = [segmentation]
        
            # Now we can get the Green View Index
            GVI = get_GVI(pickles)
            return True, [image_id, GVI, is_panoramic, False, False]
        else:
            # There are not road centres, so the image is unusable
            return True, [image_id, None, None, True, False]
    except:
        return False, [image_id, None, None, True, True]


def update_csv(newrow):
    csv_file = 'gvi-points.csv'
    dir_path = os.path.join(path, "results", city, "gvi")
    csv_path = os.path.join(dir_path, csv_file)

    # Check if the CSV file exists
    file_exists = os.path.exists(csv_path)

    # Open the CSV file in append mode with newline=''
    with open(csv_path, 'a', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(["Image ID", "GVI", "is_panoramic", "missing", "error"])

        # Write the new row to the CSV file
        writer.writerow(newrow)


class FileEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        image_path = event.src_path

        print("New file created:", image_path)

        # Add a delay of 2 seconds to allow for the file to be fully downloaded
        time.sleep(2)

        retries = 0
        success = False

        while retries < 5 and not success:
            success, newrow = process_images(image_path, processor, model)
            
            if not success:
                retries += 1
                time.sleep(2 ** (retries + 1))
        
        update_csv(newrow)
        

if __name__ == "__main__":
    args = sys.argv
    city = args[1]
    path = args[2] if len(args) > 2 else ""
    folder_to_watch = os.path.join(path, "results", city, "images")
    event_handler = FileEventHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=True)
    observer.start()

    dir_path = os.path.join(path, "results", city, "gvi")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
