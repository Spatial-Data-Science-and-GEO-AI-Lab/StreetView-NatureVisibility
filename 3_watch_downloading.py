from process_data import get_models, segment_images, find_road_centre, crop_panoramic_images, get_GVI
from concurrent.futures import ThreadPoolExecutor, as_completed
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from threading import Thread
from queue import Queue
from PIL import Image
from tqdm import tqdm
import subprocess
import pickle
import torch
import time
import csv
import sys
import os

processor, model = get_models()

def prepare_folders(city, path=""):
    dir_path = os.path.join(path, "results", city, "gvi")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join(path, "results", city, "images")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    dir_path = os.path.join(path, "results", city, "pickles")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.join(path, "results", city, "final_pickles")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_files(image_id, segmentation, pickles, city, path):
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


def process_images(image_path, processor, model, city, path):
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
            
            save_files(image_id, segmentation, pickles, city, path)
        
            # Now we can get the Green View Index
            GVI = get_GVI(pickles)
            return True, [image_id, GVI, is_panoramic, False, False]
        else:
            # There are not road centres, so the image is unusable
            return True, [image_id, None, None, True, False]
    except Exception as e:
        print(e)
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


def process_image(image_path, city, path):
    # Add a delay of 1 second to allow for the file to be fully downloaded
    time.sleep(1)

    retries = 0
    success = False

    while retries < 5 and not success:
        success, newrow = process_images(image_path, processor, model, city, path)
            
        if not success:
            retries += 1
            time.sleep(2 ** (retries + 1))
        
    update_csv(newrow)
    os.remove(image_path)
    return image_path


class FileEventHandler(FileSystemEventHandler):
    def __init__(self):
        self.queue = Queue()

    def on_created(self, event):
        image_path = event.src_path
        self.queue.put(image_path)


def image_processing_worker(queue, city, path):
    while True:
        image_path = queue.get()
        print("New file created:", image_path)
        process_image(image_path, city, path)
        print("Finished file:", image_path)
        queue.task_done()
        

if __name__ == "__main__":
    args = sys.argv
    download_images = bool(int(args[1]))
    city = args[2]
    path = args[3] if len(args) > 3 else ""
    num_threads = int(args[4]) if len(args) > 4 else 5

    prepare_folders(city, path)
    
    if download_images:
        folder_to_watch = os.path.join(path, "results", city, "images")
        event_handler = FileEventHandler()
        observer = Observer()
        observer.schedule(event_handler, folder_to_watch, recursive=True)
        observer.start()

        # Start the image processing worker thread
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for _ in range(num_threads):
                worker_thread = Thread(target=image_processing_worker, args=(event_handler.queue, city, path))
                worker_thread.daemon = True
                worker_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        event_handler.queue.join()  # Wait for the queue to be fully processed
    
    else:
        folder_path = os.path.join(path, "results", city, "images")

        file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name)) and file_name != ".DS_Store"]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for filepath in file_paths:
                future = executor.submit(process_image, filepath, city, path)
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing images: "):
                result = future.result() 
                print("Finished file:", result)
