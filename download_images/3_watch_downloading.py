from process_data import prepare_folders, get_models, process_images
from concurrent.futures import ThreadPoolExecutor
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from threading import Thread
from queue import Queue
import time
import csv
import sys
import os

processor, model = get_models()

def update_csv(newrow):
    csv_file = 'gvi-points.csv'
    dir_path = os.path.join(path, "results", city, "gvi")
    csv_path = os.path.join(dir_path, csv_file)

    # Check if the CSV file exists
    file_exists = os.path.exists(csv_path)
    mode = 'a' if file_exists else 'w'

    # Open the CSV file in append mode with newline=''
    with open(csv_path, mode, newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row if the file is newly created
        if not file_exists:
            writer.writerow(["image_id", "gvi", "is_panoramic", "missing", "error"])

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
    return image_path


class FileEventHandler(FileSystemEventHandler):
    def __init__(self):
        self.queue = Queue()

    def on_created(self, event):
        image_path = event.src_path
        file_extension = os.path.splitext(image_path)[1]
        if file_extension.lower() == '.jpg':
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
    city = args[1]
    path = args[2] if len(args) > 2 else ""
    num_threads = int(args[3]) if len(args) > 3 else 5

    prepare_folders(city, path)
    
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
