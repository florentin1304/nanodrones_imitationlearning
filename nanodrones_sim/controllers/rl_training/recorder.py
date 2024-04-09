import numpy as np
from datetime import datetime
import os
import csv
import cv2
import json

class Recorder():
    def __init__(self, metadata: dict = {}):
        # Create run name
        unique_string = self.get_random_string(5)
        datetime_string = datetime.strftime(datetime.now(), "y%y-m%m-d%d_h%H-m%M")
        self.run_name = "run_" + datetime_string + "_" +unique_string
        print("Recording run:", self.run_name)

        # Data variables
        self.meta_data = {}
        self.headers = None
        self.running_index = 0
        self.rows = [] # list of (lists of strings)
        self.images = {} # key=image_name, value=object

    def add_row(self, content: list[str]):
        content = [self.running_index, self.run_name] + content

        assert self.headers is not None, "Headers must be defined first"
        if len(content) != len(self.headers):
            raise Exception("Row and headers must have the same length")
        
        self.rows.append(content)
        self.running_index += 1

    def add_image(self, image: np.array, prefix: str = "img"):
        assert prefix != "", "A prefix is needed"
        
        key = self.run_name + f"_{prefix}{self.running_index}"
        self.images[key] = image

        return key


    def set_headers(self, header_fields: list[str]):
        self.headers = ['index', 'run_name'] + header_fields
        
    def save_data(self):
        # Create directories
        dir = os.path.dirname( os.path.abspath(__file__) )
        working_directory = os.path.join(dir, "data")
        image_directory = os.path.join(working_directory, "images")
        metadata_directory = os.path.join(working_directory, "metadata")
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(image_directory, exist_ok=True)
        os.makedirs(metadata_directory, exist_ok=True)

        # Write the main csv data file
        data_path = os.path.join(working_directory, self.run_name+".csv")
        with open(data_path, 'w',  newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.rows)

        # Save images
        for key, image in self.images.items():
            img_path = os.path.join(image_directory, key+".png")
            cv2.imwrite(img_path, image)

        # Save metadata
        json_object = json.dumps(self.meta_data, indent=4)
        meta_data_path = os.path.join(metadata_directory, self.run_name + ".metadata")
        with open(meta_data_path, "w") as outfile:
            outfile.write(json_object)


    def get_random_string(self, n: int):
        random_string = ''
        for _ in range(n):
            if np.random.uniform() < 0.5:
                i = np.random.randint(65,91)
            else:
                i = np.random.randint(97, 123)

            random_string += chr(i)
        return random_string
