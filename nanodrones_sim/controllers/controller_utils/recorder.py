from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
import csv
import cv2
import json
import tqdm


class Recorder():
    def __init__(self, metadata: dict = {}, save_dir='data', mode: str='on', save_img=True):
        # Functionality
        self.mode = mode
        assert self.mode == 'on' or self.mode == 'off', f'Unrecognized recorder mode {mode}'
        self.save_img = save_img

        # Create run name
        unique_string = self.get_random_string(5)
        datetime_string = datetime.strftime(datetime.now(), "y%y-m%m-d%d_h%H-m%M")
        self.run_name = "run_" + datetime_string + "_" +unique_string
        print("Recording run:", self.run_name)

        # Data variables
        self.meta_data = metadata
        self.traj = None
        self.vel_profile = None
        self.headers = None
        self.running_index = 0
        self.rows = [] # list of (lists of strings)
        self.images = {} # key=image_name, value=object

        # Working directories
        curr_dir = os.path.dirname( os.path.abspath(__file__) )
        path = Path(curr_dir).parent.parent.absolute()
        self.working_directory = os.path.join(path, save_dir)
        self.image_directory = os.path.join(self.working_directory, "images")
        self.metadata_directory = os.path.join(self.working_directory, "metadata", self.run_name)

        os.makedirs(self.working_directory, exist_ok=True)
        os.makedirs(self.image_directory, exist_ok=True)
        os.makedirs(self.metadata_directory, exist_ok=True)

    def is_on(self):
        return self.mode == 'on'
    
    def is_recording_images(self):
        return self.save_img
    
    def add_row(self, content: list[str]):
        content = [self.running_index, self.run_name] + content

        assert self.headers is not None, "Headers must be defined first"
        if len(content) != len(self.headers):
            raise Exception(f"Row and headers must have the same length, found {len(content)=} and {len(self.headers)=}")
        
        self.rows.append(content)
        self.running_index += 1

    def add_image(self, image: np.array, prefix: str = "img"):
        assert prefix != "", "A prefix is needed"
        
        key = self.run_name + f"_{prefix}{self.running_index}"
        self.images[key] = image

        return key

    def add_trajectory(self, report):
        self.trajectory = report

    def set_headers(self, header_fields: list[str]):
        self.headers = ['index', 'run_name'] + header_fields
        
    def get_data_df(self):
        return pd.DataFrame(self.rows, columns=self.headers)


    def save_data(self, data=True, images=True, metadata=True, trajectory=True):
        # Write the main csv data file
        if data:
            data_path = os.path.join(self.working_directory, self.run_name+".csv")
            with open(data_path, 'w',  newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                writer.writerows(self.rows)

        # Save images
        if images:
            for key, image in tqdm.tqdm(self.images.items(), desc="Saving images: "):
                img_path = os.path.join(self.image_directory, key+".png")
                cv2.imwrite(img_path, image)

        # Save metadata
        if metadata:
            json_object = json.dumps(self.meta_data, indent=4)
            meta_data_path = os.path.join(self.metadata_directory, "info.json")
            with open(meta_data_path, "w") as outfile:
                outfile.write(json_object)

        # Create and save report
        if trajectory:
            trajectory_path = os.path.join(self.metadata_directory, "trajectory.csv")
            df = pd.DataFrame(self.trajectory, columns=['x','y','z','vel'])
            df.to_csv(trajectory_path)

    def get_random_string(self, n: int):
        random_string = ''
        for _ in range(n):
            if np.random.uniform() < 0.5:
                i = np.random.randint(65, 91)
            else:
                i = np.random.randint(97, 123)

            random_string += chr(i)
        return random_string
