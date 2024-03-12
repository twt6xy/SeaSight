import os

from roboflow import Roboflow


class RoboflowDatasetLoader:
    def __init__(
        self, api_key: str, workspace: str, project_name: str, version_number: int
    ):
        self.rf = Roboflow(api_key=api_key)
        self.workspace = workspace
        self.project_name = project_name
        self.version_number = version_number
        self.data_dir = "../data/"

    def download_dataset(self):
        project = self.rf.workspace(self.workspace).project(self.project_name)
        version = project.version(self.version_number)
        dataset = version.download("tfrecord")
        self.move_dataset(dataset.location)

    def move_dataset(self, download_location):
        os.makedirs(self.data_dir, exist_ok=True)
        os.rename(
            download_location,
            os.path.join(self.data_dir, os.path.basename(download_location)),
        )
        print(f"Dataset moved to {self.data_dir}")
