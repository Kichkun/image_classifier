import os
import glob

class SplitFolder():
    def __init__(self, folder, dataset_folder, _split=[0.7,0.15,0.15]):
        self.folder = folder
        self.dataset_folder = dataset_folder
        self.ids = list(set([_file.split('_')[0] for _file in os.listdir(folder)]))

        for _id in self.ids:
           images = glob.glob(os.path.join(folder, f'{_id}*.jpg'))
