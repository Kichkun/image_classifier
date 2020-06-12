import os
import glob
import random
import shutil
from tqdm import tqdm_notebook as tqdm

class SplitFolder():
    def __init__(self, folder, dataset_folder, _split=[0.7,0.15,0.15], class_names={'0':'bad', '1':'bad', '2':'good'}):
        self.folder = folder
        self.dataset_folder = dataset_folder

        _ids = list(set([_file.split('_')[0] for _file in os.listdir(folder)]))
        self.dataset_length = len(_ids)
        random.shuffle(_ids)
        train_len = int(self.dataset_length*_split[0])
        valid_len = int(self.dataset_length * _split[1])
        self._ids = {'train': _ids[:train_len],
                     'valid': _ids[train_len:(train_len+valid_len)],
                     'test':_ids[(train_len+valid_len):]}

        for part in self._ids.keys():
            if not os.path.exists(os.path.join(dataset_folder, part)):
                os.mkdir(os.path.join(dataset_folder, part))

        for key,val in self._ids.items():
           for _id in tqdm(val):
               images = glob.glob(os.path.join(folder, f'{_id}*.jpg'))
               class_name = images[0].split('_')[-1].split('.')[0]
               if class_names is not None:
                   class_name = class_names[class_name]
               for i, _image in enumerate(images):
                   output_path = os.path.join(self.dataset_folder, key, class_name, f'{_id}_{i}.jpg')
                   if not os.path.exists(os.path.join(dataset_folder, key, class_name)):
                       os.mkdir(os.path.join(dataset_folder, key, class_name))
                   shutil.copy(_image, output_path)
