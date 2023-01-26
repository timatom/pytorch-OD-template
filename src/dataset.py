from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
import json

class Dataset(Dataset):
    def __init__(self, root, anns_file_path, transform=None):
        '''
        Initializes the dataset object.
        '''
        self.root = root
        self.anns_file_path = anns_file_path
        self.transform = transform
        
        self.data, self.anns = self.load_data()
        
    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.data)
    
    def to_tensor(self, ann):
        '''
        Converts the annotation values to tensors.
        '''
        for key, val in ann.items():
            if type(val) == float:
                ann[key] = torch.tensor([val], dtype=torch.float32)
            elif type(val) == int:
                ann[key] = torch.tensor([val], dtype=torch.int64)
            elif type(val) == list and type(val[-1]) == float:
                ann[key] = torch.tensor([val], dtype=torch.float32)
            elif type(val) == list and type(val[0]) == int:
                ann[key] = torch.tensor([val], dtype=torch.int64)
            elif type(val) == list and type(val[0]) == list and type(val[0][0]) == float:
                ann[key] = torch.tensor(val, dtype=torch.float32)
            elif type(val) == list and type(val[0]) == list and type(val[0][0]) == int:
                ann[key] = torch.tensor(val, dtype=torch.int64)
        
        return ann
    
    def match_keys(self, ann):
        '''
        Matches the keys of the annotation to the required format.
        '''
        ann["boxes"] = ann.pop("bbox")
        ann["labels"] = ann.pop("category_id")
        # ann["image_id"] = ann.pop("image_id")
        # ann["area"] = ann.pop("area")
        # ann["iscrowd"] = ann.pop("iscrowd")
        
        return ann
    
    def covert_to_xyvh(self, ann):
        '''
        Converts the bounding box coordinates from xywh to xyvh format.
        '''
        ann["boxes"][:, 2] = ann["boxes"][:, 0] + ann["boxes"][:, 2]
        ann["boxes"][:, 3] = ann["boxes"][:, 1] + ann["boxes"][:, 3]
        
        return ann
    
    def ann_transforms(self, transforms, ann):
        '''
        Transforms the annotations to the format required by the model.
        '''
        
        for transform in transforms:
            ann = transform(ann)
        
        return ann

    def __getitem__(self, idx):
        '''
        Returns the item at the given index.
        '''
        img = self.data[idx]
        ann = self.anns[idx]

        if self.transform:
            img = self.transform(img)
            
        ann = self.ann_transforms([self.to_tensor, self.match_keys, self.covert_to_xyvh], ann)

        return img, ann

    def load_data(self):
        '''
        Loads the data and annotations from the COCO formatted dataset.
        '''
        data = []
        anns = []

        with open(self.anns_file_path, 'r') as f:
            tmp_anns = json.load(f)

        self.categories = tmp_anns["categories"]
        self.images = tmp_anns["images"]

        for image, ann in zip(tmp_anns["images"], tmp_anns["annotations"]):
            if ann is not None:
                img_path = Path(self.root).joinpath(image["file_name"])

                with Image.open(img_path).convert('RGB') as img:
                    data.append(img)
                    
                anns.append(ann)
            
        return data, anns