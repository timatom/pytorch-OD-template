import torch

def ToTensor(ann):
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
    
def MatchKeys(ann):
    '''
    Matches the keys of the annotation to the required format.
    '''
    ann["boxes"] = ann.pop("bbox")
    ann["labels"] = ann.pop("category_id")
    # ann["image_id"] = ann.pop("image_id")
    # ann["area"] = ann.pop("area")
    # ann["iscrowd"] = ann.pop("iscrowd")
    
    return ann

def Convert2XYWH(ann):
    '''
    Converts the bounding box coordinates from xywh to xyvh format.
    '''
    ann["boxes"][:, 2] = ann["boxes"][:, 0] + ann["boxes"][:, 2]
    ann["boxes"][:, 3] = ann["boxes"][:, 1] + ann["boxes"][:, 3]
    
    return ann

def Compose(transforms):
    '''
    Composes multiple transforms together.
    '''
    def transform(ann):
        for t in transforms:
            ann = t(ann)
        return ann
    
    return transform