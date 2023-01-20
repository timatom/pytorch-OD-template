from pathlib import Path
import fiftyone as fo

def eval_data(dataset_path, annotations_path):
    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=dataset_path,
        labels_path=annotations_path,
    )
    
    # Launch Voxel's FiftyOne App
    session = fo.launch_app(dataset)

    # Blocks execution until the App is closed
    session.wait()

if __name__ == "__main__":
    ds_version = "v1"
    ds_type = "train"
    
    # Project root directory
    proj_dir = Path(__file__).parent
    
    # The path to the dataset's data directory
    data_path = str(proj_dir.joinpath(f"datasets/{ds_version}/{ds_type}/"))

    # The path to the dataset's COCO formatted annotations .json file
    annotations_path = str(proj_dir.joinpath(f"datasets/{ds_version}/{ds_type}/annotations.json"))
    
    eval_data(data_path, annotations_path)