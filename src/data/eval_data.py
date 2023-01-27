from pathlib import Path
import fiftyone as fo

def eval_data(data_path, annotations_path):
    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=annotations_path,
    )
    
    # Launch Voxel's FiftyOne App
    session = fo.launch_app(dataset)

    # Blocks execution until the App is closed
    session.wait()

if __name__ == "__main__":
    time_stamp = "2023-01-09T-07:29+00"
    data_type = "images"
    
    ds_version = "v1"
    ds_type = "train"
    
    # Project root directory
    proj_dir = Path(__file__).resolve().parents[1]
    
    # The path to the raw data images
    raw_data_path = str(proj_dir.joinpath(f"data/raw/{time_stamp}/{data_type}/"))
    
    # The path to the raw data annotations .json file
    raw_annotations_path = str(proj_dir.joinpath(f"data/raw/{time_stamp}/annotations.json"))
    
    # Evaluate the raw data
    eval_data(raw_data_path, raw_annotations_path)
    
    # # The path to the dataset's data directory
    # dataset_path = str(proj_dir.joinpath(f"datasets/{ds_version}/{ds_type}/"))

    # # The path to the dataset's COCO formatted annotations .json file
    # annotations_path = str(proj_dir.joinpath(f"datasets/{ds_version}/{ds_type}/annotations.json"))
    
    # # Evaluate the dataset
    # eval_data(data_path, annotations_path)