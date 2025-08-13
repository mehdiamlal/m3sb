import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor


#wrapper class to handle curropted data
class SafeDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            # Try to get the item as usual
            return self.dataset[index]
        except Exception as e:
            # If any error occurs, just print a warning and return None
            print(f"Warning: Skipping corrupted data at index {index}. Error: {e}")
            return None
            

def get_data_loader(dataset_name: str, split: str, image_col: str, label_col: str, 
                    processor_checkpoint: str, batch_size: int=32) -> DataLoader:
    """ Loads a dataset and prepares a PyTorch DataLoader for image 
    classification tasks, handling corrupted data gracefully.

    Args:
        dataset_name (str): The name or path of the dataset to load.
        split: The dataset split to use (e.g., 'train', 'test', 'validation').
        image_col (str): The column name containing image data in the dataset.
        label_col (str): The column name containing label data in the dataset.
        processor_checkpoint (str): The checkpoint name or path for the image 
            processor.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
    
    Returns:
        DataLoader: A PyTorch DataLoader that yields batches of processed images 
            and labels,robust to corrupted or missing data.
    """



    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    

    processor = AutoImageProcessor.from_pretrained(processor_checkpoint)
    def transform(examples):
        examples[image_col] = [img.convert("RGB") for img in examples[image_col]]
        examples["pixel_values"] = [processor(img, return_tensors="pt")['pixel_values'] for img in examples[image_col]]
        return examples
    
    dataset.set_transform(transform)

    safe_dataset = SafeDatasetWrapper(dataset)

    def collate_fn(batch):
        #filter out None values that might result from skipping the last item
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        return {
            'pixel_values': torch.cat([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x[label_col] for x in batch])
        }
    
    #num_workers=0 is often safer for error-prone datasets
    return DataLoader(safe_dataset, collate_fn=collate_fn, batch_size=batch_size, 
                      num_workers=0)