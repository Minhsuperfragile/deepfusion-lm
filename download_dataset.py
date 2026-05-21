import os
from datasets import load_dataset

dataset_name = "minhxthanh/Vietnam-History-200K-Vi"
dataset_store_path = "./data/vietnamese-history-qa"

def download_dataset():
    print(f"Starting download of dataset: {dataset_name}")
    
    # Load the dataset (this will download it to the Hugging Face cache first)
    try:
        dataset = load_dataset(dataset_name)
        print("Dataset downloaded successfully.")
        
        # Ensure the storage directory exists
        os.makedirs(os.path.dirname(dataset_store_path), exist_ok=True)
        
        # Save to the specific path
        print(f"Saving dataset to: {dataset_store_path}")
        dataset.save_to_disk(dataset_store_path)
        print("Dataset saved to disk successfully!")
        
    except Exception as e:
        print(f"Error downloading or saving dataset: {e}")


if __name__ == "__main__":
    download_dataset()
