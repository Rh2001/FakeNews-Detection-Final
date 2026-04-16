import os
import requests
import zipfile
import io

def download_raw_liar():
    url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    target_dir = "data"
    
    # Create the data directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading raw LIAR dataset from {url}...")
    
    try:
        # Request the file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the contents directly into the /data folder
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_dir)
            
        print(f"Success! Raw files extracted to: {os.path.abspath(target_dir)}")
        
        # List files to verify
        print("Files in data folder:", os.listdir(target_dir))
        
    except Exception as e:
        print(f"Failed to download or extract dataset: {e}")

if __name__ == "__main__":
    download_raw_liar()