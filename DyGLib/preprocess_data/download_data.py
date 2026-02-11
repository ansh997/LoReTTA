import os
import getpass
import requests
from tqdm import tqdm

# url = "https://zenodo.org/records/7213796#.Y1cO6y8r30o"

# full_data = "https://zenodo.org/api/records/7213796/files-archive"

urls = {"mooc": "https://zenodo.org/records/7213796/files/mooc.zip?download=1",
        "reddit": "https://zenodo.org/records/7213796/files/reddit.zip?download=1",
        "wikipedia": "https://zenodo.org/records/7213796/files/wikipedia.zip?download=1",
        "uci": "https://zenodo.org/records/7213796/files/uci.zip?download=1",
        "bitcoinotc": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
        "socialevo": "https://zenodo.org/records/7213796/files/SocialEvo.zip?download=1",
        "enron": "https://zenodo.org/records/7213796/files/enron.zip?download=1",
        }


scratch_location = rf'/scratch/{getpass.getuser()}'

if __name__=="__main__":
    
    for filename, url in tqdm(urls.items()):    
        response = requests.get(url)
        
        # Specify the folder path where you want to save the file
        folder_path = scratch_location + '/DG_data'
        os.makedirs(folder_path, exist_ok=True)
        # print(folder_path, " in ", scratch_location)
        
        # Full path to the file
        if filename != "bitcoinotc":
            file_path = os.path.join(folder_path, f"{filename}.zip")
        else:
            file_path = os.path.join(folder_path, "bitcoinotc.csv.gz")

        # Save the file
        with open(file_path, "wb") as file:
            file.write(response.content)

        # print(f"File downloaded to {file_path}")
