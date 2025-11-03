import os
import requests
import zipfile
import io

# Base directory for data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
# GitHub ZIP of the Packt repo
GITHUB_ZIP_URL = "https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/archive/refs/heads/master.zip"
CHAPTER_NAME = "Hands-On-Artificial-Intelligence-for-Cybersecurity-master/Chapter04"

def download_from_packt():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "packt_ai_cybersec.zip")

    # Download the GitHub repo if not already done
    if not os.path.exists(os.path.join(DATA_DIR, "Chapter04")):
        print("📦 Downloading Packt dataset from GitHub...")
        response = requests.get(GITHUB_ZIP_URL)
        response.raise_for_status()

        # Save the zip
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Extract only Chapter04
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.startswith(CHAPTER_NAME)]
            zf.extractall(DATA_DIR, members)

        # Move Chapter04 folder to root data/raw
        src_path = os.path.join(DATA_DIR, CHAPTER_NAME)
        dst_path = os.path.join(DATA_DIR, "Chapter04")
        if os.path.exists(dst_path):
            print("Chapter04 already exists.")
        else:
            os.rename(src_path, dst_path)
        print("✅ Chapter04 data ready at data/raw/Chapter04")
    else:
        print("✅ Dataset already downloaded.")

if __name__ == "__main__":
    download_from_packt()

##
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")

    if not os.path.exists(os.path.join(DATA_DIR, "SMSSpamCollection")):
        print("Downloading dataset...")
        r = requests.get(URL)
        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_dataset()


