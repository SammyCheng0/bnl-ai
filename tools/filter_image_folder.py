import os
import pandas as pd

def clean_image_folder(image_folder, annotation_path):
    # Load the CSV
    df = pd.read_csv(annotation_path)

    # Assume the filenames in the CSV have extensions like .jpg or .png
    label_filenames = set(df['filename'].astype(str))

    # List all files in the folder
    for fname in os.listdir(image_folder):
        fpath = os.path.join(image_folder, fname)
        if os.path.isfile(fpath):
            name, ext = os.path.splitext(fname)
            if fname not in label_filenames and name not in label_filenames:
                print(f"Removing {fname} (not in CSV)")
                os.remove(fpath)

# Clean both folders
clean_image_folder(r'sammy\sideview\train', r'sammy\output_4_sorted_cleaned.csv')
clean_image_folder(r'sammy\sideview\test', r'sammy\output_4_sorted_cleaned.csv')