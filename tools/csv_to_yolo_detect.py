import pandas as pd
import os

def convert_bbox(top_left_x, top_left_y, bottom_right_x, bottom_right_y, img_width=1920, img_height=1200):
    center_x = (top_left_x + bottom_right_x) / 2.0 / img_width
    center_y = (top_left_y + bottom_right_y) / 2.0 / img_height
    width = (bottom_right_x - top_left_x) / img_width
    height = (bottom_right_y - top_left_y) / img_height
    return center_x, center_y, width, height

# Load CSV file
file_path = r'datasets\topv11\annotations.csv'
df = pd.read_csv(file_path)

# Ensure the required columns exist
required_columns = ['filename', 'bbox_tl-x', 'bbox_tl-y', 'bbox_br-x', 'bbox_br-y']
if not all(col in df.columns for col in required_columns):
    raise ValueError("CSV file is missing one or more required columns.")

# Define output folder
output_folder = r'datasets\top_detection_v1\labels'
os.makedirs(output_folder, exist_ok=True)

# Process each row and save bounding box information
for _, row in df.iterrows():
    image_filename = row['filename']
    center_x, center_y, width, height = convert_bbox(
        row['bbox_tl-x'], row['bbox_tl-y'], row['bbox_br-x'], row['bbox_br-y']
    )
    
    output_filename = os.path.join(output_folder, f"{os.path.splitext(image_filename)[0]}.txt")
    with open(output_filename, 'a') as f:
        f.write(f"0 {center_x} {center_y} {width} {height}\n")

print(f"Bounding box annotations saved in: {output_folder}")
