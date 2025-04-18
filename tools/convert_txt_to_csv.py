import os
import pandas as pd
from glob import glob

# === Paths ===
aug_label_dir = r'sammy\sideview\aug_label_without_rotation'
output_csv = r'sammy\sideview\combined_annotations.csv'

# === Corrected keypoint columns (fixed typos) ===
keypoint_columns = [
    'back_croup-x','back_croup-y','back_left_knee-x','back_left_knee-y','back_left_paw-x','back_left_paw-y',
    'back_left_wrist-x','back_left_wrist-y','back_midpoint-x','back_midpoint-y','back_right_knee-x','back_right_knee-y',
    'back_right_paw-x','back_right_paw-y','back_right_wrist-x','back_right_wrist-y','back_withers-x','back_withers-y',
    'ears_midpoint-x','ears_midpoint-y','front_left_elbow-x','front_left_elbow-y','front_left_paw-x','front_left_paw-y',
    'front_right_elbow-x','front_right_elbow-y','front_right_paw-x','front_right_paw-y','left_ear_base-x','left_ear_base-y',
    'left_ear_tip-x','left_ear_tip-y','left_eye-x','left_eye-y','nose-x','nose-y','right_ear_base-x','right_ear_base-y',
    'right_ear_tip-x','right_ear_tip-y','right_eye-x','right_eye-y','tail_base-x','tail_base-y','tail_end-x','tail_end-y',
    'tail_lower_midpoint-x','tail_lower_midpoint-y','tail_midpoint-x','tail_midpoint-y','tail_upper_midpoint-x','tail_upper_midpoint-y',
    'bbox_tl-x','bbox_tl-y','bbox_br-x','bbox_br-y'
]

# Final columns for DataFrame
columns = ['filename'] + keypoint_columns

# === Read all .txt files ===
txt_files = glob(os.path.join(aug_label_dir, '*.txt'))

data = []

for txt_file in txt_files:
    filename = os.path.basename(txt_file).replace('.txt', '.jpg')  # ✨ now ending in .jpg
    
    # Create empty list of all columns (52 keypoints + 4 bbox = 56)
    row = [""] * len(keypoint_columns)

    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            x, y, idx = parts
            idx = int(idx)
            
            # If it's a valid keypoint index (0–25), place it correctly
            if 0 <= idx < 26:
                row[idx * 2] = x
                row[idx * 2 + 1] = y

            # If it's bbox: 26 = top-left, 27 = bottom-right
            elif idx == 26:
                row[-4] = x  # bbox_tl-x
                row[-3] = y  # bbox_tl-y
            elif idx == 27:
                row[-2] = x  # bbox_br-x
                row[-1] = y  # bbox_br-y

    # Add this row to data
    data.append([filename] + row)

# === Write to CSV ===
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)
print(f"✅ Combined CSV saved at: {output_csv}")
