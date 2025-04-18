# import pandas as pd
# import os
# import glob

# label_txt_dir = r"sammy\sideview\aug_label"  # Directory where your label .txt files are
# output_csv_path = r"sammy\sideview\consolidated_labels.csv"

# rows = []

# keypoint_order = [
#     "back_croup", "back_left_knee", "back_left_paw", "back_left_wrist", "back_midpoint",
#     "back_right_knee", "back_right_paw", "back_right_wrist", "back_withers", "ears_midpoint",
#     "front_left_elbow", "front_left_paw", "front_right_elbow", "front_right_paw",
#     "left_ear_base", "left_ear_tip", "left_eye", "nose", "right_ear_base", "right_ear_tip",
#     "right_eye", "tail_base", "tail_end", "tail_lower_midpont", "tail_midpoint", "tail_upper_midpont"
# ]

# # Plus 4 bbox values at the end
# columns = ["filename"]
# for part in keypoint_order:
#     columns.extend([f"{part}-x", f"{part}-y"])
# columns += ["bbox_tl-x", "bbox_tl-y", "bbox_br-x", "bbox_br-y"]


# for txt_file in glob.glob(os.path.join(label_txt_dir, '*.txt')):
#     with open(txt_file, 'r') as f:
#         keypoints = [line.strip().split() for line in f.readlines()]

#     # Create a mapping: class_id -> (x, y)
#     kp_map = {int(cls): (float(x), float(y)) for x, y, cls in keypoints}

#     # Build the row
#     filename = os.path.splitext(os.path.basename(txt_file))[0] + ".jpg"
#     row = [filename]

#     # Fill keypoints in defined order (some may be missing)
#     for idx in range(len(keypoint_order)):
#         if idx in kp_map:
#             row.extend(kp_map[idx])
#         else:
#             row.extend([None, None])  # or 0.0, 0.0

#     # Compute bounding box (if all keypoints are available)
#     xs = [x for idx, (x, y) in kp_map.items()]
#     ys = [y for idx, (x, y) in kp_map.items()]
#     if xs and ys:
#         row += [min(xs), min(ys), max(xs), max(ys)]
#     else:
#         row += [None, None, None, None]

#     rows.append(row)

# df = pd.DataFrame(rows, columns=columns)
# df.to_csv(output_csv_path, index=False)
# print(f"Consolidated CSV saved to {output_csv_path}")

import os
import pandas as pd
import numpy as np
from glob import glob

# === UPDATE THESE PATHS ===
label_csv_file = r"sammy\sideview\output_4_sorted_cleaned.csv"
train_img_dir = r"sammy\sideview\train"
test_img_dir = r"sammy\sideview\test"
output_csv = r"sammy\sideview\consolidated_labels.csv"

# === Desired column order ===
columns_order = [
    'filename',
    'back_croup-x', 'back_croup-y',
    'back_left_knee-x', 'back_left_knee-y',
    'back_left_paw-x', 'back_left_paw-y',
    'back_left_wrist-x', 'back_left_wrist-y',
    'back_midpoint-x', 'back_midpoint-y',
    'back_right_knee-x', 'back_right_knee-y',
    'back_right_paw-x', 'back_right_paw-y',
    'back_right_wrist-x', 'back_right_wrist-y',
    'back_withers-x', 'back_withers-y',
    'ears_midpoint-x', 'ears_midpoint-y',
    'front_left_elbow-x', 'front_left_elbow-y',
    'front_left_paw-x', 'front_left_paw-y',
    'front_right_elbow-x', 'front_right_elbow-y',
    'front_right_paw-x', 'front_right_paw-y',
    'left_ear_base-x', 'left_ear_base-y',
    'left_ear_tip-x', 'left_ear_tip-y',
    'left_eye-x', 'left_eye-y',
    'nose-x', 'nose-y',
    'right_ear_base-x', 'right_ear_base-y',
    'right_ear_tip-x', 'right_ear_tip-y',
    'right_eye-x', 'right_eye-y',
    'tail_base-x', 'tail_base-y',
    'tail_end-x', 'tail_end-y',
    'tail_lower_midpont-x', 'tail_lower_midpont-y',
    'tail_midpoint-x', 'tail_midpoint-y',
    'tail_upper_midpont-x', 'tail_upper_midpont-y',
    'bbox_tl-x', 'bbox_tl-y',
    'bbox_br-x', 'bbox_br-y'
]

# Load master CSV
df = pd.read_csv(label_csv_file)

df['filename'] = df['filename'].apply(lambda x: os.path.basename(str(x)).strip() + '.png')


# Clean filename column
# df['filename'] = df['filename'].apply(lambda x: os.path.basename(str(x)).strip())

# Gather all image paths
image_paths = glob(os.path.join(train_img_dir, '*')) + glob(os.path.join(test_img_dir, '*'))

print(f"First few filenames in DataFrame: {df['filename'].head()}")
print(f"First few image paths: {image_paths[:10]}")

# for img_path in image_paths:
#     fname = os.path.basename(img_path)
#     print(f"Matching label for: {fname}")  # Debug output to confirm which filenames are being processed
#     row = df[df['filename'] == fname]
#     if row.empty:
#         # print(f"⚠️ Label not found for: {fname}")
#     else:
#         # print(f"Found label for: {fname}")

consolidated_rows = []

for img_path in image_paths:
    fname = os.path.basename(img_path)
    
    # Match label row
    row = df[df['filename'] == fname]
    if row.empty:
        # print(f"⚠️ Label not found for: {fname}")
        continue

    row = row.iloc[0]  # Take the first matched row
    entry = {'filename': fname}

    # Add keypoints in required order
    missing_keys = []
    for key in columns_order[1:-4]:  # Skip filename and bbox columns for now
        if key in row:
            entry[key] = row[key]
        else:
            entry[key] = None
            missing_keys.append(key)

    # Handle bounding box calculation
    x_coords = []
    y_coords = []
    for k in entry:
        if k.endswith('-x') and entry[k] is not None and not pd.isna(entry[k]):
            x_coords.append(entry[k])
        elif k.endswith('-y') and entry[k] is not None and not pd.isna(entry[k]):
            y_coords.append(entry[k])
    
    if x_coords and y_coords:
        entry['bbox_tl-x'] = min(x_coords)
        entry['bbox_tl-y'] = min(y_coords)
        entry['bbox_br-x'] = max(x_coords)
        entry['bbox_br-y'] = max(y_coords)
    else:
        entry['bbox_tl-x'] = entry['bbox_tl-y'] = entry['bbox_br-x'] = entry['bbox_br-y'] = None

    consolidated_rows.append(entry)

# Convert to DataFrame
final_df = pd.DataFrame(consolidated_rows)

print(final_df.columns)

# Ensure correct column order
final_df = final_df[columns_order]



# Save
final_df.to_csv(output_csv, index=False)
print(f"✅ Consolidated label CSV saved to: {output_csv}")
