import pandas as pd
import os
from glob import glob

# === Paths ===
train_csv_path = r'sammy\sideview\combined_annotations.csv'
test_csv_path = r'sammy\sideview\output_4_sorted_cleaned.csv'
test_image_folder = r'sammy\sideview\test'
output_merged_csv_path = r'sammy\sideview\merged_labels.csv'  # Or overwrite train_csv_path if needed

# === Load CSVs ===
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# === Fix column names for consistency (replace 'midpont' with 'midpoint') ===
train_df.columns = [col.replace('midpont', 'midpoint') for col in train_df.columns]
test_df.columns = [col.replace('midpont', 'midpoint') for col in test_df.columns]

# === Normalize filenames ===
train_df['filename'] = train_df['filename'].apply(lambda x: os.path.basename(str(x)).strip())
test_df['filename'] = test_df['filename'].apply(lambda x: os.path.basename(str(x)).strip())

# === Append .jpg to train_df entries without extension ===
train_df['filename'] = train_df['filename'].apply(
    lambda x: x if os.path.splitext(x)[1] else x + '.jpg'
)

# === Get list of test image filenames in folder ===
image_extensions = ('*.jpg', '*.jpeg', '*.png')
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(test_image_folder, ext)))
test_image_filenames = [os.path.basename(p) for p in image_paths]

# === Filter test_df to only those images present in test_image_folder ===
test_df_filtered = test_df[test_df['filename'].isin(test_image_filenames)]

# === Combine with train_df ===
merged_df = pd.concat([train_df, test_df_filtered], ignore_index=True)

# === Save merged CSV ===
merged_df.to_csv(output_merged_csv_path, index=False)
print(f"Merged CSV saved to: {output_merged_csv_path}")
print(f"Train rows: {len(train_df)}, Added test rows: {len(test_df_filtered)}, Total: {len(merged_df)}")
