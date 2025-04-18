import pandas as pd

def remove_empty_bboxes(label_file, output_file):
    # Read the CSV file containing the bounding boxes and keypoints
    labels = pd.read_csv(label_file)
    
    # List of bounding box columns (based on your dataset's structure)
    bbox_columns = ['bbox_tl-x', 'bbox_tl-y', 'bbox_br-x', 'bbox_br-y']
    
    # Remove rows where any of the bbox columns have NaN values
    cleaned_labels = labels.dropna(subset=bbox_columns, how='any')
    
    # Save the cleaned DataFrame to a new CSV file
    cleaned_labels.to_csv(output_file, index=False)
    
    print(f"Rows with empty bounding boxes removed. Cleaned data saved to {output_file}.")

# Example usage
label_file = "output_4_sorted.csv"
output_file = "output_4_sorted_cleaned.csv"
remove_empty_bboxes(label_file, output_file)
