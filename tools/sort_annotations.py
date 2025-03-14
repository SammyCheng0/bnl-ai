import pandas as pd

def sort_csv_axes(input_file, output_file):
    # Example usage:
    # sort_csv_axes("annotations.csv", "sorted_annotations.csv")
    
    # Load the CSV file
    df = pd.read_csv(input_file)
    
    # Extract column names excluding 'filename'
    columns = df.columns.tolist()
    columns.remove("filename")
    
    # Group columns based on prefix and sort
    sorted_columns = sorted(columns, key=lambda col: (col.rsplit('-', 1)[0], col))
    
    # Reorder the dataframe with 'filename' first
    df = df[["filename"] + sorted_columns]
    
    # Save the sorted file
    df.to_csv(output_file, index=False)

sort_csv_axes("annotations.csv", "sorted_annotations.csv")
