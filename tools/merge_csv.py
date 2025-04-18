import sys
import pandas as pd
import os

"""
    Usage: python merge_csv.py file1.csv file2.csv ...
"""

def check_column_consistency(file_list):
    column_sets = {}
    
    for file in file_list:
        try:
            df = pd.read_csv(file, nrows=1)  # Read only first row to check columns
            column_sets[file] = set(df.columns)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return None
    
    all_columns = set.union(*column_sets.values())
    
    inconsistent_files = {}
    for file, columns in column_sets.items():
        unique_columns = columns - set.intersection(*column_sets.values())
        if unique_columns:
            inconsistent_files[file] = unique_columns
    
    if inconsistent_files:
        print("Column inconsistency detected:")
        for file, unique_cols in inconsistent_files.items():
            print(f"{file} has unique columns: {unique_cols}")
        sys.exit(1)
    
    return list(all_columns)

def find_output_filename(base_name="output.csv"):
    if not os.path.exists(base_name):
        return base_name
    
    i = 1
    while os.path.exists(f"output_{i}.csv"):
        i += 1
    return f"output_{i}.csv"

def merge_csv_files(file_list, output_file):
    dfs = [pd.read_csv(file) for file in file_list]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved as {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_csv.py file1.csv file2.csv ...")
        sys.exit(1)
    
    file_list = sys.argv[1:]
    common_columns = check_column_consistency(file_list)
    if common_columns:
        output_file = find_output_filename()
        merge_csv_files(file_list, output_file)

if __name__ == "__main__":
    main()
