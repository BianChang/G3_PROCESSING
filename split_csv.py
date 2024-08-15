import os
import pandas as pd

# Define the input folder containing the CSV files
input_folder = r'D:\Chang_files\work_records\Sara_results_2\G4\individual_addtype'
output_folder = r'D:\Chang_files\work_records\Sara_results_2\G4\individual_addtype_By_tumour'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # Load the CSV file
        file_path = os.path.join(input_folder, filename)
        data = pd.read_csv(file_path)

        # Specify the column to split by
        column_to_split_by = 'Tumour.NonTumour'

        # Get the base name without the extension
        base_filename = os.path.splitext(filename)[0]

        # Iterate over each unique value in the column
        for value in data[column_to_split_by].unique():
            # Filter the data for that value
            subset = data[data[column_to_split_by] == value]

            # Create the output file path with the original name + value
            output_file_path = os.path.join(output_folder, f'{base_filename}_{value}.csv')

            # Save the subset to a new CSV file
            subset.to_csv(output_file_path, index=False)

        print(f"Files have been saved successfully for {filename}.")

print("All files have been processed successfully.")
