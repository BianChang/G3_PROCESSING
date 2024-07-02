import pandas as pd

# Load the CSV file
data = pd.read_csv(r'D:\Chang_files\work_records\Sara_results\G4\addtype/Tumour_Grade_4_addtype.csv')

# Replace '/path/to/your/file.csv' with the actual path of your CSV file

# Specify the column to split by
column_to_split_by = 'Sample.ID'

# Iterate over each unique value in the column
for value in data[column_to_split_by].unique():
    # Filter the data for that value
    subset = data[data[column_to_split_by] == value]
    file_path = fr'D:\Chang_files\work_records\Sara_results\G4\individual_addtype/{value}.csv'
    # Save the subset to a new CSV file
    subset.to_csv(file_path, index=False)

print("Files have been saved successfully.")
