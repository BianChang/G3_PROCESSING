import pandas as pd
import os

# Paths
input_csv_path = r"D:\Chang_files\workspace\data\G4_SAMPLES\G4_batchcorrected_dataset_cluster.ids_Cell_Means.csv"
matching_csv_folder = r"D:\Chang_files\workspace\data\G4_SAMPLES\SAMPLES"
output_folder = r"D:\Chang_files\workspace\data\G4_SAMPLES\samples_clusterid"

# Read the first CSV file
df1 = pd.read_csv(input_csv_path)

rename_dict = {
    "CAIX+ Tumour cells": "Tumour cells",
    "CD4+ T cells": "CD4 T cells",
    "CD8+ T cells": "CD8 T cells",
    "Endo": "Endothelial cells",
    "CD20+ B cells": "B cells"
}

# Iterate over unique Sample.IDs in the first CSV
for sample_id in df1['Sample.ID'].unique():
    # Construct the matching CSV filename based on the Sample.ID
    print(sample_id)
    matching_files = [f for f in os.listdir(matching_csv_folder) if sample_id in f and f.endswith('.csv')]

    if matching_files:
        matching_csv_filename = matching_files[0]
    else:
        raise FileNotFoundError(f"No CSV file found for sample ID {sample_id} in {matching_csv_folder}")

    matching_csv_path = os.path.join(matching_csv_folder, matching_csv_filename)

    # Read the matching CSV file
    df_matching = pd.read_csv(matching_csv_path)

    # Merge the two dataframes based on X and Y columns
    merged_df = pd.merge(df_matching, df1[['X', 'Y', 'Cluster.id']], on=['X', 'Y'], how='left')
    # Rename values in the Cluster.id column
    merged_df['Cluster.id'] = merged_df['Cluster.id'].replace(rename_dict)

    # Print total number of cells (rows) in the matching CSV file
    total_cells = df_matching.shape[0]
    unfilled_count = merged_df['Cluster.id'].isna().sum()
    print(f"For {matching_csv_filename}, there are {total_cells} cells in total. "
          f"there are {unfilled_count} unfilled blanks in the 'Cluster.id' column.")

    # Save the merged dataframe to the specified output folder
    output_path = os.path.join(output_folder, matching_csv_filename)
    merged_df.to_csv(output_path, index=False)

print("Processing completed and new CSV files saved!")
