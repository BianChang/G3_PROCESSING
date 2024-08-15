import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def process_cell_data(file_path, thresholds, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # 1. Read the corresponding csv file given the file path
    df = pd.read_csv(file_path)
    df.fillna('NULL', inplace=True)

    # 2. Load thresholding results of these markers
    # Assuming thresholds is a dictionary like {'CAIX': value, 'CD3': value, ...}

    # 3. Judge the positive/negative status of every marker for each cell
    for marker, threshold in thresholds.items():
        column_name = f"{marker}_Cell_Mean"
        status_column = f"{marker}_AutoThreStatus"

        # 4. Add a column for each marker to store their positive/negative status
        df[status_column] = df[column_name].apply(lambda x: 'Positive' if x >= threshold else 'Negative')

    # 5. Add a Cell_Type column to reflect the cell type for each row
    def celltype_rule(row):
        # CD8 T cells
        if (row['CD8_AutoThreStatus'] == 'Positive' and
                row['CD88_AutoThreStatus'] == 'Negative' and
                row['SIRPa_AutoThreStatus'] == 'Negative' and
                row['CD4_AutoThreStatus'] == 'Negative' and
                row['HLA_DR_AutoThreStatus'] in ['Positive', 'Negative'] and
                row['CD56_AutoThreStatus'] == 'Negative' and
                row['CD11c_AutoThreStatus'] in ['Positive', 'Negative'] and
                row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
                row['CD3_AutoThreStatus'] == 'Positive' and
                row['CD45_AutoThreStatus'] == 'Positive' and
                row['CD10_AutoThreStatus'] == 'Negative' and
                row['CD31_AutoThreStatus'] == 'Negative' and
                row['CAIX_AutoThreStatus'] == 'Negative' and
                row['CD20_AutoThreStatus'] == 'Negative' and
                row['IRF8_AutoThreStatus'] == 'Negative' and
                row['CD68_AutoThreStatus'] == 'Negative'):
            return 'CD8 T cells'

        # CD4 T cells
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Positive' and
              row['HLA_DR_AutoThreStatus'] == 'Negative' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] == 'Negative' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Positive' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'CD4 T cells'

        # cDC1s
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Positive' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] == 'Positive' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Positive' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'cDC1s'

        # cDC2s
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['SIRPa_AutoThreStatus'] == 'Positive' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Positive' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] == 'Positive' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'cDC2s'

        # Monocytes
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Positive' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Positive' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'Monocytes'

        # Macrophages
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Positive' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Positive'):
            return 'Macrophages'

        # Tumour cells
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Negative' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] == 'Negative' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Negative' and
              row['CD10_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Positive' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'Tumour cells'

        # B cells
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Negative' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] == 'Negative' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Positive' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'B cells'

        # NK cells
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Negative' and
              row['CD56_AutoThreStatus'] == 'Positive' and
              row['CD11c_AutoThreStatus'] == 'Negative' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Positive' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Negative' and
              row['CAIX_AutoThreStatus'] == 'Negative' and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'NK cells'

        # Endothelial cells
        elif (row['CD8_AutoThreStatus'] == 'Negative' and
              row['CD88_AutoThreStatus'] == 'Negative' and
              row['SIRPa_AutoThreStatus'] == 'Negative' and
              row['CD4_AutoThreStatus'] == 'Negative' and
              row['HLA_DR_AutoThreStatus'] == 'Negative' and
              row['CD56_AutoThreStatus'] == 'Negative' and
              row['CD11c_AutoThreStatus'] == 'Negative' and
              row['PD_1_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD3_AutoThreStatus'] == 'Negative' and
              row['CD45_AutoThreStatus'] == 'Negative' and
              row['CD10_AutoThreStatus'] == 'Negative' and
              row['CD31_AutoThreStatus'] == 'Positive' and
              row['CAIX_AutoThreStatus'] in ['Positive', 'Negative'] and
              row['CD20_AutoThreStatus'] == 'Negative' and
              row['IRF8_AutoThreStatus'] == 'Negative' and
              row['CD68_AutoThreStatus'] == 'Negative'):
            return 'Endothelial cells'

        # If no rule matches:
        return 'others'

    df['AutoThre_Cell_Type'] = df.apply(celltype_rule, axis=1)
    # df['Cell_Type'] = df.apply(celltype_rule, axis=1)

    # Derive new CSV and plot paths based on the original filename
    base_filename = os.path.basename(file_path).split('.')[0]
    new_csv_path = os.path.join(output_directory, f"{base_filename}_addtype.csv")
    plot_path = os.path.join(output_directory, f"{base_filename}.png")

    # 6. Plot a histogram of the distribution of each cell type
    # Set the style and color palette
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", len(df['AutoThre_Cell_Type'].unique()))
    # palette = sns.color_palette("husl", len(df['Cell_Type'].unique()))

    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='AutoThre_Cell_Type', palette=palette, order=df['AutoThre_Cell_Type'].value_counts().index)
    # ax = sns.countplot(data=df, x='Cell_Type', palette=palette,
    #                  order=df['Cell_Type'].value_counts().index)

    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Set the title and labels
    plt.title(base_filename + ' - Distribution of Cell_types', fontsize=15)
    plt.xlabel('AutoThre_Cell_Type', fontsize=12)
    # plt.xlabel('Cell_Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x labels for better visibility
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path)

    # 7. Save the new csv in the same path
    df.to_csv(new_csv_path, index=False)

'''
# All 2PT BGMM
thresholds = {
    'CD8': 10203.53243,
    'CD88': 13221.18093,
    'CD4': 9156.68838,
    'SIRPa': 6835.448997,
    'CD56': 16401.92406,
    'HLA_DR': 4028.927883,
    'CD11c': 6263.882393,
    'PD_1': 6106.16637,
    'CD3': 8572.11186,
    'CD45': 14755.30625,
    'CD10': 16669.86499,
    'CD31': 19592.16034,
    'CAIX': 15138.94958,
    'CD20': 20038.86437,
    'IRF8': 4260.202427,
    'CD68': 2191.631027
}
'''
'''
# All Rosin
thresholds = {
    'CD8': 487.5126648,
    'CD88': 1079.36145,
    'CD4': 1138.599426,
    'SIRPa': 624.0407051,
    'CD56': 743.9141026,
    'HLA_DR': 933.0858773,
    'CD11c': 523.5916651,
    'PD-1': 432.7674863,
    'CD3': 488.6399234,
    'CD45': 1769.803482,
    'CD10': 625.3630062,
    'CD31': 1085.051346,
    'CAIX': 2736.195525,
    'CD20': 407.9211122,
    'IRF8': 785.7022934,
    'CD68': 373.2286594
}
'''
'''
# G2 2PT BGMM
thresholds = {
    'CD8': 11332.64225,
    'CD88': 1963.22507,
    'CD4': 4943.949163,
    'SIRPa': 13494.74061,
    'CD56': 18125.33494,
    'HLA-DR': 3935.410967,
    'CD11c': 6083.057657,
    'PD-1': 3787.584023,
    'CD3': 7643.29244,
    'CD45': 8199.67109,
    'CD10': 14809.84223,
    'CD31': 18291.34119,
    'CAIX': 17319.75815,
    'CD20': 21302.07037,
    'IRF8': 2669.348107,
    'CD68': 1143.7452
}
'''
'''
# G2 Rosin
thresholds = {
    'CD8': 325.0084432,
    'CD88': 254.823401,
    'CD4': 747.1008882,
    'SIRPa': 499.2325641,
    'CD56': 743.9141026,
    'HLA-DR': 395.5668086,
    'CD11c': 295.8173828,
    'PD-1': 167.0150637,
    'CD3': 325.759949,
    'CD45': 920.3717961,
    'CD10': 1181.214447,
    'CD31': 992.6309206,
    'CAIX': 855.2663424,
    'CD20': 203.9605561,
    'IRF8': 543.8493686,
    'CD68': 124.2809396
}
'''
'''
# G3 thresholds 2pt bgmm
thresholds = {
    'CD8': 11125.32839,
    'CD88': 2813.552883,
    'CD4': 6204.244643,
    'SIRPa': 4369.521737,
    'CD56': 12039.82195,
    'HLA-DR': 5308.27696,
    'CD11c': 643.7204433,
    'PD-1': 5490.972147,
    'CD3': 8043.374503,
    'CD45': 10462.75712,
    'CD10': 22206.33835,
    'CD31': 21353.19394,
    'CAIX': 6639.457687,
    'CD20': 12566.53437,
    'IRF8': 4707.701687,
    'CD68': 2572.958443
}
'''
thresholds = {
    'CD8': 12180.29509,
    'CD88': 8785.424343,
    'CD4': 5393.780423,
    'SIRPa': 11117.61695,
    'CD56': 7318.747543,
    'HLA_DR': 6431.605307,
    'CD11c': 1949.798476,
    'PD_1': 5351.71697,
    'CD3': 5864.93412,
    'CD45': 9591.893443,
    'CD10': 38.03025,
    'CD31': 19577.61131,
    'CAIX': 19879.3941,
    'CD20': 1585.558666,
    'IRF8': 2197.568567,
    'CD68': 8583.052377
}




cell_file_folder = r'D:\Chang_files\work_records\Sara_results_2\G4\individual_addthreshold'
file_name = 'AU94843.csv'
cell_file_path = os.path.join(cell_file_folder, file_name)
output_directory = r'D:\Chang_files\work_records\Sara_results_2\G4\add_phenotype'
process_cell_data(cell_file_path, thresholds, output_directory)
