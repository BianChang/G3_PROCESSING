import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings('ignore')
from tqdm import tqdm



folder_path = r'D:\Chang_files\workspace\Github_workspace\G3_PROCESSING\results\V2\add_automatedThre\add_mauanlclustering'
plot_folder = os.path.join(folder_path, 'clustering_results')

if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

dataframes = []
for csv_file in tqdm(os.listdir(folder_path)):
    if csv_file.endswith('.csv'):
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        df = df.replace(np.nan, 'NULL')
        df = df.sample(frac=0.5)  # Adjust the fraction based on your requirements
        df['Slide'] = csv_file[:-4]
        dataframes.append(df)


data = pd.concat(dataframes, axis=0, ignore_index=True)

# Ensure your Phenotype and Slide columns are categorical
data['Slide'] = data['Slide'].astype('category')
columns = [
    "CAIX_Status", "CD3_Status", "CD4_Status", "CD8_Status", "CD10_Status",
    "CD11c_Status", "CD20_Status", "CD31_Status", "CD45_Status", "CD56_Status",
    "CD68_Status", "CD88_Status", "HLA-DR_Status", "IRF8_Status", "PD-1_Status",
    "SIRPa_Status", "Cell_Type", "CAIX_AutoThreStatus", "CD3_AutoThreStatus",
    "CD4_AutoThreStatus", "CD8_AutoThreStatus", "CD10_AutoThreStatus",
    "CD11c_AutoThreStatus", "CD20_AutoThreStatus", "CD31_AutoThreStatus",
    "CD45_AutoThreStatus", "CD56_AutoThreStatus", "CD68_AutoThreStatus",
    "CD88_AutoThreStatus", "HLA-DR_AutoThreStatus", "IRF8_AutoThreStatus",
    "PD-1_AutoThreStatus", "SIRPa_AutoThreStatus", "AutoThre_Cell_Type", "Cluster.id"
]

for col in columns:
    data[col] = data[col].astype('category')




# Select columns of interest for clustering, scale the data, and create an AnnData object
cols = [
    "CD8_Cell_Mean", "CD88_Cell_Mean", "CD4_Cell_Mean", "SIRPa_Cell_Mean",
    "CD56_Cell_Mean", "HLA-DR_Cell_Mean", "CD11c_Cell_Mean", "PD-1_Cell_Mean",
    "CD3_Cell_Mean", "CD45_Cell_Mean", "CD10_Cell_Mean", "CD31_Cell_Mean",
    "CAIX_Cell_Mean", "CD20_Cell_Mean", "IRF8_Cell_Mean", "CD68_Cell_Mean"
]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[cols])
data_scaled_df = pd.DataFrame(data_scaled, columns=cols)
if data_scaled_df.isnull().values.any():
    print("NaN values found in scaled data. Dropping rows with NaN values...")
    data_scaled_df = data_scaled_df.dropna()
    data = data.loc[data_scaled_df.index]
    print("Rows with NaN values have been dropped.")
else:
    print("No NaN values found in scaled data.")
data_scaled = data_scaled_df.values

adata = sc.AnnData(data_scaled)

data.index = data.index.astype(str)
adata.obs = data[['Slide'] + columns]

# Batch correction
sc.pp.pca(adata)
# Add batch information for BBKNN
adata.obs['batch'] = data['Slide']
# Batch correction using BBKNN
sc.external.pp.bbknn(adata, batch_key='batch')

# Compute the neighborhood graph, perform clustering, and dimensionality reduction
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
sc.tl.leiden(adata, resolution=0.21)
sc.tl.umap(adata, min_dist=0.05, spread=5, n_components=2)

# Create UMAP plots for each method's phenotype
print(f"Total number of cells: {adata.n_obs}")

all_labels = set()
for col in ["Cell_Type", "AutoThre_Cell_Type", "Cluster.id"]:
    all_labels.update(adata.obs[col].cat.categories.tolist())

all_labels = sorted(list(all_labels))
color_map = {label: plt.cm.tab20(i % 20) for i, label in enumerate(all_labels)}

# List of columns to plot
for col in columns:
    fig, axs = plt.subplots(figsize=(6, 4), dpi=300)

    # Plot UMAP for the current column
    if col in ["Cell_Type", "AutoThre_Cell_Type", "Cluster.id"]:
        sc.pl.umap(adata, color=col, ax=axs, show=False, size=30, alpha=0.7, title=col, palette=color_map)
    else:
        sc.pl.umap(adata, color=col, ax=axs, show=False, size=30, alpha=0.7, title=col)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'clustering_{col}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Leiden plot
fig, axs = plt.subplots(figsize=(6, 4), dpi=300)
sc.pl.umap(adata, color='leiden', ax=axs, show=False, size=30, alpha=0.7, title='Leiden')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'clustering_leiden.png'), dpi=300, bbox_inches='tight')
plt.close()

# Slide plot
fig, axs = plt.subplots(figsize=(8, 4), dpi=300)
sc.pl.umap(adata, color='Slide', ax=axs, show=False, size=30, alpha=0.7, title='Slide')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'clustering_Slide.png'), dpi=300, bbox_inches='tight')
plt.close()