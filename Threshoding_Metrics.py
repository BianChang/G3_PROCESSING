import os
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from tqdm import tqdm
import textwrap
import math
from scipy.stats import pearsonr
from math import sqrt
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter1d


def find_intersections(pdf1, pdf2, x):
    intersections = []
    for i in range(1, len(x)):
        if (pdf1[i] - pdf2[i]) * (pdf1[i-1] - pdf2[i-1]) < 0:  # Check for sign change
            if pdf1[i] != 0 and pdf2[i] != 0:  # Exclude intersection at zero
                intersections.append(x[i])
    return intersections


def bgmm_threshold_mean(data, plot_path, compareflag, threshold):
    component = 2
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values
    gmm = BayesianGaussianMixture(n_components=component, max_iter=3000).fit(data.reshape(-1, 1))

    min_value = np.min(data)
    max_value = np.max(data)
    # Extend the range by a percentage of the range
    extension = 0.1 * (max_value - min_value)
    # Create x based on the extended range
    x = np.arange(min_value - extension, max_value + extension, 0.01).reshape(-1, 1)

    # predictions = gmm.predict(x)
    # switch_indices = np.where(np.diff(predictions))[0]

    # Plot histogram of original data
    plt.figure(figsize=(6, 6), dpi=300)
    plt.hist(data, bins=50, color='yellow', density=True, edgecolor='black')

    # Generate EM simulation and plot on top of original data
    sum_pdf = np.zeros_like(x)
    pdfs = []
    for i in range(component):

        pdf = gmm.weights_[i] * stats.norm.pdf(x, gmm.means_[i], np.sqrt(gmm.covariances_[i]))
        pdfs.append(pdf)
        sum_pdf += pdf  # add each pdf to the sum
        plt.plot(x, pdf, color='b')
        if i > 0:
            intersections = find_intersections(pdfs[i], pdfs[i - 1], x)
            if intersections:
                # Get the means of the two components
                mean1 = gmm.means_[i - 1]
                mean2 = gmm.means_[i]

                # Order the means so mean1 is less than mean2
                if mean2 < mean1:
                    mean1, mean2 = mean2, mean1

                # Choose the intersection that is between the means
                chosen_intersection = None
                for intersection in intersections:
                    if mean1 <= intersection <= mean2:
                        chosen_intersection = intersection
                if chosen_intersection == None:
                    print('intersection falls out of means range')
                    component_index = np.argmax(gmm.weights_)
                    mean = gmm.means_[component_index]
                    std = np.sqrt(gmm.covariances_[component_index])
                    chosen_intersection = mean + 3 * std


            else:
                # No intersections found between means, choose the mean + 3 std of the component with the maximum weight
                component_index = np.argmax(gmm.weights_)
                mean = gmm.means_[component_index]
                std = np.sqrt(gmm.covariances_[component_index])
                chosen_intersection = mean + 3 * std

    plt.axvline(x=chosen_intersection, color='g', linestyle='--', label='Intersection')
    plt.plot(x, sum_pdf, color='r')
    plt.axvline(x=threshold, color='r', linestyle='--', label='Manual')

    plt.xlabel('Data')
    plt.ylabel('Probability')
    plt.title('Histogram of data and EM simulation')

    # Save the figure to a file, and close it
    plt.savefig(os.path.join(plot_path, compareflag + 'Simulation.png'))
    plt.close()

    if type(chosen_intersection) != float:
        chosen_intersection = chosen_intersection.item()

    return chosen_intersection, intersections


def rosin_threshold(data):
    # Ensure data is numpy array
    data = np.array(data)

    # Create histogram
    hist, bins = np.histogram(data, bins=256)

    # Get maximum point of histogram (peak)
    max_hist = np.argmax(hist)
    max_val = hist[max_hist]

    # Get last non-zero element (end)
    end_hist = np.where(hist > 0)[0][-1]
    end_val = hist[end_hist]

    # Create line from peak to end
    line = np.linspace(max_val, end_val, end_hist - max_hist)

    # Get distances between line and histogram
    distances = line - hist[max_hist:end_hist]

    # Get maximum distance
    max_dist_idx = np.argmax(distances)
    max_dist = distances[max_dist_idx]

    # Return threshold
    return bins[max_hist + max_dist_idx]


def threshold_log(data):
    col_log = np.log(data)
    total_cells = len(col_log)
    col_log = col_log[col_log > -np.inf]  # only keep values > -inf
    cells_after_removal = len(col_log)
    dropped_cells = total_cells - cells_after_removal
    return col_log, dropped_cells, total_cells


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)  # this returns both correlation and p-value
    # Calculate Spearman's rank correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)

    return r2, rmse, corr, spearman_corr, mae

def extract_marker_name(column_name):
    # Assuming the column name format is consistent
    marker_name = column_name.split('(')[0].strip()  # Extract the marker name
    # Remove unnecessary prefixes like "Membrane" or "Nucleus"
    prefixes = ['Membrane', 'Nucleus']
    for prefix in prefixes:
        marker_name = marker_name.replace(prefix, '').strip()
    return marker_name


def bland_altman_plot(data1, data2, save_dir, column_name, method):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    marker_name = extract_marker_name(column_name)  # Extract marker name

    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(mean, diff)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean Score')
    plt.ylabel('Difference Score')
    plt.title(f'{method}-Bland-Altman Plot-{marker_name}')
    plt.savefig(os.path.join(save_dir, f'BlandAltman_{marker_name}_{method.replace(" ", "_")}.png'))
    plt.close()


def binarize_data(data, threshold):
    return np.where(data >= threshold, 1, 0)


def GHT(data, nu=0, tau=0, kappa=0, omega=0.5):
    n, bin_edges = np.histogram(data, bins=256)  # 'auto' lets numpy decide the number of bins
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    csum = lambda z: np.cumsum(z)[:-1]
    dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
    argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
    clip = lambda z: np.maximum(1e-30, z)
    """Some math that is shared across multiple algorithms."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2

    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    v0 = clip((p0 * nu * tau ** 2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau ** 2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa * omega) * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)


    return argmax(x, f0 + f1)


def get_manual_phenotype(PDL1_stat, TIM3_stat):
    if PDL1_stat == 'POS' and TIM3_stat == 'POS':
        return 'PDL1+TIM3+'
    elif PDL1_stat == 'POS':
        return 'PDL1+ONLY'
    elif TIM3_stat == 'POS':
        return 'TIM3+ONLY'
    else:
        return 'NULL'

def get_phenotype(row, PDL1_threshold, TIM3_threshold):

    PDL1 = row['Membrane PDL1 (Opal 650) Mean (Normalized Counts, Total Weighting)']
    TIM3= row['Nucleus TIM3 (Opal 690) Mean (Normalized Counts, Total Weighting)']

    if PDL1 >= PDL1_threshold and TIM3 >= TIM3_threshold:
        return 'PDL1+TIM3+'
    elif PDL1 >= PDL1_threshold:
        return 'PDL1+ ONLY'
    elif TIM3 >= TIM3_threshold:
        return 'TIM3+ONLY'
    else:
        return 'NULL'


def main():
    folder_path = r'D:\Chang_files\workspace\data\Final CSV files G3 samples\V2\slides'
    cell_paths_add_threds = r'D:\Chang_files\workspace\data\Final CSV files G3 samples\V2\slides\addThreds'
    save_dir = r'D:\Chang_files\work_records\G3_thresholding_comparision\Metrics_results'
    slide_dir = r'D:\Chang_files\work_records\G3_thresholding_comparision\slide_results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    count_file = open(os.path.join(save_dir, 'cell_count_info.txt'), 'w')
    metrics_file = open(os.path.join(save_dir, 'metrics.txt'), 'w')

    column_names = ['Membrane PDL1 (Opal 650) Mean (Normalized Counts, Total Weighting)',
                    'Nucleus TIM3 (Opal 690) Mean (Normalized Counts, Total Weighting)']
    threshold_column_names = ['Threshold Membrane PDL1', 'Threshold Nucleus TIM3']

    methods = ['Otsu Threshold', 'BGMM Threshold', 'Otsu Log Threshold', 'Rosin Threshold',
               'filtered_bgmm_threshold_raw Threshold', 'filtered_bgmm_threshold_log Threshold',
               'filtered_otsu_threshold_raw Threshold', 'filtered_otsu_threshold_log Threshold',
               'filtered_Rosin Threshold','GHT Threshold', 'filtered_GHT_raw Threshold']
    result = {method: {column_name: [] for column_name in column_names} for method in methods}
    binary_result = {method: {column_name: [] for column_name in column_names} for method in methods}
    manual_result = {column_name: [] for column_name in column_names}


    for csv_file in tqdm(os.listdir(folder_path)):
        if csv_file.endswith('.txt'):
            print(csv_file)

            file_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(file_path)
            slide = os.path.splitext(csv_file)[0]

            slide_path = os.path.join(slide_dir, slide)
            if not os.path.exists(slide_path):
                os.makedirs(slide_path)

            for column_name, threshold_column_name in zip(column_names, threshold_column_names):
                print(column_name)
                np.random.seed(3)
                sample_data = np.random.choice(df[column_name], 1000)
                sample_data = median_filter(sample_data, size=5)
                sample_data = uniform_filter1d(sample_data, size=5)
                sample_data_log, dropped_cells, total_cells = threshold_log(sample_data)

                count_file.write(
                    f'{slide} - {column_name} - Dropped cells: {dropped_cells}, Total cells: {total_cells}\n')

                manual_threshold = df[threshold_column_name][0]
                otsu_threshold_raw = threshold_otsu(sample_data)
                otsu_threshold_log = threshold_otsu(sample_data_log.reshape(-1,1))
                print('raw')
                bgmm_threshold_raw, candidate_thresholds_raw = bgmm_threshold_mean(sample_data,slide_path,'raw '+ column_name, manual_threshold)
                print('log')
                bgmm_threshold_log, candidate_thresholds_log = bgmm_threshold_mean(sample_data_log.reshape(-1,1),slide_path,'log ' + column_name, np.log(manual_threshold))
                rosin_threshold_raw = rosin_threshold(sample_data)
                GHT_threshold_raw = GHT(data=sample_data)

                # Filter the data above the Otsu threshold
                filtered_sample_data = sample_data[sample_data >= otsu_threshold_raw]
                filtered_sample_data_log, _, _ = threshold_log(filtered_sample_data)
                print('raw filter')
                filtered_bgmm_threshold_raw, candidate_thresholds_raw = bgmm_threshold_mean(filtered_sample_data, slide_path,
                                                                                   'raw filter ' + column_name,
                                                                                   manual_threshold)
                filtered_otsu_threshold_raw = threshold_otsu(filtered_sample_data)
                filtered_Rosin_threshold_raw = rosin_threshold(filtered_sample_data)
                filtered_GHT_threshold_raw = GHT(data=filtered_sample_data)
                print('log filter')
                filtered_bgmm_threshold_log, candidate_thresholds_log = bgmm_threshold_mean(filtered_sample_data_log.reshape(-1, 1),
                                                                                   slide_path, 'log filter ' + column_name,
                                                                                   np.log(manual_threshold))
                filtered_otsu_threshold_log = threshold_otsu(filtered_sample_data_log)

                # convert the thresholds into their original space
                bgmm_threshold_log = np.exp(bgmm_threshold_log)
                otsu_threshold_log = np.exp(otsu_threshold_log)
                filtered_bgmm_threshold_log = np.exp(filtered_bgmm_threshold_log)
                filtered_otsu_threshold_log = np.exp(filtered_otsu_threshold_log)

                # Plot histogram of the original data with thresholds
                plt.figure(figsize=(6, 6), dpi=300)
                plt.hist(sample_data, bins=50, color='lightblue', edgecolor='black')
                plt.axvline(x=manual_threshold, color='r', linestyle='--', label='Manual')
                plt.axvline(x=otsu_threshold_raw, color='b', linestyle='--', label='Otsu')
                plt.axvline(x=otsu_threshold_log, color='g', linestyle='--', label='Otsu Log')
                plt.axvline(x=bgmm_threshold_log, color='y', linestyle='--', label='BGMM Log')
                plt.axvline(x=bgmm_threshold_raw, color='c', linestyle='--', label='BGMM')
                plt.axvline(x=rosin_threshold_raw, color='m', linestyle='--', label='Rosin')
                plt.axvline(x=filtered_bgmm_threshold_raw, color='purple', linestyle='--',
                            label='filtered_bgmm_threshold_raw')
                plt.axvline(x=filtered_bgmm_threshold_log, color='orange', linestyle='--',
                            label='filtered_bgmm_threshold_log')
                plt.axvline(x=filtered_otsu_threshold_raw, color='k', linestyle='--',
                            label='filtered_otsu_threshold_raw')
                plt.axvline(x=filtered_otsu_threshold_log, color='pink', linestyle='--',
                            label='filtered_otsu_threshold_log')
                plt.axvline(x=filtered_Rosin_threshold_raw, color='yellow', linestyle='--',
                            label='filtered_rosin_threshold')
                plt.axvline(x=GHT_threshold_raw, color='magenta', linestyle='--',
                            label='GHT_threshold')
                plt.axvline(x=filtered_GHT_threshold_raw, color='purple', linestyle='--',
                            label='filtered GHT_threshold')



                plt.legend()
                title = f'Original data - {slide} - {column_name}'
                plt.title("\n".join(textwrap.wrap(title, 60)))
                plt.savefig(os.path.join(slide_path, f'original_all_thresholds-{column_name}.png'))
                plt.close()

                # Plot histogram of the log data with thresholds
                plt.figure(figsize=(6, 6), dpi=300)
                plt.hist(sample_data_log, bins=50, color='lightblue', edgecolor='black')
                plt.axvline(x=np.log(manual_threshold), color='r', linestyle='--', label='Manual')
                plt.axvline(x=np.log(otsu_threshold_raw), color='b', linestyle='--', label='Otsu')
                plt.axvline(x=np.log(bgmm_threshold_raw), color='c', linestyle='--', label='BGMM')
                plt.axvline(x=np.log(otsu_threshold_log), color='g', linestyle='--', label='Otsu Log')
                plt.axvline(x=np.log(bgmm_threshold_log), color='y', linestyle='--', label='BGMM Log')
                plt.axvline(x=np.log(rosin_threshold_raw), color='m', linestyle='--', label='Rosin')
                plt.axvline(x=np.log(filtered_bgmm_threshold_raw), color='purple', linestyle='--',
                            label='filtered_bgmm_threshold_raw')
                plt.axvline(x=np.log(filtered_bgmm_threshold_log), color='orange', linestyle='--',
                            label='filtered_bgmm_threshold_log')
                plt.axvline(x=np.log(filtered_otsu_threshold_raw), color='k', linestyle='--',
                            label='filtered_otsu_threshold_raw')
                plt.axvline(x=np.log(filtered_otsu_threshold_log), color='pink', linestyle='--',
                            label='filtered_otsu_threshold_log')
                plt.axvline(x=np.log(filtered_Rosin_threshold_raw), color='yellow', linestyle='--',
                            label='filtered_rosin_threshold')
                plt.axvline(x=np.log(GHT_threshold_raw), color='magenta', linestyle='--',
                            label='GHT_threshold')
                plt.axvline(x=np.log(filtered_GHT_threshold_raw), color='purple', linestyle='--',
                            label='filtered GHT_threshold')

                plt.legend()
                title = (f'Log data - {slide} - {column_name}')
                plt.title("\n".join(textwrap.wrap(title, 60)))
                plt.savefig(os.path.join(slide_path, f'logarithm_all_thresholds-{column_name}.png'))
                plt.close()

                result['Otsu Threshold'][column_name].append((manual_threshold, otsu_threshold_raw))
                result['BGMM Threshold'][column_name].append((manual_threshold, bgmm_threshold_raw))
                result['Otsu Log Threshold'][column_name].append((manual_threshold, otsu_threshold_log))
                result['BGMM Log Threshold'][column_name].append((manual_threshold, bgmm_threshold_log))
                result['Rosin Threshold'][column_name].append((manual_threshold, rosin_threshold_raw))
                result['filtered_bgmm_threshold_raw Threshold'][column_name].append((manual_threshold, filtered_bgmm_threshold_raw))
                result['filtered_bgmm_threshold_log Threshold'][column_name].append((manual_threshold, filtered_bgmm_threshold_log))
                result['filtered_otsu_threshold_raw Threshold'][column_name].append(
                    (manual_threshold, filtered_otsu_threshold_raw))
                result['filtered_otsu_threshold_log Threshold'][column_name].append(
                    (manual_threshold, filtered_otsu_threshold_log))
                result['filtered_Rosin Threshold'][column_name].append(
                    (manual_threshold, filtered_Rosin_threshold_raw))
                result['GHT Threshold'][column_name].append(
                    (manual_threshold, GHT_threshold_raw))
                result['filtered_GHT_raw Threshold'][column_name].append(
                    (manual_threshold, filtered_GHT_threshold_raw))

                for method in methods:
                    thresholds = np.array(result[method][column_name][-1])
                    binary_manual = binarize_data(sample_data, thresholds[0])
                    binary_calculated = binarize_data(sample_data, thresholds[1])
                    binary_result[method][column_name].extend(binary_calculated)

                    # Add calculated thresholds to dataframe
                    df[f'{method}_Threshold_{column_name}'] = thresholds[1]

                manual_result[column_name].extend(binary_manual)
            # Add manual phenotype to dataframe
            df['MANUAL_Phenotype'] = df.apply(
                lambda row: get_phenotype(row, row['Threshold Membrane PDL1'], row['Threshold Nucleus TIM3']), axis=1)


            #df['MANUAL_Phenotype'] = df.apply(
                #lambda row: get_manual_phenotype(row['PDL1_stat'], row['TIM3_stat']), axis=1)
            # Save the updated dataframe to the original csv file
            # df.to_csv(os.path.join(cell_paths_add_threds, csv_file), index=False)
            # Select the specific columns to keep from the dataframe
            df = df[columns_to_keep + [f'{method}_Threshold_{column_name}' for method in methods for column_name in
                                       column_names]]

            # Save the updated dataframe to the original csv file
            csv_file = csv_file.replace('.txt', '.csv')
            df.to_csv(os.path.join(cell_paths_add_threds, csv_file), index=False)



    for column_name in column_names:
        for method in methods:
            thresholds = np.array([item for item in result[method][column_name]])
            r2, rmse, corr, spearman_corr, mae = calculate_metrics(thresholds[:, 0], thresholds[:, 1])
            kappa = cohen_kappa_score(manual_result[column_name], binary_result[method][column_name])

            plt.figure(figsize=(6, 6), dpi=300)
            plt.scatter(thresholds[:, 0], thresholds[:, 1], color='c', alpha=0.8, edgecolor='red')
            plt.xlabel('Manual Threshold')
            plt.ylabel('Calculated Threshold')
            title = (f'{method} vs manually thresholding results')
            plt.title("\n".join(textwrap.wrap(title, 60)))

            # Get the overall min and max
            overall_min = np.min(thresholds)
            overall_max = np.max(thresholds)

            # Plot x=y line
            x = np.linspace(overall_min, overall_max, 100)
            plt.plot(x, x, label='x=y', color='k', linestyle='--')

            # Fit a line to the scatter plot
            model = np.polyfit(thresholds[:, 0], thresholds[:, 1], 1)
            predicted = np.poly1d(model)

            # Create an array of x values that span the entire range
            x = np.linspace(overall_min, overall_max, 100)

            # Plot the fitted line
            plt.plot(x, predicted(x), color='r', label=f'y={model[0]:.2f}x+{model[1]:.2f}')
            plt.legend(loc='upper left')

            # Set the x and y limits
            plt.xlim(overall_min, overall_max)
            plt.ylim(overall_min, overall_max)

            # Ensure that one unit on the x-axis equals one unit on the y-axis
            plt.gca().set_aspect('equal', adjustable='box')

            plt.savefig(os.path.join(save_dir, f'{column_name.replace(" ", "_")}_{method.replace(" ", "_")}.png'))
            plt.close()

            bland_altman_plot(thresholds[:, 0], thresholds[:, 1], save_dir, column_name, method)

            metrics_file.write(f'{column_name} - {method}\n')
            metrics_file.write(f'R^2: {r2:.2f}, RMSE: {rmse:.2f}, Pearson: {corr:.2f}, '
                               f'Spearsmanr: {spearman_corr:.2f}, MAE: {mae:.2f}, CohensKappa:{kappa:.2f}\n\n')

    count_file.close()
    metrics_file.close()


if __name__ == '__main__':
    main()
