from bptt.models import Model
import torch as tc
from evaluation.pse import power_spectrum_error
from evaluation.klx import klx_metric
from evaluation.klx_gmm import calc_kl_from_data

import numpy as np
import pickle
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import List,Tuple, Callable, Any, Dict

import json

def convert_dict_values(orig_dict: Dict[str, List[tc.Tensor]]) -> Dict[str, List[float]]:
    new_dict = {}
    for key, value in orig_dict.items():
        new_dict[key] = [np.float64(x.item()) for x in value]
    return new_dict

def save_dict_to_file(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a JSON file at the specified file path.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str): The path to save the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file)

def load_dict_from_file(file_path: str) -> dict:
    """
    Load a dictionary from a JSON file at the specified file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded dictionary.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def flatten_and_unique(nested_list: List[List[str]]) -> List[str]:
    """
    Flattens a nested list of strings and returns a list with unique elements.

    Args:
        nested_list (List[List[str]]): A nested list of strings.

    Returns:
        List[str]: A flattened list with unique elements.
    """
    flattened_list = [item for sublist in nested_list for item in sublist]
    unique_list = list(set(flattened_list))

    return unique_list

def tuple_index_list(input_list: List, indices: Tuple) -> List:
    """
    Retrieves elements from the input_list at the specified tuple indices.

    Args:
        input_list (List): The list to be indexed.
        indices (Tuple): A tuple of indices.

    Returns:
        List: A list containing the elements at the specified indices.
    """
    return [input_list[i] for i in indices]


def concat_arrays(arrays):
    """
    Concatenate a list of numpy arrays and return the length of each original array.

    Args:
        arrays: A list of numpy arrays.

    Returns:
        A tuple containing the concatenated numpy array and a list of lengths of each original array.
    """
    # Concatenate the arrays along the first dimension
    concatenated = np.concatenate(arrays, axis=0)

    # Get the length of each original array
    lengths = [a.shape[0] for a in arrays]

    return concatenated, lengths


def deconcatenate(arr, lengths):
    """
    Deconcatenates an array given the lengths of the original arrays.

    Parameters:
        arr (numpy.ndarray): The concatenated array.
        lengths (list): A list of integers representing the lengths of the original arrays.

    Returns:
        list: A list of numpy arrays with the specified lengths.
    """
    assert sum(lengths) == len(arr), "Total length of lengths does not match length of array."
    deconcatenated = []
    index = 0
    for length in lengths:
        deconcatenated.append(arr[index:index+length])
        index += length
    return deconcatenated


def compare_data(real_data: Any, generated_data: Any, comparison_function: Callable[[Any, Any], Any]) -> Any:
    """
    Compares real data with model-generated data using a provided comparison function.

    Args:
        real_data (Any): The real data to be compared.
        generated_data (Any): The model-generated data to be compared.
        comparison_function (Callable[[Any, Any], Any]): The function used to compare the real and generated data.

    Returns:
        Any: The result of the comparison.
    """

    return comparison_function(real_data, generated_data)

# Example usage:

def example_comparison_function(real_data: list, generated_data: list) -> float:
    # Implement your comparison logic here.
    # This is a simple example that calculates the mean squared error between two lists.
    mse = sum([np.nanmean((real - generated) ** 2) for real, generated in zip(real_data, generated_data)]) / len(real_data)
    return mse

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def create_boxplot_html(data: Dict[str, list], output_file: str = 'boxplot.html') -> None:
    """
    Creates an HTML boxplot visualization of the input dictionary using Plotly.

    Args:
        data (Dict[str, list]): The input dictionary with keys representing categories and values as lists of data points.
        output_file (str, optional): The output HTML file name. Defaults to 'boxplot.html'.
    """

    # Sort the dictionary items based on the mean of their values
    sorted_data = sorted(data.items(), key=lambda item: np.nanmedian(item[1]))

    # Create a subplot with boxplots for each key in the sorted dictionary
    # fig = make_subplots(rows=1, cols=len(data), subplot_titles=[key for key, _ in sorted_data])
    fig = make_subplots(rows=1, cols=len(data))

    # Find the minimum and maximum values across all data
    min_value = min([min(values) for values in data.values()])
    max_value = max([max(values) for values in data.values()])

    for index, (key, values) in enumerate(sorted_data, start=1):
        fig.add_trace(go.Box(y=values, name=key, showlegend=False), row=1, col=index)

        # Update the y-axis range for each subplot
        fig.update_yaxes(range=[min_value, max_value], row=1, col=index)

    # Update layout
    fig.update_layout(title="Boxplot Visualization", height=600, width=len(data) * 300,
                      yaxis_title="MSE")

    # Save the figure as an HTML file
    pio.write_html(fig, file=output_file, auto_open=False)




do = 1
data_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/PM_test_great/'  # experiment
data_sets = os.listdir(data_path)  # data sets
n_datasets = len(data_sets)  # amount datasets
pm_configs = flatten_and_unique([os.listdir(os.path.join(data_path, i)) for i in data_sets])
n_pmconfigs = len(pm_configs)

data_set_path = r'/zi-flstorage/Max.Thurm/PhD/nonstationary-trial-data-autoencoder/analysis_data/AE_test_data/'
res_file_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/PM_test_great_plots/res_dic_mse.json'

if not os.path.exists(res_file_path) and do:
    res_dic = {}

    # Iterate through all data_set directories
    for data_set_name in data_sets:
        data_set_dir = os.path.join(data_path, data_set_name)


        #Load dataset
        ds_list = np.load(os.path.join(data_set_path, data_set_name + '.npy'), allow_pickle=True)
        in_list = np.load(os.path.join(data_set_path, data_set_name[:-4] + 'inputs.npy'), allow_pickle=True)
        

        # Iterate through all pm_config directories inside each data_set directory
        for pm_config_name in pm_configs:
            pm_config_dir = os.path.join(data_set_dir, pm_config_name)

            if not pm_config_name in res_dic.keys():
                res_dic[pm_config_name] = []

            # Check if pm_config_dir exists
            if not os.path.exists(pm_config_dir):
                continue

            # Find all 'model.pt' files inside each pm_config directory
            model_pt_files = glob.glob(os.path.join(pm_config_dir, "*"))

            # Iterate through all found 'model.pt' files
            for model_pt_file in model_pt_files:
                run_dir = os.path.dirname(model_pt_file)
                run_name = os.path.basename(run_dir)

                # Load the 'model.pt' file
                model = Model()
                model.init_from_model_path(model_pt_file, 170000)

                gen_data = []
                for i, (trial, inp) in enumerate(zip(ds_list, in_list)):
                    trial = tc.tensor(trial, dtype=tc.float32)
                    inp = tc.tensor(inp, dtype=tc.float32)
                    generated_data = model.generate_free_trajectory(trial, inp, trial.shape[0], i)[0].detach().numpy()
                    gen_data.append(generated_data)
                    print(i/len(ds_list))

                true_data, L = concat_arrays(ds_list)
                gene_data, L = concat_arrays(gen_data)

                true_data = tc.tensor(true_data, dtype=tc.float32)
                gene_data = tc.tensor(gene_data, dtype=tc.float32)

                # result = compare_data(ds_list, gen_data, example_comparison_function)
                result = compare_data(gene_data, true_data, calc_kl_from_data)
                print(f"Comparison result: {result}")

                res_dic[pm_config_name].append(result)


                print(f"Loaded {model_pt_file} for {data_set_name}/{pm_config_name}/{run_name}")

    #save dic as .json
    save_dict_to_file(res_dic, res_file_path)
else:
    res_dic = load_dict_from_file(res_file_path)


#all
out_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/PM_test_great_plots/boxplot_all.html'
create_boxplot_html(res_dic, out_path)

# keys = list(res_dic.keys())

# #enc dim = 3
# new_keys = [i for i in keys if 'EnCdim_03' in i]

# #activation
# ndic = {'ReLU':[],'Tanh':[],'GELU':[]}
# for nk in new_keys:
#     if 'ReLU' in nk:
#         ndic['ReLU'].append(res_dic[nk])
#     elif 'Tanh' in nk:
#         ndic['Tanh'].append(res_dic[nk])
#     elif 'GELU' in nk:
#         ndic['GELU'].append(res_dic[nk])

# for i, val in ndic.items():
#     ndic[i] = [item for sublist in val for item in sublist]

# out_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/AE_PM_res_plots/boxplot_3_actfn.html'
# create_boxplot_html(ndic, out_path)

# #activation
# ndic = {'1':[],'2':[],'3':[]}
# for nk in new_keys:
#     if 'nlayers_01' in nk:
#         ndic['1'].append(res_dic[nk])
#     elif 'nlayers_02' in nk:
#         ndic['2'].append(res_dic[nk])
#     elif 'nlayers_03' in nk:
#         ndic['3'].append(res_dic[nk])

# for i, val in ndic.items():
#     ndic[i] = [item for sublist in val for item in sublist]

# out_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/AE_PM_res_plots/boxplot_3_layers.html'
# create_boxplot_html(ndic, out_path)

# #total dim 3
# ndic = {i : res_dic[i] for i in new_keys}
# out_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/AE_PM_res_plots/boxplot_3.html'
# create_boxplot_html(ndic, out_path)

# #activation
# ndic = {'3':[],'5':[],'10':[]}
# for nk in keys:
#     if 'EnCdim_03' in nk:
#         ndic['3'].append(res_dic[nk])
#     elif 'EnCdim_05' in nk:
#         ndic['5'].append(res_dic[nk])
#     elif 'EnCdim_10' in nk:
#         ndic['10'].append(res_dic[nk])

# for i, val in ndic.items():
#     ndic[i] = [item for sublist in val for item in sublist]

# out_path = r'/zi-flstorage/Max.Thurm/PhD/Paper1_BPTT/AE_PM_res_plots/boxplot_enc_dim.html'
# create_boxplot_html(ndic, out_path)




print()

