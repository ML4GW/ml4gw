import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm


def load_data_from_h5(filename):
    """
    Loads data from an HDF5 file while preserving the original structure.
    args:
        filename (str): The path to the HDF5 file.
    returns:
        data (dict): A dictionary containing the data from the HDF5 file.
    """
    data = {}
    with h5py.File(filename, "r") as f:
        for group_key in tqdm(f.keys()):  # Iterate over singular tests
            h5_group = f[group_key]
            if group_key not in data:
                data[group_key] = {}
            # Adds group_key (params and errors) to data if it doesn't exist
            for dataset_key in h5_group.keys():
                dataset = h5_group[dataset_key]  # add group do data dict
                if dataset.shape == ():  # Check if the dataset is a scalar
                    data[group_key][dataset_key] = np.array([dataset[()]])
                else:
                    data[group_key][dataset_key] = dataset[:]
    return data


def plot_err_chirp_mass(err, chirp_mass, mass_ratio, file):
    """
    Plots the error data as a 2d historgram with chirp mass and
    mass ratio as the axis.
    """
    # Make sure the data is 1D
    chirp_mass = np.asarray(chirp_mass).reshape(-1)
    mass_ratio = np.asarray(mass_ratio).reshape(-1)
    err = np.asarray(err).reshape(-1)

    # Make plots
    plt.figure(figsize=(10, 5))
    cmap = plt.get_cmap("viridis")
    min_color = cmap(0.0)
    cmap.set_under(min_color)
    cmap.set_bad(min_color)
    plt.hist2d(
        chirp_mass,
        mass_ratio,
        bins=250,
        norm=LogNorm(),
        weights=err,
        cmap=cmap,
    )
    plt.colorbar(label="Error")
    plt.xlabel("Chirp Mass")
    plt.ylabel("Mass Ratio")
    plt.title("Histogram of Differences between lal and ml4gw waveforms")
    plt.suptitle(f"Total number of differences ({len(err)})")
    plt.show()
    plt.savefig(f"benchmark_plots/{file}_histogram.png")
    plt.close()


def plot_tests(
    tests,
    plotting_function=plot_err_chirp_mass,
    plotting_keys=["chirp_mass", "mass_ratio"],
):
    """
    Function to plot data using plot_err_chirp_mass.
    args:
        tests (list): List of tests to plot.
        (Uses these strings to find the files)
    returns:
        None
    """
    # Loop through provided tests
    for test in tests:
        # Change this based on where data is stored and what it is called
        file_prefix = "benchmark_data"
        folder = "benchmark_data"
        num_files = 2  # Number of data files to include in plots,
        # 0 defaults to all files
        start_file = 0  # Start file index

        segmented_data = (
            {}
        )  # Dictionary to hold segmented data as each file is loaded-
        # seperately before conjoining into one dataset

        # Find all files matching the pattern if num_files = 0
        if num_files == 0:
            index = 0
            while os.path.exists(
                f"{folder}/{file_prefix}_{test}_{index}.h5"
            ):  # Find all files with the test name
                num_files += 1
                index += 1

            if num_files == 0:
                continue  # Skip to the next test

        # Load data from each file
        for i in range(num_files):
            filename = f"{folder}/{file_prefix}_{test}_{i + start_file}.h5"
            file_data = load_data_from_h5(filename)
            if i not in segmented_data:
                segmented_data[i] = {}
            segmented_data[i].update(file_data)

        # Find all the error keys to plot using file names
        # Data files must have err in the file name or program will not-
        # recognize
        err_keys = []
        for file_key in segmented_data:
            for dataset_key in segmented_data[file_key].keys():
                for key in segmented_data[file_key][dataset_key].keys():
                    if "err" in key and key not in err_keys:
                        err_keys.append(key)

        # Loop over segmented data to compile into one dataset using each-
        # unique err_key
        for err_key in err_keys:
            # Initialize data
            data = {}
            # Loop over segmented data to compile all datsets that contain-
            # specific err_key
            for file_key in segmented_data:
                for dataset_key in segmented_data[file_key].keys():
                    if err_key in segmented_data[file_key][dataset_key].keys():
                        for key in segmented_data[file_key][dataset_key]:
                            if key not in data:
                                data[key] = []
                            data[key].extend(
                                segmented_data[file_key][dataset_key][key]
                            )
            plotting_function(
                data[err_key],
                data[plotting_keys[0]],
                data[plotting_keys[1]],
                f"{test}_{err_key}",
            )


# Example Usage
tests = ["phenom_p", "phenom_d"]

plot_tests(
    tests,
    plotting_function=plot_err_chirp_mass,
    plotting_keys=["chirp_mass", "mass_ratio"],
)
