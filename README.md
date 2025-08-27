
# M³SB: Multi-Model Merging via Spherical Barycenters

This repository contains the code for the Deep Learning and Applied AI project "Multi-Model Merging via Spherical Barycenters." The `m3sb` package provides a suite of tools for implementing and analyzing various model merging techniques, including a spherical barycenter approach, linear interpolation, and pairwise SLERP. It also includes a robust experimental framework for evaluating the performance of merged models on multiple downstream tasks.

## Installation

To set up the environment and run the experiments, you will need Python 3.10+ and uv package manager.

1.  Clone the repository:
    ```bash
    git clone https://github.com/mehdiamlal/m3sb.git
    cd m3sb
    ```

2.  Sync the uv package:
    ```bash
    uv sync
    ```

## Package Structure

The project is organized into a Python package named `m3sb` with the following modules:

-   `m3sb/merging.py`: Contains the high-level logic for the different merging algorithms (Barycenter, Linear, Pairwise SLERP) and model reconstruction utilities.
-   `m3sb/geometry.py`: Implements the low-level geometric functions required for spherical operations (SLERP, Log Map, Exp Map).
-   `m3sb/eval.py`: Provides functions for evaluating model performance (accuracy, precision, recall, F1-score).
-   `m3sb/data.py`: Handles the loading and preprocessing of datasets from the Hugging Face Hub.
-   `m3sb/experiment.py`: Contains the `Experiment` and `TaskVectorExperiment` classes for automating experimental runs.
-   `m3sb/visualization.py`: Includes helper functions for generating PCA plots and various charts.
- `experiment-results`: contains the results of the experiments saved as csv files.

## Usage

**Note**: in order to run the experiments in a reasonable time, make sure you have access to a GPU.

The primary way to run experiments is by defining a configuration and using the `Experiment` or `TaskVectorExperiment` classes. This is typically done in a separate script or a Jupyter Notebook.

### Example: Running a 3-Model Merge Experiment

The following example demonstrates how to replicate the 3-model merge experiment (merging full model weights).

1.  **Create a main script** (e.g., `run_experiment.py`):


    ```python
    import pandas as pd
    from m3sb.experiment import Experiment

    # 1. Define the complete configuration for the experiment.
    exp_3_model_config = {
        "name": "3-Model Merge (Full Weights)",
        "model_checkpoints": [
            "pkr7098/cifar100-vit-base-patch16-224-in21k",
            "nateraw/food",
            "MaulikMadhavi/vit-base-flowers102"
        ],
        "datasets_config": [
            {"dataset_name": "cifar100", "split": "test", "image_col": "img", "label_col": "fine_label"},
            {"dataset_name": "food101", "split": "validation", "image_col": "image", "label_col": "label"},
            {"dataset_name": "nkirschi/oxford-flowers", "split": "test", "image_col": "image", "label_col": "label"}
        ],
        "merge_configs": {
            "barycenter": {"weights": [0.3333, 0.3334, 0.3333]},
            "linear": {"weights": [0.3333, 0.3334, 0.3333]},
            "pairwise_slerp": {"weights": [0.3333, 0.3334, 0.3333]}
        },
        "base_model_checkpoint": "google/vit-base-patch16-224-in21k"
    }

    # 2. Create an instance of the runner and execute the experiment.
    experiment = Experiment(**exp_3_model_config)
    experiment.run()

    # 3. Get and display the final results as a pandas DataFrame.
    results_df = experiment.get_results_df()
    print("\n--- Final Results ---")
    print(results_df)

    # Optionally, save the results to a CSV file
    results_df.to_csv("experiment_3_model_results.csv", index=False)
    ```

2.  **Run the script from your terminal:**

    ```bash
    uv run run_experiment.py
    ```

This will execute the full experimental pipeline, including loading the models, performing the three types of merges, evaluating each merged model on all three datasets, and printing a final summary of the results.

### Reproducing Report Results
To reproduce you can run the experiments in [this Kaggle Notebook](https://www.kaggle.com/code/mehdiamlal/m3sb-experiments/).

Or you can download and executes the `experiments.ipynb` notebook.