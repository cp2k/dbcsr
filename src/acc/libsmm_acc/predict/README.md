# Training Procedure for Predictive Modelling of Optimal Parameters in `libsmm_acc`

The performance of the matrix-matrix multiplication kernels is highly dependent on the choice of algorithm and parameters, this is why [*autotuning*](tune.md) is used to find optimal kernel parameters.

However, the autotuning procedure is expensive, and the space of (m,n,k)-triplets to explore is large. The following predictive modeling procedure is set up to predict optimal parameters for (m,n,k)-triplets that have not been autotuned from the data gathered from autotuning other (m,n,k)-triplets.

---

### Requirements

Python version required: `python 3.6`

Install all python packages required (if you do not want this project's requirements to interfere with your other Python projects, consider doing so in a [virtual environment](https://docs.python.org/3/tutorial/venv.html)), using

```%bash
pip install -r requirements.txt
```

---

### Predictive parameters

The input features for the predictive models can be 'raw' parameters (left-most-column in the figure below), or hand-engineered features 'derived' from the raw features (matrix sizes, launch parameters and resource usage estimations).

![libsmm_acc_predictive_modeling_features](../../../../../docs/images/libsmm_acc_predictive_modeling_features.png)
---

### Predictive modeling procedure

#### 1. Get the data

Get the data to be used for training, either by downloading data from the [dedicated repository](https://github.com/cp2k/dbcsr-data), or by autotuning new kernels yourself and combining them with pre-existing data.

##### 1.a Download pre-collected data from dedicated repository

- Download data from the dedicated repository:

  ```%bash
  wget https://github.com/cp2k/dbcsr-data/blob/master/GPU/raw_training_data_ALGORITHM.csv  # for ALGORITHM = tiny, small, medium, largeDB1, largeDB2
  ```

- Compute derived parameters from raw parameters and create a record of baseline and maximum performances: run [`prepare_training_data.py`](prepare_training_data.py), providing the CUDA/HIP architecture number and the location of the downloaded data:

  ```%bash
  ./prepare_training_data.py # –arch 60 --folder /scratch/autotuning_dataset, e.g.
  ```

##### 1.b (optional) Aquire data from autotuning

- We would appreciate if you would upload the data resulting from your autotuning procedure to the [dedicated repository](https://github.com/cp2k/dbcsr-data). For this, please take note, at this stage, of the [information required to upload your data](https://github.com/cp2k/dbcsr-data/blob/master/git-commit.template).

- If you're autotuning data for a new GPU, make sure that the GPU's compute architecture properties are given in the file [`kernels/gpu_properties.json`](kernels/gpu_properties.json). If not, please add them.

- Follow the [instructions for autotuning](tune.md).

- If all went well, you now have directories named `tune_mxnxk` containing log files in which parameter sets and their corresponding measured performances are recorded.

- Collect the information in all the `tune_mxnxk` directories into CSV files: run [`predict_collect.py`](predict_collect.py), providing the location of the autotuning data:

  ```%bash
  ./predict_collect.py # --folder /scratch/autotuning_dataset, e.g.
  ```

You should now have 5 CSV files containing raw data (`raw_training_data_ALGORITHM.csv`, for `ALGORITHM = tiny, small, medium, largeDB1, largeDB2`)

#### 2. Prepare the data for predictive modeling

A few steps are needed to make the data ready for training:

- Record maximum and baseline performances of (m,n,k)-triplets in JSON files
- Compute derived training data and write it to a CSV file
- Compress training data files from CSV to Parquet files

```%bash
./prepare_data.py  # --folder /scratch/autotuning_dataset -a 60 -j12, e.g. to run with 12 threads
```

The data preparation is relatively computationally expensive, especially for large data sets.
A good way of running it, is to

1. Compute just the maximum and baseline parameters for each algorithm separately (`-l ALGORITHM --skip_derived_data=True`), adjusting the `-j` parameter so it runs fast enough, while not running into "out-of-memory"-errors
2. Run again with `--skip_derived_data=True` to create the files that aggregate maximum and baseline performances for all algorithms.
3. Compute derived data records for each algorithm separately (`-l ALGORITHM`), adjusting the `-j` option.
4. Run the script again without specifying the algorithm nor skipping the derived data to make sure all necessary files have been generated.

##### At the end, you should end up with the following files:

- `raw_training_data_ALGORITHM.csv` (containing all *raw* parameters for training a model for algorithm ALGORITHM, obtained in step 1)
- `training_data_ALGORITHM.csv` (containing all *derived* parameters for training a model for algorithm ALGORITHM)
- `training_data_ALGORITHM.parquet` (containing all *raw* and *derived* parameters for training a model for algorithm ALGORITHM in Parquet files, convenient for reading in parallel using Dask)
- `baseline_performances_ALGORITHM.json` and `baseline_performances_by_algo.json` (containing, for each (m, n, k)-triplet in the training data, its baseline performance, i.e. its performance were it to be run with a set of parameters that are an expert's "best guess"). Additionally, the baseline performances are plotted in `baseline_performances.svg`.
- `maximum_performances_ALGORITHM.json`, `max_performances_by_algo.json` and `max_performances.json` (containing, for each (m, n, k)-triplet, its maximum performance). Additionally, the maximum performances are plotted in `maximum_performances.svg`.

#### 3. (optional) Explore the data

Explore the data interactively using the [provided Jupyter notebook](notebooks/inspect_training_data.ipynb).

#### 4. Train

For each algorithm, build a predictive model using decision trees and feature selection based on the features' permutation importance. 

```%bash
./predict_train.py  # --algo medium --folder /scratch/autotuning_dataset, e.g.
```

Use the command-line parameters `--folder` and `--destination_folder` to choose the folder from which data is read, as well as the folder to which models, logs, etc. are written.
Repeat this step for all algorithms.
This may take several hours. For example, training algorithm 'medium' for the P100 took 11 hours on a single Greina (CSCS) node.
Moreover, depending on the size of the training data, large amounts of memory may be needed. For example, training algorithm 'medium' for the P100 was run on a 192 GB node.

#### 5. Generate optimal parameters

Given predictive models (in the form of serialized [scikit-learn](https://scikit-learn.org/) model objects) for all unseen (m,n,k)s, generate or update a file of optimal parameters

```%bash
./predict_genpars.py  -c 5000 \  # chunk size
    -j 12 \ # 12 threads
    --largeDB2 /scratch/largeDB2/feature_tree_refit.p \ # path to models
    --largeDB1 /scratch/largeDB1/feature_tree_refit.p \
    --medium /scratch/medium/feature_tree_refit.p \
    --small /scratch/small/feature_tree_refit.p \
    --tiny /scratch/tiny/feature_tree_refit.p
```

This may take several hours. For example, generating parameters for the P100 took 8 hours on a single Piz Daint (CSCS) node. For this reason, intermediate results are stored in JSON files in a folder `predict_genpars_ckpt`. Once this scipt has finished running, and you've successfully obtained a new `parameters_GPU.json` file, you may delete the checkpoint folder `predict_genpars_ckpt`.

#### 6. Evaluate the predicted parameters

```%bash
./predict_evaluate.py -f libsmm_acc_predicted.out -n libsmm_acc_baseline.out
```

#### 7. Contribute your new parameters and data

##### Contribute training data

See [instructions](https://github.com/cp2k/dbcsr-data#contributing) in our [dedicated repository](https://github.com/cp2k/dbcsr-data)

##### Contribute predicted parameters

Submit a pull request updating the `parameters_GPU.json` file in question.

---

### Contributing to the training procedure

#### Adding a new predictive feature

- Choose the new feature's name, "`NAME`"
- Add the feature as a method of `class PredictiveParameters`, named `get_NAME`
- Add the derived feature to the data structure `derived_parameters` in [`kernels/smm_acc_predict.py`](kernels/smm_acc_predict.py)
