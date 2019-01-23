# Training Procedure for Predictive Modelling of Optimal Parameters in `libcusmm`

The performance of the matrix-matrix multiplication kernels is highly dependant on the choice of algorithm and parameters, this is why [*autotuning*](https://www.cp2k.org/howto:libcusmm) is used to find optimal kernel parameters.

However, the autotuning procedure is expensive, and the space of (m,n,k)-triplets to explore is large. This predictive modeling procedure is set up to predict optimal parameters for (m,n,k)-triplets that have not been autotuned from the data gathered from autotuning other (m,n,k)-triplets.

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

![libcusmm_predictive_modeling_features](../../../../docs/images/libcusmm_predictive_modeling_features.png)



---

### Predictive modeling procedure

#### 1. Get the data

Get the data to be used for training, either by downloading data from the [dedicated repository](https://github.com/cp2k/dbcsr-data), or by autotuning new kernels yourself and combining them with pre-existing data.



##### 1.a Download pre-collected data from dedicated repository

- Download data from the dedicated repository:

  ```%bash
  wget https://github.com/cp2k/dbcsr-data/blob/master/GPU/raw_training_data_ALGORITHM.csv
  ```

- Compute derived parameters from raw parameters and create a record of baseline and maximum performances: run [`predict_derivepars.py`](predict_derivepars.py) , providing the CUDA architecture number and the location of the downloaded data:

  ```%bash
  ./predict_derivepars.py # –arch 60 --folder /scratch/autotuning_dataset, e.g.
  ```



##### 1.b (optional) Aquire data from autotuning

- We would appreciate if you would upload the data resulting from your autotuning procedure to the [dedicated repository](https://github.com/cp2k/dbcsr-data). For this, please take note, at this stage, of the [information required to upload your data](https://github.com/cp2k/dbcsr-data/blob/master/git-commit.template).

- If you're autotuning data for a new GPU, make sure that the GPU's compute architecture properties are given in the file [`kernels/gpu_properties.json`](kernels/gpu_properties.json). If not, please add them.

- Follow the [instructions for autotuning](https://www.cp2k.org/howto:libcusmm).

- If all went well, you now have directories named `tune_mxnxk` containing log files in which parameter sets and their corresponding measured performances are recorded.

- Collect the information in all the `tune_mxnxk` directories into CSV files: run [`predict_collect.py`](predict_collect.py), providing the CUDA architecture number and the location of the autotuning data:

  ```%bash
  ./predict_collect.py # –arch 60 --folder /scratch/autotuning_dataset, e.g.
  ```

- Follow the instructions given at the end to merge the CSV files.

##### At the end, you should end up with the following files:

- `raw_training_data_ALGORITHM.csv`  (containing all *raw* parameters for training a model for algorithm ALGORITHM)
- `training_data_ALGORITHM.csv` (containing all *derived* parameters for training a model for algorithm ALGORITHM)



#### 2. (optional) Explore the data

Explore the data interactively using the [provided jupyter notebook](notebooks/inspect_training_data.ipynb).



#### 3. Train

For each algorithm, build a predictive model using decision trees and feature selection based on the features' permutation importance. 


```%bash
./predict_train.py  # –algo medium --folder /scratch/autotuning_dataset, e.g.
```

Repeat this step for all algorithms.
This may take several hours. For example, training algorithm 'medium' for the P100 took 11 hours on a single Greina (CSCS) node.
Moreover, depending on the size of the training data, large amounts of memory may be needed. For example, training algorithm 'medium' for the P100 was run on a 192 GB node.



#### 4. Generate optimal parameters

Given predictive models (in the form of serialized [scikit-learn](https://scikit-learn.org/) model objects) for all unseen (m,n,k)s, generate or update a file of optimal parameters

```%bash
./predict_genpars.py  -c 5000 \  # chunk size
    --largeDB2 /scratch/largeDB2/feature_tree_refit.p \ # path to models
    --largeDB1 /scratch/largeDB1/feature_tree_refit.p \
    --medium /scratch/medium/feature_tree_refit.p \
    --small /scratch/small/feature_tree_refit.p \
    --tiny /scratch/tiny/feature_tree_refit.p
```

This may take several hours. For example, generating parameters for the P100 took 8 hours on a single Piz Daint (CSCS) node.



#### 5. Evaluate the predicted parameters

```%bash
./predict_evaluate.py -f libcusmm_predicted.out -n libcusmm_baseline.out
```



#### 6. Contribute your new parameters and data

##### Contribute training data

See [instructions](https://github.com/cp2k/dbcsr-data#contributing) in our [dedicated repository](https://github.com/cp2k/dbcsr-data)

##### Contribute predicted parameters

Submit a pull request updating the `parameters_GPU.json` file in question.



---

### Contributing to the training procedure

##### Adding a new predictive feature

- Choose the new feature's name
- Add the feature as a method of `class PredictiveParameters`, named `get_NAME`
- Add the derived feature to the data structure `derived_parameters` in [`kernels/cusmm_predict.py`](kernels/cusmm_predict.py)
