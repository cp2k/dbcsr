# Training Procedure for Predictive Modelling of Optimal Parameters in `libcusmm`

The performance of the matrix-matrix multiplication kernels is highly dependant on the choice of algorithm and parameters, this is why [*autotuning*](https://www.cp2k.org/howto:libcusmm) is used to find optimal kernel parameters.

Yet the autotuning procedure is expensive, and the space of (m,n,k)-triplets to explore is large. This predictive modelling procedure is there to predict optimal parameters for (m,n,k)-triplets that have not been autotuned from the data gathered from autotuning other (m,n,k)-triplets.



---

### Requirements

Python version required: `python 3.6`

Install all python packages required (if you do not want this project's requirements to interfere with your other Python projects, consider doing so in a [virtual environment](https://docs.python.org/3/tutorial/venv.html)), using

```%bash
pip install -r requirements.txt
```



---

### Predictive parameters

![libcusmm_predictive_modelling_features](../../../../docs/images/libcusmm_predictive_modelling_features.png)



---

### Predictive modelling procedure



#### 1. Get the data

Get the data to be used for training, either by downloading data from our dedicated repository, or autotuning new kernels yourself and combining them with pre-existing data. 



##### 1.a Download pre-collected data from dedicated repository

- Download data from the dedicated repository:

  ```%bash
  wget https://github.com/cp2k/dbcsr-data/BLAHBLAH
  ```

- Compute derived parameters from raw parameters and create a record of baseline and maximum performances. Run `predict_derivepars.py` , providing the CUDA architecture number and the location of the downloaded data:

  ```%bash
  ./predict_derivepars.py # –arch 60 --folder /scratch/snx3000/alicej/tune_dataset, e.g.
  ```



##### 1.b (optional) Aquire data from autotuning

- We would appreciate if you would upload the data resulting from your autotuning procedure to our dedicated repository. For this, please take note, at this stage, of the [information required to upload your data](https://github.com/cp2k/dbcsr-data/git-commit.template).

- If you're autotuning data for a new GPU, make sure that the GPU's compute architecture properties are given in the file ` 'kernels/gpu_properties.json'`. If not, please add them.

- Follow the [instructions for autotuning](https://www.cp2k.org/howto:libcusmm).

- If all went well, you now have directories named `tune_mxnxk` containing log files in which parameter sets and their corresponding measured performances are recorded.

- Collect the information in all the `tune_mxnxk` directories into CSV files. Run `predict_collect.py`, providing the CUDA architecture number and the location of the autotuning data:

  ```%bash
  ./predict_collect.py # –arch 60 --folder /scratch/snx3000/alicej/tune_dataset, e.g.
  ```

- Follow the instructions given at the end to merge the CSV files.

##### At the end, you should end up with the following files:

- `raw_training_data_ALGO.csv`  (containing all *raw* parameters for training a model for algorithm ALGO)
- `training_data_ALGO.csv` (containing all *derived* parameters for training a model for algorithm ALGO)



#### 2. (optional) Explore the data

Explore the data interactively using the [provided jupyter notebook](notebooks/inspect_training_data.ipynb).



#### 3. Train

For each algorithm, build a predictive model using Decision trees and feature selection based on features' permutation importance. 

Kept option to run with Random forests in case, but abandoned in first iter because doesn't bring much and makes models heavier in memory

```%bash
./predict_train.py  # options cf what I have on Piz Daint ... 
```



#### 4. Generate optimal parameters

Given predictive models for all unseen (m,n,k)s, generate or update a file of optimal parameters

```%bash
./predict_genpars.py
```

Predict_genpars.py
•    Cf job_genpars.sh
•    Time to solution: 08:17:52
•    



#### 5. Evaluate the predicted parameters

```%bash
./predict_evaluate.py
```



#### 6. Contribute your new parameters and data

##### Contribute training data

See [instructions](https://github.com/cp2k/dbcsr-data/README.md#contributing.md) in our [dedicated repository](https://github.com/cp2k/dbcsr-data)

##### Contribute predicted parameters

Submit a pull request updating the `parameters_GPU.json` file in question.



---

### Contributing to the training procedure



#### Adding a new predictive feature

- Choose the new feature's name
- Add the feature as a method of `class PredictiveParameters`, named `get_NAME`
- Add the derived feature to the data structure `derived_parameters` in [`kernels/cusmm_predict.py`](kernels/cusmm_predict.py)
