# Training Procedure for predictive modelling of optimal parameters in `libcusmm` 

Introduction for what this is … blahblah 
Use notebook/autotuningbalhblah to explore…

### Requirements

Install all python packages required (if you do not want this project's requirements to interfere with your other Python projects, consider doing so in a (virtual environment)[https://docs.python.org/3/tutorial/venv.html]), using

```%bash
pip install -r requirements.txt
```

### 1. Collect data for ML procedure

#### Autotuning
- Follow CP2K wiki instructions 
- Before doing this, we will ask you to kindly upload your data, please take note of … (git commit template) 
- …
- Result : multiple folders « tuneMxNxK », which contain : 
- Makefile
- Libcusmm_benchmark.o 
- Slurm-xxx.out
- Information contained: 
- Ptxas compiltion information 
- (Overall winner)
- Tune_MxNxK.job
- N times: 
- Tune_MxNxK_exeN (executable) 
- Tune_MxNxK_exeN.log
- Information contained 
- Parameter sets and corresponding performance measured
- Tune_MxNxK_exeN_main.cu, .o
- Tune_MxNxK_exeN_partJ.cu, .o 

#### AND/OR: Download pre-collected data from dedicated repo

```%bash
wget blahblah 
```

### 2.	Predict_collect

e.g. 
predict_collect.py –arch 60 --folder /scratch/snx3000/alicej/tune_dataset
The arch number is needed because this script uses the GPU cards’ properties to compute derived features from raw features. Accordingly, before running the script, make sure that the GPU properties of the GPU you’re using are given in the file ` 'kernels/gpu_properties.json'`. If not, please add them. 

Show: ![parameters dependency graph](../../../../../docs/images/libcusmm_predictive_modelling_features.png)

### 3.	explore this data

EDA with notebooks: notebooks/inspect_training_data

### 4.	merge with already eisting data if applicable 

### 5.	(if you’ve collected new data) Please upload to our repo 

… instructions … 

### 6.	train

### 7.	genpars

Predict_genpars.py
•	Cf job_genpars.sh
•	Time to solution: 08:17:52
•	

### 8. Evaluate

### 9. Contribute your new parameters

