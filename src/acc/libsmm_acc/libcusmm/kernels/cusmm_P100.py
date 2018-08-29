# hardware-dependent constraints (GPU P100) 
# source: CUDA occupancy calculator: 
# http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls

maxTHREADSperBLOCK = 1024
maxBLOCKSperSM = 32
maxTHREADSperSM = 2048
SMEMperBLOCK = 49152 # bytes
SMEMperSM = 65536 # bytes
warp_size = 32 # threads
nSM = 56

