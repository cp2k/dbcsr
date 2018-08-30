# Hardware-dependent constraints (GPU P100) 
# source: CUDA occupancy calculator: 
# http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls

warp_size = 32 # threads
maxTHREADSperBLOCK = 1024
maxTHREADSperSM = 2048
maxBLOCKSperSM = 32
SMEMperBLOCK = 49152 # bytes
SMEMperSM = 65536 # bytes
nSM = 56

