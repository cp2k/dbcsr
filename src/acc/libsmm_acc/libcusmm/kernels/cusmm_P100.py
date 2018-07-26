# hardware-dependent limits and sizes
# for P100 
# source: 
maxTHREADSperBLOCK = 1024
maxBLOCKSperSM = 32
maxTHREADSperSM = 2048
SMEMperBLOCK = 49152 # bytes
SMEMperSM = 65536 # bytes
warp_size = 32 # threads

# sizeof
sizeof_int = 4
sizeof_double = 8

def round_up_to_multiple(x, step):
    if x % step == 0:
        return x
    else:
        return x + step - x % step

