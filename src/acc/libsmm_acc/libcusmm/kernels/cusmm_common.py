# Number of parameters in one stack entry
stack_size = 16005
npar=3

# sizeof
sizeof_int = 4
sizeof_double = 8

# utility functions
def round_up_to_multiple(x, step):
    if x % step == 0:
        return x
    else:
        return x + step - x % step
