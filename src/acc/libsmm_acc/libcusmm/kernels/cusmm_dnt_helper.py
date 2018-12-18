

#===============================================================================
from kernels.cusmm_dnt_largeDB1 import Kernel_dnt_largeDB1
from kernels.cusmm_dnt_largeDB2 import Kernel_dnt_largeDB2
from kernels.cusmm_dnt_medium   import Kernel_dnt_medium
from kernels.cusmm_dnt_small    import Kernel_dnt_small
from kernels.cusmm_dnt_tiny     import Kernel_dnt_tiny

kernel_algorithm = {
    'tiny': Kernel_dnt_tiny,
    'small': Kernel_dnt_small,
    'medium': Kernel_dnt_medium,
    'largeDB1': Kernel_dnt_largeDB1,
    'largeDB2': Kernel_dnt_largeDB2
}


#===============================================================================
def params_dict_to_kernel(**params):
    return kernel_algorithm[params.pop('algorithm')](**params)


def descr_to_kernel(kernel_descr):
    import re
    from ast import literal_eval
    re_kernel_descr = re.compile(r"Kernel_dnt_(\w+)(\(.*\)) , # (\d+\.\d+) GFlop/s")
    match = re_kernel_descr.search(kernel_descr).groups()
    algo = match[0]
    m = match[1].replace('=', '\':')
    m = m.replace(', ', ', \'')
    m = m.replace('(', '{\'')
    m = m.replace(')', '}')
    params = dict(literal_eval(m))
    params['perf'] = float(match[2])
    return kernel_algorithm[algo](**params)


