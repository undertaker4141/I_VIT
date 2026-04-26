import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cmodel_rtl_reference.linear_cmodel_reference import (
    int_dense_kernel,
    int_matmul_kernel
)
from cmodel_rtl_reference.nonlinear_cmodel_reference import (
    int_layer_norm_fixed,
    int_gelu_kernel_fixed,
    int_softmax_kernel_fixed
)
print("Modules imported successfully!")
