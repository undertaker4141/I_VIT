import numpy as np
import sys
sys.path.append("/home/undertaker4141/I_VIT")
from cmodel_rtl_reference.nonlinear_cmodel_reference import int_exp_shift_kernel_standard

x_algo = np.zeros((1, 3, 197, 197), dtype=np.int64)
x0_int = np.array([221], dtype=np.int32)
# Or maybe x0_int is float? Let's trace
scale_val = np.array([0.0031])
x0_int = np.floor(-np.log(2) / scale_val).astype(np.int32)
print("x0_int type:", type(x0_int), type(x0_int[0]))
# int_exp_shift_kernel_standard(x_algo, x0_int, 15)
q = x_algo // x0_int
r = x_algo - x0_int * q
print("q type:", q.dtype, "r type:", r.dtype)
exp_int = (r >> 1) - x0_int
print("success")
