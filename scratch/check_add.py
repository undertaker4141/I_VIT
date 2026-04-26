import numpy as np

t = np.load('/home/undertaker4141/I_VIT/patterns_tvm/golden/embed_add_pos_int.npy')
c = np.load('/home/undertaker4141/I_VIT/patterns_cmodel_golden/golden/embed_add_pos_int.npy')
print('TVM embed_add_pos:', t.min(), t.max())
print('CMOD embed_add_pos:', c.min(), c.max())
print('TVM sample:', t.flatten()[:10])
print('CMOD sample:', c.flatten()[:10])
