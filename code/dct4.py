import numpy as np

# 빨강 2 / 파랑 3
p_r = 2/5
p_b = 3/5

h_zero=-p_r*np.log2(p_r) - p_b*np.log2(p_b)
h_zero

# 빨강 1 / 파랑 3
p_r = 1/4
p_b = 3/4

h_1_r=-p_r*np.log2(p_r) - p_b*np.log2(p_b)
h_1_r