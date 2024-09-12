import numpy as np

# 1

x = np.arange(2,13)
prob = np.array([1,2,3,4,5,6,5,4,3,2,1]) / 36
Ex = sum(x*prob)

mean=sum(x*prob)

var= sum((x-Ex)**2*prob)

# 2

my_Ex = Ex * 2 + 3

my_std =  np.sqrt(var*4)