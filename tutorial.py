import torch
import numpy as np

# initializing a tensor

## directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

## from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


## from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


