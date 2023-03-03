import torch

# create a tensor of size (3, 4) with random values
x = torch.randn(3, 4)
print("x before:", x)

# create a tensor of size (2, 4) with random values
y = torch.randn(2, 4)
print("y:", y)

# scatter the values of y into x at the specified indices
indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
x.scatter_(0, indices, y)

print("x after:", x)

