import torch

x = torch.randn(1000, 1000).cuda()
y = torch.matmul(x, x)
print("Success on GPU")