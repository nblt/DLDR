import torch 


x = torch.tensor([[1.],[2.],[3.],[4.],[5.],[6.]], requires_grad=True)
P = torch.rand((6, 6))
z = torch.mm(P, x)
a = z[0:2].reshape(-1, 1)
b = z[2:4].reshape(-1, 1)
c = z[4:6].reshape(-1, 1)

y = a+b+c
y = y.sum()
y.backward()

print(x.grad)
