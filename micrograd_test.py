import micrograd as mg
import torch


#checking for basic value functions and forward and backward passes

a = mg.Value(-4.0)
b = mg.Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).tanh()
d += 3 * d + (b - a).exp()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
g.backward()
a_mg, b_mg, g_mg = a, b, g

a = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
c = a + b
d = a * b + b**3
c = c + c + 1
c = c + 1 + c + (-a)
d = d + d * 2 + (b + a).tanh()
d = d + 3 * d + (b - a).exp()
e = c - d
f = e**2
g = f / 2.0
g = g + 10.0 / f
g.backward()
a_pt, b_pt, g_pt = a, b, g

print("diff of (g_mg.data - g_pt.item()) =", abs(g_mg.data - g_pt.item()))
print("diff of (a_mg.grad - a_pt.grad.item()) =", abs(a_mg.grad - a_pt.grad.item()))
print("diff of (b_mg.grad - b_pt.grad.item()) =", abs(b_mg.grad - b_pt.grad.item()))
