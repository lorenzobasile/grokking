import torch
from models import OneLayer
from operations import monomial, other, composite
from train import generate_data
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=10_000)

loss=torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
representations={}
for key, value in {**other, **monomial, **composite}.items():
    representations[key]=torch.load(f'representations/{key}/final.pt')
head=OneLayer(128).to(device)
optimizer=torch.optim.Adam(head.parameters(), 0.1)
eq_token = 97
op_token = 97 + 1
data = generate_data(97, eq_token, op_token, monomial['x^2']).to(device)
accuracies=[]
for e in range(500):
    out=head(representations['x'])
    l=loss(out, data[-1])
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    acc = (out.argmax(-1) == data[-1]).float().mean()
    accuracies.append(acc.item())
    print(acc)
plt.plot(range(500), accuracies, label="train")
plt.title(f'Linear regression from x representation to x^2')
plt.xlabel("Optimization Steps")
plt.ylabel("Accuracy")
#plt.xscale("log", base=10)
plt.savefig(f'figures/x2fromx.png', dpi=150)
plt.close()
'''
out=head(representations['x'])
l=loss(out, data[-1])
optimizer.zero_grad()
l.backward()
optimizer.step()
acc = (out.argmax(-1) == data[-1]).float().mean()
print(acc)
'''
