import torch
from models import OneLayer
from sklearn import linear_model
from sklearn.metrics import r2_score
from operations import monomial, other, composite
from train import generate_data
import numpy as np

torch.set_printoptions(threshold=10_000)

loss=torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
representations={}
for key, value in {**other, **monomial, **composite}.items():
    representations[key]=torch.load(f'representations/{key}/final.pt')
head=OneLayer(128)
optimizer=torch.nn.optim.Adam(head.parameters(), 0.1)
eq_token = args.p
op_token = args.p + 1
data = generate_data(args.p, eq_token, op_token, monomial['x^2']).to(device)
for e in range(10):
    dl = torch.split(data, 16, dim=1)
    for input in dl:
        out=head(representations['x'])
        l=loss(out, data[-1])
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        acc = (out.argmax(-1) == data[-1]).float().mean()
        print(acc)
