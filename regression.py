import torch
from sklearn import linear_model
from sklearn.metrics import r2_score

torch.set_printoptions(threshold=10_000)

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations={}
    for key, value in {**other, **monomial, **composite}.items():
        representations[key]=torch.load(f'weights/{key}/final.pt')

    lm = linear_model.Ridge(1e-8)
    lm.fit(representations['x'], representations['x+y'])
