import torch
from sklearn import linear_model
from sklearn.metrics import r2_score
from operations import monomial, other, composite
import numpy as np

torch.set_printoptions(threshold=10_000)

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations={}
    representationsNew={}
    for key, value in {**other, **monomial, **composite}.items():
        representations[key]=torch.load(f'representations/{key}/final.pt')
        representationsNew[key]=torch.load(f'representationsNew/{key}/final.pt')
    lm = linear_model.LinearRegression()
    x=representations['x^2'].cpu().numpy()
    x_n=representationsNew['x'].cpu().numpy()
    xy=representations['xy'].cpu().numpy()
    x2=representations['x^2'].cpu().numpy()
    print(x.shape)
    lm.fit(x_n, x)
    print(r2_score(lm.predict(x_n), x, multioutput='variance_weighted'))
