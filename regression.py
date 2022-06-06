import torch
from sklearn import linear_model
from sklearn.metrics import r2_score
from operations import monomial, other, composite
from svcca.cca_core import get_cca_similarity
import numpy as np

torch.set_printoptions(threshold=10_000)

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations={}
    representationsNew={}
    for key, value in {**other, **monomial, **composite}.items():
        representations[key]=torch.load(f'representations/{key}/final.pt')
        if key=='x^2':
            representationsNew[key]=torch.load(f'representations/{key}/random.pt')

    x=representations['x'].cpu().numpy()
    y=representations['x'].cpu().numpy()
    result=get_cca_similarity(x.T, y.T)
    print(result['cca_coef1'])
    #lm = linear_model.LinearRegression()
    #x=representations['x+y'].cpu().numpy()
    #lm.fit(x_n, x)
    #print(r2_score(lm.predict(x_n), x, multioutput='variance_weighted'))
