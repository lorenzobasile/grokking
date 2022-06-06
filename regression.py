import torch
from sklearn import linear_model
from sklearn.metrics import r2_score
from operations import monomial, other, composite
from svcca.cca_core import get_cca_similarity
import numpy as np

torch.set_printoptions(threshold=10_000)

def svd_reduction(M):
    M-=np.mean(M, axis=1, keepdims=True)
    U,s,V=np.linalg.svd(M, full_matrices=False)
    sv_sum = np.sum(s)
    i = 0
    quality_ratio = 0
    while i < len(s) and quality_ratio < 0.99:
        i += 1
        quality_ratio = np.sum(s[:i])/sv_sum

    M_reduced = np.dot(U[:, :i], s[:i]*np.eye(i))
    return M_reduced

with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations={}
    representationsNew={}
    for key, value in {**other, **monomial, **composite}.items():
        representations[key]=torch.load(f'representations/{key}/final.pt')
        if key=='x^2':
            representationsNew[key]=torch.load(f'representations/{key}/random.pt')

    x=representations['y^2'].cpu().numpy()
    y=representationsNew['x^2'].cpu().numpy()

    x=svd_reduction(x).T
    y=svd_reduction(y).T

    '''
    x = x - np.mean(x, axis=1, keepdims=True)
    y = y - np.mean(y, axis=1, keepdims=True)
    
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(x, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(y, full_matrices=False)

    svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)
    '''
    svcca_results = get_cca_similarity(x, y, epsilon=1e-10, verbose=False)


    #print(x.shape, y.shape)
    #result=get_cca_similarity(x.T, y.T)
    print(svcca_results['cca_coef1'])
    #lm = linear_model.LinearRegression()
    #x=representations['x+y'].cpu().numpy()
    #lm.fit(x_n, x)
    #print(r2_score(lm.predict(x_n), x, multioutput='variance_weighted'))
