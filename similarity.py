import torch
from sklearn import linear_model
from sklearn.metrics import r2_score
from operations import monomial, other, composite
from svcca.cca_core import get_cca_similarity
import seaborn as sns
import numpy as np
from train import generate_data

torch.set_printoptions(threshold=10_000)
np.set_printoptions(precision=2)

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
def main():
    with torch.no_grad():
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for filename in ["10"]:#, "100", "1000", "10000", "100000", "final"]:
            operation_names=[]
            representations={}
            for key, value in {**other, **monomial, **composite}.items():
                operation_names.append(key)
                representations[key]=torch.load(f'representations/{key}/{filename}.pt').cpu().numpy()
            representations['random']=torch.load('old/representations/x^2/random.pt').cpu().numpy()
            operation_names.append('random')
            results=np.zeros((len(operation_names), len(operation_names)))
            for i in range(len(operation_names)):
                for j in range(len(operation_names)):
                    x=representations[operation_names[i]]
                    y=representations[operation_names[j]]
                    x=svd_reduction(x).T
                    y=svd_reduction(y).T
                    svcca_results = get_cca_similarity(x, y, epsilon=1e-10, verbose=False)
                    results[i,j]=svcca_results['cca_coef1'].mean()
            print(results)
            hm=sns.heatmap(results, annot=True, cmap='Blues', fmt='.2f', xticklabels=operation_names, yticklabels=operation_names).set_title("SVCCA similarity score")
            figure = hm.get_figure()
            figure.savefig('figures/heatmap_'+filename+'.png', dpi=400)
            figure.clf()
        for operation in operation_names:
            eq_token = 97
            op_token = 97 + 1
            data = generate_data(97, eq_token, op_token, operation)
            print(data, data.shape)
        #lm = linear_model.LinearRegression()
        #x=representations['x+y'].cpu().numpy()
        #lm.fit(x_n, x)
        #print(r2_score(lm.predict(x_n), x, multioutput='variance_weighted'))
if __name__ == "__main__":
    main()
