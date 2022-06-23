import torch
from sklearn import linear_model
from sklearn.metrics import r2_score
from operations import monomial, other, composite, generate_data
from svcca.cca_core import get_cca_similarity
import seaborn as sns
import numpy as np

torch.set_printoptions(threshold=10_000)
np.set_printoptions(precision=3)

def svd_reduction(M):
    M-=np.mean(M, axis=1, keepdims=True)
    U,s,V=np.linalg.svd(M, full_matrices=False)
    sv_sum = np.sum(s)
    i = 0
    quality_ratio = 0
    while i < len(s) and quality_ratio < 0.999:
        i += 1
        quality_ratio = np.sum(s[:i])/sv_sum
    M_reduced = np.dot(U[:, :i], s[:i]*np.eye(i))
    return M_reduced
def main():
    with torch.no_grad():
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ops={**other, **monomial, **composite}
        for filename in ["10", "100", "1000", "10000", "100000", "final"]:
            operation_names=[]
            representations={}
            for key, value in ops.items():
                operation_names.append(key)
                representations[key]=torch.load(f'representations/{key}/{filename}.pt').cpu().numpy()
            representations['init']=torch.load('representations/init.pt').cpu().numpy()
            operation_names.append('init')
            results=np.zeros((len(operation_names), len(operation_names)))
            for i in range(len(operation_names)):
                for j in range(len(operation_names)):
                    x=representations[operation_names[i]]
                    y=representations[operation_names[j]]
                    x=svd_reduction(x).T
                    y=svd_reduction(y).T
                    #print(x.shape, y.shape)
                    svcca_results = get_cca_similarity(x, y, epsilon=1e-10, verbose=False)
                    results[i,j]=svcca_results['cca_coef1'].mean()
            print(results)
            hm=sns.heatmap(results, annot=True, cmap='Blues', fmt='.2f', xticklabels=operation_names, yticklabels=operation_names).set_title("SVCCA similarity score")
            figure = hm.get_figure()
            figure.savefig('figures/heatmap_'+filename+'.png', dpi=400)
            figure.clf()
        representations={}
        results=np.zeros((len(operation_names)-1, len(operation_names)-1))
        for operation in operation_names[:-1]:
            eq_token = 97
            op_token = 97 + 1
            representations[operation] = generate_data(97, eq_token, op_token, ops[operation])[-1]
            print(operation)
        for i in range(len(operation_names)-1):
            for j in range(len(operation_names)-1):
                xi=representations[operation_names[i]].float()
                xj=representations[operation_names[j]].float()
                results[i,j]=torch.sum(torch.eq(xi, xj))/(len(xi))
                #results[i,j]=torch.dot(xi, xj)/torch.norm(xi, 2)/torch.norm(xj, 2)
        hm=sns.heatmap(results, annot=True, cmap='Blues', fmt='.0f', xticklabels=operation_names[:-1], yticklabels=operation_names[:-1]).set_title("Overlap score")
        figure = hm.get_figure()
        figure.savefig('figures/heatmap_overlap.png', dpi=400)
        figure.clf()           
if __name__ == "__main__":
    main()
