from models import Decoder, Block
from argparse import ArgumentParser
import torch
from operations import monomial, composite, other
from train import generate_data
import os
#from svcca.cca_core import get_cca_similarity
#from similarity import svd_reduction

def main(args):
    torch.set_printoptions(threshold=10_000)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for key, value in {**other, **monomial, **composite}.items():
            if not os.path.exists(f'representations/{key}'):
                os.makedirs(f'representations/{key}')
            print("operation: ", key)
            eq_token = args.p
            op_token = args.p + 1
            data = generate_data(args.p, eq_token, op_token, value).to(device)
            model = Decoder(
                        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
                    ).to(device)
            model.load_state_dict(torch.load(f'weights/{key}/final.pt'))
            repr=model.extract_representation(data[:-1])[-1]
            print(repr.shape)
            torch.save(repr, f'representations/{key}/final.pt')
            print(torch.argmax(model(data[:-1])[-1], 1).reshape(97,96))

'''
        eq_token = args.p
        op_token = args.p + 1
        data = generate_data(args.p, eq_token, op_token, composite['xy']).to(device)
        model = Decoder(
                    dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
                ).to(device)
        model.load_state_dict(torch.load(f'weights/xy/final.pt'))
        #print(mo)
        #x=model.extract_representation(data[:-1])[-1].cpu().numpy()
        #torch.save(repr, f'representations/xy/new.pt')
        y=torch.load('representations/xy/final.pt')      
        #x=y
        #x=svd_reduction(x).T
        #y=svd_reduction(y).T
        #svcca_results = get_cca_similarity(x, y, epsilon=1e-10, verbose=False)
        #print(svcca_results['cca_coef1'].mean())
        #print(torch.argmax(model(data[:-1])[-1], 1).reshape(97,96))
        print(torch.argmax(model.head(y), 1).reshape(97,96))
        

'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    args = parser.parse_args()
    main(args)
