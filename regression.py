from models import Decoder, Block
from argparse import ArgumentParser
import torch
from operations import monomial, composite
from train import generate_data

def main(args):
    torch.set_printoptions(threshold=10_000)
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''
        for key, value in composite.items():
            print(key)
            eq_token = args.p
            op_token = args.p + 1
            model = Decoder(
                dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
            ).to(device)
            model.load_state_dict(torch.load(f'weights/{key}/final.pt'))
            data = generate_data(args.p, eq_token, op_token, value).to(device)
            print(model.extract_representation(data).shape)
        '''
        eq_token = args.p
        op_token = args.p + 1
        data = generate_data(args.p, eq_token, op_token, monomial['x']).to(device)
        model = Decoder(
                    dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
                ).to(device)
        print(torch.argmax(model(data[:-1])[-1], 1).reshape(97,96))
        
        model.load_state_dict(torch.load(f'weights/x/weights1.pt'))
        
        print(torch.argmax(model(data[:-1])[-1], 1).reshape(97,96))




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    main(args)
