from models import Decoder
from argparse import ArgumentParser

from operations import monomial, composite
from train import generate_data

def main(args):
    for key, value in monomial.items():
        eq_token = args.p
        op_token = args.p + 1
        model = Decoder(
            dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
        ).to(device)
        model.load_state_dict(torch.load(f'weights/{key}/final.pt'))
        data = generate_data(args.p, eq_token, op_token, value)
        print(data.shape)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e5)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    main(args)
