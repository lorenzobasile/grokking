from models import Decoder, Block
from argparse import ArgumentParser
import torch
from operations import monomial, composite, other
from train import generate_data

def main(args):
    torch.set_printoptions(threshold=10_000)

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for key, value in {**other, **monomial, **composite}.items():
            print("operation: ", key)
            eq_token = args.p
            op_token = args.p + 1
            data = generate_data(args.p, eq_token, op_token, value).to(device)
            model = Decoder(
                        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
                    ).to(device)
            model.load_state_dict(torch.load(f'weights/{key}/final.pt'))
            repr=model.extract_representation(data[:-1])
            print(repr.shape)
            torch.save(repr, f'representations/{key}/final.pt')
            print(torch.argmax(model(data[:-1])[-1], 1).reshape(97,96))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    args = parser.parse_args()
    main(args)
