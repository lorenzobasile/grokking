import math
from argparse import ArgumentParser
from itertools import permutations

import operations

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import os

from models import Decoder

def generate_data(p, eq_token, op_token, operation):
    """
    x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = operation(x,y) % p

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    return torch.stack([x, op, y, eq, result])


def main(args):
    torch.manual_seed(0)
    ops={**operations.monomial, **operations.composite, **operations.other}

    if not os.path.exists(f'weights/{args.operation}'):
        os.makedirs(f'weights/{args.operation}')
    if not os.path.exists(f'figures/{args.operation}'):
        os.makedirs(f'figures/{args.operation}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token = args.p + 1

    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
    ).to(device)

    # "We train on the binary operation of division mod 97 with 50% of the data
    # in the training set."
    data = generate_data(args.p, eq_token, op_token, ops[args.operation])
    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    '''
    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    '''
    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    exp=0
    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(valid_data, False), (train_data, True)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()

                total_acc += acc.item() * input.shape[-1]

            if is_train:
                #print("Train ", total_acc)
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
            else:                
                #print("Test ", total_acc)
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])

        if (e + 1) % 100 == 0:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, val_acc, label="val")
            plt.legend()
            plt.title(f'{args.operation}(training on 50% of data)')
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.savefig(f'figures/{args.operation}/acc.png', dpi=150)
            plt.close()

            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.legend()
            plt.title(f'{args.operation}(training on 50% of data)')
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.savefig(f'figures/{args.operation}/loss.png', dpi=150)
            plt.close()

        if (e+1)*steps_per_epoch > 10**exp:
            torch.save(model.state_dict(), f'weights/{args.operation}/weights{10**exp}.pt')
            exp+=1
    torch.save(model.state_dict(), f'weights/{args.operation}/final.pt')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=3e6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--operation", default="mul")
    args = parser.parse_args()
    main(args)
