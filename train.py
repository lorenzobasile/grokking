import math
from argparse import ArgumentParser

import operations

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np

from models import Decoder

def running_mean(array, window):
    if window==1:
        return array
    N=window
    #if len(array)<=window:
    #return array
    y_padded = np.pad(array, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return y_smooth
def sparsity(n_data, n_heads, l0_norm, head_dim):
    low=n_data*n_heads*head_dim
    up=n_data*n_heads*np.cumsum(np.arange(1,head_dim+1))[-1]
    sparsity=1-(l0_norm-low)/(up-low)
    return sparsity


def main(args):
    torch.manual_seed(0)
    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.

    run_name=args.run_name

    alldata=operations.generate_data(args.p, operations.composite[args.operation]).T

    if not os.path.exists(f'weights/{args.operation}/{run_name}'):
        os.makedirs(f'weights/{args.operation}/{run_name}')
    if not os.path.exists(f'figures/{args.operation}/{run_name}'):
        os.makedirs(f'figures/{args.operation}/{run_name}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    factor=('factor' in run_name)


    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=128, num_layers=args.layers, num_heads=args.heads, num_tokens=args.p + 2, drop_p=args.drop_p, factor=factor
    ).to(device)

    # "We train on the binary operation of division mod 97 with 50% of the data
    # in the training set."
    
    train_idx, valid_idx = torch.randperm(len(alldata)).split((len(alldata) // 2)+1)
    train_data_unshuffled, valid_data = alldata[train_idx], alldata[valid_idx]

    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98

    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    
    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data_unshuffled.shape[0] / args.batch_size)

    l0_norm=[[] for i in range(args.layers)]

    train_acc, val_acc, train_loss, val_loss= [], [], [], []
    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        # randomly shuffle train data
        train_data = train_data_unshuffled[torch.randperm(train_data_unshuffled.shape[0])]

        for data, is_train in [(valid_data, False), (train_data, True)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=0)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:,:-1])[:,-1]
                    # calculate loss only on the answer part of the equation (last element)
                    loss = F.cross_entropy(logits, input[:,-1])
                    total_loss += loss.item() * len(input)

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                acc = (logits.argmax(-1) == input[:,-1]).float().mean()

                total_acc += acc.item() * len(input)

            if is_train:
                train_acc.append(total_acc / len(train_data))
                train_loss.append(total_loss / len(train_data))
            else:
                val_acc.append(total_acc / len(valid_data))
                val_loss.append(total_loss / len(valid_data))
            
                _, all_repr, all_att=model.extract_representation(alldata.to(device)[:,:-1])
    
                for i, layer_att in enumerate(all_att):
                    
                    #if i==len(all_att)-1:
                    '''   
                    rep=all_repr[i].detach().cpu().numpy()[:,-1]
                    data = Data(rep, maxk=3)
                    d=data.compute_id_2NN()[0]
                    id[i].append(d)
                    
                    att_tensor=torch.stack(all_att)
                    rep=att_tensor.permute(1,0,2,3,4).detach().cpu().numpy().reshape(9409,-1)
                    data = Data(rep, maxk=3)
                    d=data.compute_id_2NN()[0]
                    id_att.append(d)
                    '''
                        
                        

                    #else:
                    #    pass
                    l0_norm[i].append(torch.sum(layer_att>1e-3).item())



        if (len(train_acc)*steps_per_epoch) % 1000 == 0:
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            torch.save(model.state_dict(), f'weights/{args.operation}/{run_name}/{steps[-1].item()+steps_per_epoch}.pt')    
            if (len(train_acc)*steps_per_epoch) % 1000 == 0:  
                plt.plot(steps, train_acc, label="train")
                plt.plot(steps, val_acc, label="val")
                plt.legend()
                plt.title(f'{args.operation}(training on 50% of data)')
                plt.xlabel("Optimization Steps")
                plt.ylabel("Accuracy")
                plt.xscale("log", base=10)
                plt.savefig(f'figures/{args.operation}/{run_name}/acc.png', dpi=150)
                plt.close()

                plt.plot(steps, train_loss, label="train")
                plt.plot(steps, val_loss, label="val")
                plt.legend()
                plt.title(f'{args.operation}(training on 50% of data)')
                plt.xlabel("Optimization Steps")
                plt.ylabel("Loss")
                plt.xscale("log", base=10)
                plt.yscale("log", base=10)
                plt.savefig(f'figures/{args.operation}/{run_name}/loss.png', dpi=150)
                plt.close()

                for l in range(args.layers):
                    plt.plot(steps, running_mean(sparsity(len(alldata), args.heads, np.array(l0_norm[l]), 4), 1), label=f'layer {l}')
                plt.plot(steps, train_acc, label="train acc")
                plt.plot(steps, val_acc, label="val acc")
                plt.legend()
                plt.title(f'{args.operation}(training on 50% of data)')
                plt.xlabel("Optimization Steps")
                plt.xscale("log", base=10)
                plt.savefig(f'figures/{args.operation}/{run_name}/joint.png', dpi=150)
                plt.close()

            


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--budget", type=int, default=1e6)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--operation", default="xy")
    parser.add_argument("--run_name", default="default")
    parser.add_argument("--drop_p", type=float, default=0.0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)

    
    args = parser.parse_args()
    main(args)
