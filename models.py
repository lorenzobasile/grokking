import torch
from torch import nn

class OneLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Linear(dim, 97)

    def forward(self, x):
        return self.layer(x)

class Block(nn.Module):
    """
    Causal transformer block
    """

    def __init__(self, dim, num_heads, drop_p, factor):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.ln_emb = nn.LayerNorm(dim) if factor else None
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, emb):
        attn_mask = torch.full(
            (x.shape[1], x.shape[1]), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)
        emb = self.ln_emb(emb) if self.ln_emb is not None else x
        x = self.ln_1(x)
        a, att = self.attn(emb, emb, x, attn_mask=attn_mask, need_weights=True)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, att


class Decoder(nn.Module):
    """
    Causal Transformer decoder
    """

    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, drop_p=0, factor=False):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(4, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads, drop_p, factor))

        self.factor=factor

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def extract_representation(self, x):
        att=[]
        rep=[]
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[1], device=x.device)
        #h = h + 
        emb=self.position_embeddings(positions).expand_as(h)
        h = h if self.factor else h+emb
        for layer in self.layers:
            h, att_l = layer(h, emb)
            rep.append(h)
            att.append(att_l)
        return self.ln_f(h), rep, att

    def forward(self, x):
        out, rep, att=self.extract_representation(x)
        return self.head(out)
        '''
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits
        '''
