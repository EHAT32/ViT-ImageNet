import pytorch_lightning as pl
import torch.nn as nn
import torch

class ViT(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
class Embedding(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_emb = Patch2Embedding(cfg)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg['embedding_dim']))
        self.pos_emb = nn.Parameter(torch.randn(1, self.patch_emb.patch_num+1, cfg['embedding_dim']))
        self.dropout = nn.Dropout(cfg['embedding_dropout'])
        
    def forward(self, x):
        #x : N, patch_num, embedding_dim
        out = self.patch_emb(x)
        batch_size = out.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.pos_emb
        out = self.dropout(out)
        return out
    
        
class Patch2Embedding(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = cfg['img_size']
        self.patch_size = cfg['patch_size']
        self.channel_num = cfg['channel_num']
        self.embed_dim = cfg['embedding_dim']
        self.patch_num = (self.img_size // self.patch_size) ** 2
        self.patch_conv = nn.Conv2d(self.channel_num, self.embed_dim, self.patch_size, self.patch_size)
        
    def forward(self, x):
        #x : (N, C, H, W)
        #out : (N, patch_num, embed_dim)
        out = self.patch_conv(x)
        out = out.flatten(2).transpose(1,2)
        return out
    
