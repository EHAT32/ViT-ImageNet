import pytorch_lightning as pl
import torch.nn as nn
import torch

class ViT(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_proj = Embedding(cfg)
        self.encoder = Encoder(cfg)
        self.classifier = ClassifierHead(cfg)
        
    def forward(self, x):
        emb = self.embedding_proj(x)
        emb = self.encoder(emb)
        logits, probs = self.classifier(emb)
        return logits, probs


class ClassifierHead(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hid_dim = cfg['hidden_dim']
        self.class_num = cfg['class_number']
        self.linear = nn.Linear(self.hid_dim, self.class_num)
    
    def forward(self, x):
        logits = self.linear(x[:, 0]) #cls token
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class Encoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_num = cfg['block_number']
        self.blocks = nn.Sequential(*[BaseBlock(cfg) for _ in range(self.block_num)])
        
    def forward(self, x):
        out = self.blocks(x)
        return out

class BaseBlock(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hid_dim = cfg['hidden_dim']
        self.norm1 = nn.LayerNorm(self.hid_dim)
        self.mha = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(self.hid_dim)
        self.ffn = FFN(cfg)
        
    def forward(self, x):
        out = self.norm1(x)
        out = out + self.mha(out)
        out = self.norm2(out)
        out = out + self.ffn(out)
        return out

class FFN(nn.Module):
    #do not search "mlp in transformer" in google
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hid_dim = cfg['hidden_dim']
        self.ffn_dim = cfg['ffn_dim']
        self.linear1 = nn.Linear(self.hid_dim, self.ffn_dim)
        self.linear2 = nn.Linear(self.ffn_dim, self.hid_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(cfg['ffn_dropout'])
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(x)
        out = self.linear2()
        out = self.dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_num = cfg['attention_head_number']
        self.hid_dim = cfg['hidden_dim']
        assert self.hid_dim % self.head_num == 0
        head_dim = self.hid_dim // self.head_num
        self.heads = nn.ModuleList([AttentionHead(self.hid_dim, head_dim, cfg['attention_dropout']) for _ in range(self.head_num)])
        self.linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.dropout = nn.Dropout(cfg['multihead_attention_dropout'])
        
    def forward(self, x):
        outputs = []
        for head in self.heads:
            output = head(x)
            outputs.append(output)
        output = torch.cat(outputs, dim=-1)
        output = self.linear(output)
        output = self.dropout(output)
        return output
        

class AttentionHead(nn.Module):
    def __init__(self, hid_dim, head_dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hid_dim = hid_dim
        self.head_dim = head_dim
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(hid_dim, head_dim)
        self.key = nn.Linear(hid_dim, head_dim)
        self.value = nn.Linear(hid_dim, head_dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(k.size(-1))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.bmm(attn_probs, v)
        attn_out = self.dropout(attn_out)
        return attn_out
        
        
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