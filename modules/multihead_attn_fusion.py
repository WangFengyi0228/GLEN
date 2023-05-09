import torch
import torch.nn as nn

      
    
class MultiheadAttentionFusion(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super(MultiheadAttentionFusion, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim, bias=False)
        self.w_k = nn.Linear(hid_dim, hid_dim, bias=False)
        self.w_v = nn.Linear(hid_dim, hid_dim, bias=False)
        
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim/n_heads])).to(device)
        
        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, query, key, value):
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        residual = V

        Q = Q.view(-1, self.n_heads, self.hid_dim//self.n_heads).permute(1, 0, 2)
        K = K.view(-1, self.n_heads, self.hid_dim//self.n_heads).permute(1, 0, 2)
        V = V.view(-1, self.n_heads, self.hid_dim//self.n_heads).permute(1, 0, 2)
                   
        attention = torch.matmul(Q, K.permute(0, 2, 1))/self.scale
        attention = self.dropout(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(-1, self.n_heads * (self.hid_dim//self.n_heads))
        x = self.fc(x)
        x = self.layer_norm(self.dropout(x+residual))
        return x

