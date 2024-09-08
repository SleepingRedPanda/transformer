import torch
from torch import nn


class InputEmbeddin(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)
        
        # create a matri of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a position tensor of shape (seq_len, 1)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(10000.0) / d_model))
        # fill the even indices with sin and odd indices with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1)]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(1))
        self.b_2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.linea_2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, h, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model must be divisible by h'
        
        self.d_k = d_model // h
        self.linear_q = nn.Linear(d_model, d_model) # W_q
        self.linear_k = nn.Linear(d_model, d_model) # W_k
        self.linear_v = nn.Linear(d_model, d_model) # W_v
        
        self.linear_out = nn.Linear(d_model, d_model) # W_o
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=nn.Dropout):
        d_k = query.size(-1)
        
        # )batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # batch_size, h, seq_len, seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
            
    def forward(self, q, k, v, mask=None):
        query, key, value = self.linear_q(q), self.linear_k(k), self.linear_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = self.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = self.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = self.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.h * self.d_k)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.linear_out(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention:MultiHeadAttention, feed_forward:FeedForward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __initt__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)