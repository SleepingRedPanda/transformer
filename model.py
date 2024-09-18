import torch
from torch import nn
import math


class InputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x.int()) * math.sqrt(self.d_model)


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
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        # fill the even indices with sin and odd indices with cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1)]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

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
        self.linear_q = nn.Linear(d_model, d_model)  # W_q
        self.linear_k = nn.Linear(d_model, d_model)  # W_k
        self.linear_v = nn.Linear(d_model, d_model)  # W_v

        self.linear_out = nn.Linear(d_model, d_model)  # W_o
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=nn.Dropout):
        d_k = query.size(-1)

        # )batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        # batch_size, h, seq_len, seq_len
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        query, key, value = self.linear_q(q), self.linear_k(k), self.linear_v(
            v)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                        self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], x.shape[2], self.h * self.d_k)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.linear_out(x)


class ResidualConnection(nn.Module):

    def __init__(self, features, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, features, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(
            x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, trg_embed: InputEmbedding, src_pos: PositionalEncoding, trg_pos: PositionalEncoding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.proj = proj
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, trg, encoder_output, src_mask, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)
    
    def project(self, x):
        return self.proj(x)
    

def build_transformer(src_vocab_size, trg_vocab_size, src_sq_len, trg_sq_len, d_model = 512, n = 6, h = 8, dropout = 0.1, d_ff = 2048):
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    trg_embed = InputEmbedding(d_model, trg_vocab_size)
    
    #Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_sq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_sq_len, dropout)
    
    # Create encoder blocks
    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
        
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)
    
    # Create transformer
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer