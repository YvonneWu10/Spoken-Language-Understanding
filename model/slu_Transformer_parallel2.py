#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.autograd import Variable
import math
from einops import rearrange
from functools import partial


def default(val, d):
    if (val is not None):
        return val
    else:
        return d

class TagTransformerParallel2(nn.Module):

    def __init__(self, config):
        super(TagTransformerParallel2, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, 768 // 2, num_layers=config.num_layer_rnn, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(config.embed_size, config.embed_size)
        self.transformer = Transformer(config.embed_size, config.num_layer_attn, 2, config.batch_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.norm = nn.LayerNorm(config.embed_size)
        self.output_layer = TaggingFNNDecoder(config.embed_size, config.num_tags, config.tag_pad_idx)
        

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        embed = self.fc(embed)
        # trans_input = self.dropout_layer(trans_input)

        if self.config.CNN:
            CNN = CNNEncoder(self.config.dropout).to(self.config.device)
            cnn_out = CNN(embed)
            cnn_out = self.dropout_layer(cnn_out)
            trans_out = self.transformer(cnn_out)
            trans_out = self.norm(trans_out)
            hiddens = (trans_out)
        else:
            packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
            packed_rnns_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
            rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnns_out, batch_first=True)
            rnn_out = self.dropout_layer(rnn_out)
            trans_out = self.transformer(embed)
            trans_out = self.norm(trans_out)
            hiddens = (trans_out + rnn_out)
        
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return prob

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class FullAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads = 8,
                 dim_head = 64,
                 dropout = 0.,
                 qkv_bias = False
                ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.dim_head = dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # [B, N, H*D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) # [B, H, N, D]
        
        scaled_attention_logits = torch.einsum("bhxd,bhyd -> bhxy", q, k) / torch.sqrt(torch.tensor(self.dim_head, dtype=torch.float32))
        attention_weights = F.softmax(scaled_attention_logits, dim=-1) # [B, H, N, N]
        
        attn_output = torch.einsum("bhnx,bhxd -> bhnd", attention_weights, v) # [B, H, N, D]
        out = rearrange(attn_output, 'b h n d -> b n (h d)', h = h) # [B, N, H*D]
        out = self.to_out(out)
        
        return self.dropout(out)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, batch, max_len=120):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.batch = batch

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        print('pe',pe.shape)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        self.batch = batch

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 batch_size,
                 dim_head = 64,
                 ff_chunks = 1,
                 ff_mult = 4,
                 ff_glu = False,
                 ff_dropout = 0.,
                 attn_dropout = 0.,
                 qkv_bias = True,
                ):
        super().__init__()
        self.depth = depth
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.pos_emb = PositionalEncoding(dim, 0.2, batch_size)

        wrapper_fn = partial(PreLayerNorm, dim)
        for _ in range(depth):
            self.attns.append(wrapper_fn(FullAttention(dim, heads, dim_head, attn_dropout, qkv_bias)))
            self.ffns.append(wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)))
    
    def forward(self, x):
        x += self.pos_emb(x)
        for i in range(self.depth):
            x = x + self.attns[i](x) # residual link
            x = x + self.ffns[i](x) # residual link
        return x

class CNNEncoder(nn.Module):
    def __init__(self, p_dropout=0.5):
        super(CNNEncoder, self).__init__()
        self.filter_number = 192             
        self.kernel_number = 4
        self.embed_size = 768
        self.activation = nn.ReLU()
        self.p_dropout = p_dropout
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.filter_number, kernel_size=(2,),
                               padding="same", padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")
        self.conv3 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.filter_number, kernel_size=(5,),
                               padding="same", padding_mode="zeros")
        self.conv4 = nn.Conv1d(in_channels=self.embed_size, out_channels=self.filter_number, kernel_size=(1,),
                               padding="same", padding_mode="zeros")

    def forward(self, bert_last_hidden):
        trans_embedded = torch.transpose(bert_last_hidden, dim0=1, dim1=2)
        convolve1 = self.activation(self.conv1(trans_embedded))
        convolve2 = self.activation(self.conv2(trans_embedded))
        convolve3 = self.activation(self.conv3(trans_embedded))
        convolve4 = self.activation(self.conv4(trans_embedded))
        convolve1 = torch.transpose(convolve1, dim0=1, dim1=2)
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        convolve3 = torch.transpose(convolve3, dim0=1, dim1=2)
        convolve4 = torch.transpose(convolve4, dim0=1, dim1=2)
        output = torch.cat((convolve4, convolve1, convolve2, convolve3), dim=2)
        return output