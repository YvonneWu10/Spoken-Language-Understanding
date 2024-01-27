''' Define the Bert-like Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import torch.nn.functional as F
import math
from einops import rearrange
from functools import partial

def default(val, d):
    if (val is not None):
        return val
    else:
        return d

def get_slf_attn_w(score_seq):
    _, l = score_seq.size()
    attn_w = score_seq.unsqueeze(1).expand(-1, l, -1)  # b x l x l
    return attn_w

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    pad_id = 0
    padding_mask = seq_k.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask



class BertSelfAttention(nn.Module):
    def __init__(self, bert_config, bert_self_attention):
        super().__init__()

        self.rm_qkv = False

        if not self.rm_qkv:
            self.query = copy.deepcopy(bert_self_attention.query)
            self.key = copy.deepcopy(bert_self_attention.key)
            self.value = copy.deepcopy(bert_self_attention.value)

        self.num_attention_heads = bert_config.num_attention_heads
        self.attention_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(bert_config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask):

        if not self.rm_qkv:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            mixed_query_layer = hidden_states
            mixed_key_layer = hidden_states
            mixed_value_layer = hidden_states

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # (b, nh, l, l)

        # pad mask
        attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)  # (b, nh, l, l)
        attention_scores = attention_scores.masked_fill(attention_mask, -np.inf)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # (b, nh, l, dk)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (b, l, nh, dk)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # (b, l, d)

        return context_layer, attention_probs


class BertAttention(nn.Module):
    def __init__(self, bert_config, bert_attention, dp):
        super().__init__()
        self.output = copy.deepcopy(bert_attention.output)
        if dp != 0.1:
            self.output.dropout = nn.Dropout(dp)

        bert_self_attention = copy.deepcopy(bert_attention.self)
        self.self = BertSelfAttention(bert_config, bert_self_attention)

    def forward(self, hidden_states, attention_mask, head_mask):
        self_outputs = self.self(hidden_states, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, bert_config, bert_layer, dp):
        super().__init__()
        self.is_decoder = False
        self.intermediate = copy.deepcopy(bert_layer.intermediate)
        self.output = copy.deepcopy(bert_layer.output)
        if dp != 0.1:
            self.output.dropout = nn.Dropout(dp)

        bert_attention = copy.deepcopy(bert_layer.attention)
        self.attention = BertAttention(bert_config, bert_attention, dp)

    def forward(self, hidden_states, attention_mask, head_mask):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, outputs


class BertEncoder(nn.Module):
    def __init__(self, opt, bert_config, pretrained_model_opts,
            n_layers):
        super().__init__()
        self.opt = opt
        self.bert_config = bert_config
        self.pretrained_model_opts = pretrained_model_opts
        self.n_layers = n_layers

        # load bert
        bert_model = pretrained_model_opts['model']
        dp = pretrained_model_opts['dp']

        self.embeddings = copy.deepcopy(bert_model.embeddings)
        if dp != 0.1:
            self.embeddings.dropout = nn.Dropout(dp)

        bert_layers = copy.deepcopy(bert_model.encoder.layer)
        self.layer_stack = nn.ModuleList([
            BertLayer(bert_config, bert_layers[i], dp)
            for i in range(self.n_layers)
        ])
        # the BiLSTM, maintain the same shape after going through this utterance encoder
        self.rnn = getattr(nn, opt.encoder_cell)(opt.emb_size, opt.emb_size // 2, num_layers=opt.n_layers, bidirectional=True, batch_first=True)

        # transformer related layers
        self.fc = nn.Linear(opt.emb_size, opt.emb_size)
        self.transformer = Transformer(opt.emb_size, opt.trans_layer, 2, opt.batchSize)
        self.norm = nn.LayerNorm(opt.emb_size)

    def forward(self, inputs, attention_mask=None, return_attns=False):
        src_seq, src_type, mask, lengths = \
            inputs['tokens'], inputs['segments'], inputs['mask'], inputs['token_lens']

        # encode to word vector using BERT
        embed = self.pretrained_model_opts['model'](src_seq, src_type, mask)[0]
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        enc_output, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) # enc_out: bsize, 
        # seqlen, dim 

        lin_in = torch.cat((enc_output[:, 0, :self.opt.emb_size // 2], enc_output[:, -1, self.opt.emb_size // 2:]), dim=-1)
        # print(lin_in.shape)
        # enc_output_ = self.embeddings(src_seq, src_type, mask)  # (b, l, d)

        # if return_attns:
            # return enc_output, enc_slf_attn_list

        trans_input = self.fc(embed)
        trans_out = self.transformer(trans_input)
        enc_output = self.norm(trans_out + enc_output)

        return lin_in, enc_output


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
    def __init__(self, d_model, dropout, batch, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.batch = batch
        self.max_len = max_len

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
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
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