#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertModel


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()

        self.config = config

        self._memory = []
        self._memory_len = 128

        self.bert_embed = BertModel.from_pretrained("../bert")
        for param in self.bert_embed.parameters():
            param.requires_grad = False

        self.c_rnn = getattr(nn, config.c_encoder_cell)(config.embed_size, self._memory_len // 2, num_layers=1, bidirectional=True, batch_first=True, dropout=config.dropout)
        self.m_rnn = getattr(nn, config.m_encoder_cell)(config.embed_size, self._memory_len // 2, num_layers=1, bidirectional=True, batch_first=True, dropout=config.dropout)

        self.ff_layer = nn.Linear(self._memory_len * 2, self._memory_len)
        self.session_encoder = nn.GRU(self._memory_len, self._memory_len // 2, batch_first=True, bidirectional=True)

        self.knowledge_encoder = nn.Linear(self._memory_len, config.embed_size)

        self.output_layer = TaggingFNNDecoder(config.embed_size, config.num_tags, config.tag_pad_idx, config)
    
    def clear_memory(self):
        self._memory.clear()

    def forward(self, batch, ci, ui):
        if batch.tag_ids:
            tag_ids = batch.tag_ids[ci][ui]
        else:
            tag_ids = None
        tag_mask = batch.tag_mask[ci][ui]
        input_ids = torch.unsqueeze(batch.input_ids[ci][ui], dim=0)

        embed = self.bert_embed(input_ids)[0][0]
        
        u = self.c_rnn(embed)[0]
        c_memory = self.m_rnn(embed)[0]

        if len(self._memory) != 0:
            G = []
            p = torch.zeros(len(self._memory))
            for i in range(len(self._memory)):
                p[i] = torch.inner(torch.flatten(self._memory[i]), torch.flatten(u))
                concat = torch.concat((self._memory[i], u), dim=1)
                g = self.ff_layer(concat)
                G.append(g)
            p = torch.softmax(p, dim=0)
            G = sum(G[i] * p[i] for i in range(len(self._memory)))
            h = self.session_encoder(G)[0]
        else:
            h = torch.zeros(self._memory_len)
        h = h.to(batch.device)


        o = self.knowledge_encoder(u + h).to(batch.device)

        self._memory.append(c_memory.detach())

        tag_output = self.output_layer(embed + o, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch, ci, ui):
        batch_size = len(batch)

        output = self.forward(batch, ci, ui)
        prob = output[0]

        pred = torch.argmax(prob, dim=-1).cpu().tolist()
        pred_tuple = []
        idx_buff, tag_buff, pred_tags = [], [], []
        pred = pred[:len(batch.utt[ci][ui])]

        for idx, tid in enumerate(pred):
            tag = label_vocab.convert_idx_to_tag(tid)
            pred_tags.append(tag)

            if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[ci][ui][j] for j in idx_buff])

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
            value = ''.join([batch.utt[ci][ui][j] for j in idx_buff])
            pred_tuple.append(f'{slot}-{value}')

        if len(output) == 1:
            return pred_tuple
        else:
            labels = batch.labels[ci][ui]
            loss = output[1]
            return pred_tuple, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id, config):
        super(TaggingFNNDecoder, self).__init__()
        self.config = config
        self.num_tags = num_tags
        self.hidden_size = 128
        self.rnn = getattr(nn, config.tagger_rnn)(input_size, self.hidden_size // 2,
                                              num_layers=config.num_layer, batch_first=True, bidirectional=True, dropout=config.dropout)
        self.output_layer = nn.Linear(self.hidden_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, x, mask, labels=None):
        x = self.rnn(x)[0]
        logits = self.output_layer(x)

        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags).squeeze(dim=0) * -1e32
        prob = torch.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss

        return (prob, )
