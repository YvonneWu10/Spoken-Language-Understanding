import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.Bert_Models import BertEncoder
from models.modules.attention import SimpleSelfAttention
from models.modules.transformer_hd_with_bert import TransHierarDecoder_withBERT


def make_model(opt):
    if 'bert_model_name' in opt:
        pretrained_model_opts = {
            'model': opt.pretrained_model,
            'fix': opt.fix_bert_model,
            'model_name': opt.bert_model_name,
            'dp': opt.bert_dropout
        }
    else:
        pretrained_model_opts = None

    opt.word_vocab_size = opt.enc_word_vocab_size

    return Transformer_STC(opt, pretrained_model_opts)


class Transformer_STC(nn.Module):
    '''WCN Transformer Semantic Tuple Classifier'''
    def __init__(self, opt, pretrained_model_opts):

        super(Transformer_STC, self).__init__()

        self.pretrained_model_opts = pretrained_model_opts

        # encoders
        if pretrained_model_opts is not None:
            bert_config = pretrained_model_opts['model'].config
            self.utt_sa_bert_encoder = BertEncoder(opt, bert_config, pretrained_model_opts, 0)

        self.dropout_layer = nn.Dropout(opt.dropout)

        self.device = opt.device
        self.sent_repr = opt.sent_repr
        self.cls_type = opt.cls_type

        # feature dimension
        fea_dim = opt.d_model
        if self.sent_repr in ['tok_sa_cls']: fea_dim *= 2

        if self.sent_repr in ['attn', 'tok_sa_cls']:
            self.slf_attn = SimpleSelfAttention(opt.d_model, opt.dropout, opt.device)

        self.fea_proj = nn.Linear(fea_dim, opt.d_model * 2)
        if pretrained_model_opts is not None:
            bert_emb = self.utt_sa_bert_encoder.embeddings
            self.tf_hd = TransHierarDecoder_withBERT(opt.dec_word_vocab_size, opt.act_vocab_size, opt.slot_vocab_size,
                opt.emb_size, opt.d_model, opt.hidden_size, opt.max_n_pos,
                opt.n_dec_layers, opt.n_head, opt.d_k, opt.d_v, opt.dropout, with_ptr=opt.with_ptr,
                decoder_tied=opt.decoder_tied,
                bert_emb=bert_emb)
        # tying the src & tgt word embedding
        if self.pretrained_model_opts is None and opt.word_vocab_size == opt.dec_word_vocab_size and opt.decoder_tied:
            self.utt_sa_encoder.src_word_emb.weight = self.tf_hd.decoder.tgt_word_emb.weight

        # weight initialization
        self.init_weight(opt.init_type, opt.init_range)

    def init_weight(self, init_type='uf', init_range=0.2):
        named_parameters = list(filter(
            lambda n_p: 'bert_encoder' not in n_p[0] and 'tf_hd.act_emb' not in n_p[0]
                and 'tf_hd.slot_emb' not in n_p[0] and 'tf_hd.decoder.embeddings' not in n_p[0],
            self.named_parameters()
        ))
        if init_type == 'uf':  # uniform
            for name, p in named_parameters:
                if p.dim() > 1 and 'position_enc' not in name:
                    nn.init.uniform_(p, a=-init_range, b=init_range)
        elif init_type == 'xuf':
            for name, p in named_parameters:
                if p.dim() > 1 and 'position_enc' not in name:
                    nn.init.xavier_uniform_(p)
        elif init_type == 'normal':
            for name, p in named_parameters:
                if p.dim() > 1 and 'position_enc' not in name and 'src_word_emb' not in name:
                    nn.init.normal_(p, mean=0, std=init_range)

    def get_sent_repr(self, enc_out, lens):
        if self.sent_repr == 'cls':
            sent_fea = enc_out[:, 0, :]  # (b, dm)
        elif self.sent_repr == 'maxpool':
            sent_fea = enc_out.max(1)[0]  # (b, dm)
        elif self.sent_repr == 'attn':  # self attn
            sent_fea = self.slf_attn(enc_out, lens)
        elif self.sent_repr == 'tok_sa_cls':
            cls_outs = enc_out[:, 0, :]
            seq_outs = enc_out[:, 1:, :]
            seq_lens = [l - 1 for l in lens]
            sent_fea = torch.cat([self.slf_attn(seq_outs, seq_lens), cls_outs], dim=1)
        else:
            raise RuntimeError('Wrong sent repr: %s' % (self.sent_repr))

        return sent_fea

    def encode_utt_sa_one_seq(self, inputs, masks):
        # inputs
        if self.pretrained_model_opts is not None:
            model_inputs_utt_sa = inputs['pretrained_inputs']
            lens = model_inputs_utt_sa['token_lens']
            self_mask = masks['self_mask']

        src_seq = model_inputs_utt_sa['tokens']

        # encoder
        if self.pretrained_model_opts is not None:
            lin_in, enc_out = self.utt_sa_bert_encoder(model_inputs_utt_sa, attention_mask=self_mask)  # enc_out -> (b, l, dm)

        # utterance-level feature
        # by default to be of type cls
        # lin_in = self.get_sent_repr(enc_out, lens)

        return lin_in, src_seq, enc_out

    def forward(self, inputs, masks):

        # encoder, lin_in is the [CLS] embedding, src_seq is the src string, and enc_out are output of the encoder
        # enc_out is encoded by BERT first then followed by a BiLSTM
        lin_in, src_seq, enc_out = self.encode_utt_sa_one_seq(inputs, masks)

        lin_in = self.fea_proj(self.dropout_layer(lin_in))
        act_inputs = inputs['hd_inputs']['act_inputs']
        act_slot_pairs = inputs['hd_inputs']['act_slot_pairs']
        value_inps = inputs['hd_inputs']['value_inps']
        if self.pretrained_model_opts is not None:
            value_seg = inputs['hd_inputs']['segments']
        enc_batch_extend_vocab_idx = inputs['hd_inputs']['enc_batch_extend_vocab_idx']
        oov_lists = inputs['hd_inputs']['oov_lists']

        act_scores_out = []
        slot_scores_out = []
        value_scores_out = []
        for i in range(len(oov_lists)):
            # whether there are oov word outside the training vocab
            if len(oov_lists[i]) == 0:
                extra_zeros = None
            else:
                extra_zeros = torch.zeros(1, len(oov_lists[i])).to(self.device)
            if self.pretrained_model_opts is not None:
                act_scores, slot_scores, value_scores = self.tf_hd(
                    src_seq[i:i+1], 
                    enc_out[i:i+1], lin_in[i:i+1],
                    act_inputs[i], act_slot_pairs[i], value_inps[i],
                    value_seg[i],
                    extra_zeros, enc_batch_extend_vocab_idx[i])
            act_scores_out.append(act_scores)
            slot_scores_out.append(slot_scores)
            value_scores_out.append(value_scores)

        scores = (act_scores_out, slot_scores_out, value_scores_out)
        return scores

    def decode_batch_tf_hd(self, inputs, masks, word_vocab, label_vocab, device, tokenizer=None):
        '''
        for cls_type == 'tf_hd'
        '''
        # inputs
        # encoder
        lin_in, src_seq, enc_out = self.encode_utt_sa_one_seq(inputs, masks)

        enc_batch_extend_vocab_idx = inputs['hd_inputs']['enc_batch_extend_vocab_idx']
        oov_lists = inputs['hd_inputs']['oov_lists']

        lin_in = self.fea_proj(lin_in)

        # Hierarchical decoder
        batch_pred_triples = []
        for i in range(len(oov_lists)):
            if len(oov_lists[i]) == 0:
                extra_zeros = None
            else:
                extra_zeros = torch.zeros(1, len(oov_lists[i])).to(device)

            utt_src = src_seq[i : i + 1]
            utt_enc_out = enc_out[i : i + 1]
            utt_fea = lin_in[i:i+1]
            utt_extend_idx = enc_batch_extend_vocab_idx[i]

            triples = self.tf_hd.decode(
                utt_src, utt_enc_out, utt_fea,
                extra_zeros, utt_extend_idx, word_vocab, label_vocab, oov_lists[i],
                device, tokenizer=tokenizer)
            batch_pred_triples.append(triples)

        return batch_pred_triples

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'),
                map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
