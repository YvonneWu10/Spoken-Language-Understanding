from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import utils.Constants as Constants
import json
import re


def prepare_dataloader(data_path, word_vocab, label_vocab, batch_size, device, noise, shuffle_flag=False):
    dataset = json.load(open(data_path, 'r', encoding='UTF-8'))
    dataset = My_Dataset(dataset, word_vocab, label_vocab, device, noise)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        collate_fn=lambda batch: batch
    )

    return dataloader


def seq2extend_ids(lis, word_vocab):
    ids = []
    oovs = []
    for w in lis:
        if w in word_vocab.word2id:
            ids.append(word_vocab.word2id[w])
        else:
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(len(word_vocab) + oov_num)
    return ids, oovs


def value2ids(lis, word_vocab):
    ids = []
    for w in lis:
        if w in word_vocab.word2id:
            ids.append(word_vocab.word2id[w])
        else:
            ids.append(Constants.UNK)
    return ids


def value2extend_ids(lis, word_vocab, oovs):
    ids = []
    for w in lis:
        if w in word_vocab.word2id:
            ids.append(word_vocab.word2id[w])
        else:
            if w in oovs:
                ids.append(len(word_vocab.word2id) + oovs.index(w))
            else:
                ids.append(Constants.UNK)
    return ids


class My_Dataset(Dataset):
    def __init__(self, data, word_vocab, label_vocab, device, noise):
        super(My_Dataset, self).__init__()
        idx = 0
        utts = []
        in_idx_seqs = []
        acts = []
        act_slot_dict = []
        asv_dict = []
        label_lists = []
        max_len = 0
        pattern = re.compile(r'\(.*\)')

        for di, data in enumerate(data):
            for ui, raw in enumerate(data):
                if not noise:
                    try:
                        utt = re.sub(pattern, '', raw['manual_transcript'])
                    except:
                        utt = re.sub(pattern, '', raw['asr_1best'])
                else:
                    utt = re.sub(pattern, '', raw['asr_1best'])

                utt = ''.join(utt.split(' '))
                utt = utt.lower()

                utts.append(utt)
                act = []
                act_slot_d = defaultdict(list)

                slot = {}
                for label in raw['semantic']:
                    act_slot = f'{label[0]}-{label[1]}'
                    act.append(label[0])
                    act_slot_d[label[0]].append(label[1])
                    if len(label) == 3:
                        slot[act_slot] = label[2]

                # tags = ['O'] * len(utt)
                # for slo in slot:
                #     value = slot[slo]
                #     bidx = utt.find(value)
                #     if bidx != -1:
                #         tags[bidx: bidx + len(value)] = [f'I-{slo}'] * len(value)
                #         tags[bidx] = f'B-{slo}'

                slotvalue = [f'{slo}-{value}' for slo, value in slot.items()]
                label_lists.append(slotvalue)

                # input sequence to idx, if unknown, UNK
                input_idx = [word_vocab[c] for c in utt]
                in_idx_seqs.append(input_idx)

                # tag_id = [label_vocab.convert_tag_to_idx(tag) for tag in tags]

                max_len = max(max_len, len(utt))
                acts.append(act)
                act_slot_dict.append(act_slot_d)
                asv_dict.append(slot)

        cls = True

        # padding seqs
        batch_in = np.array([
            [Constants.CLS] * cls + seq + [Constants.PAD] * (max_len - len(seq))
            for seq in in_idx_seqs
        ])


        # final processing
        batch_in = torch.LongTensor(batch_in).to(device)

        #################### processing inputs & outputs for hierarchical decoding ####################
        # oov processing, oov_lists contain the oov word outside the training vocab
        oov_lists = []
        extend_ids = []
        for seq in utts:
            seq = Constants.CLS_WORD * cls + seq
            ids, oov = seq2extend_ids(seq, word_vocab)
            extend_ids.append(torch.tensor(ids).view(1, -1).to(device))
            oov_lists.append(oov)

        # act predictor labels
        acts_indices = [[label_vocab.act2idx[a] for a in acts_utt]
            for acts_utt in acts]
        acts_map = torch.zeros(len(utts), len(label_vocab.act2idx)).to(device)
        for i, a in enumerate(acts_indices):
            for idx in a:
                acts_map[i][idx] = 1

        act_inputs = [list(dic.keys()) for dic in act_slot_dict]
        act_inputs = [[label_vocab.act2idx[a] for a in acts_utt] for acts_utt in act_inputs]
        act_inputs = [torch.tensor(acts_utt).view(-1, 1).to(device) if len(acts_utt) > 0 else None
            for acts_utt in act_inputs]  # list: batch x tensor(#acts, 1)

        # slots_map = torch.zeros(len(batch), max_act_len, len(slot2idx))
        slots_map = []  # list: batch x tensor(#double_acts, #slots)
        for i, dic in enumerate(act_slot_dict):
            if len(dic) == 0:
                slots_map.append(None)
            else:
                tmp = torch.zeros(len(dic), len(label_vocab.slot2idx)).to(device)
                for j, (a, slots) in enumerate(dic.items()):
                    for s in slots:
                        if s in label_vocab.slot2idx:
                            tmp[j][label_vocab.slot2idx[s]] = 1
                        else:
                            tmp[j][Constants.PAD] = 1
                slots_map.append(tmp)

        act_slot_ids = []  # list: batch x tensor(#triple_act_slots, 2)
        value_inp_ids = []
        value_out_ids = []
        for i, dic in enumerate(asv_dict):
            if len(dic) == 0:
                act_slot_ids.append(None)
                value_inp_ids.append(None)
                value_out_ids.append(None)
            else:
                tmp = []
                tmp_v_inp, tmp_v_out = [], []
                for j, (a_s, value) in enumerate(dic.items()):
                    act_slot = a_s.strip().split('-')
                    a_id = label_vocab.act2idx[act_slot[0]]
                    s_id = label_vocab.slot2idx[act_slot[1]] \
                        if act_slot[1] in label_vocab.slot2idx else Constants.PAD
                    tmp.append([a_id, s_id])
                    inp_ids = value2ids(value.strip().split(), word_vocab)
                    out_ids = value2extend_ids(value.strip().split(), word_vocab, oov_lists[i])
                    tmp_v_inp.append(torch.tensor([Constants.BOS] + inp_ids).view(1, -1).to(device))
                    tmp_v_out.append(torch.tensor(out_ids + [Constants.EOS]).to(device))
                act_slot_ids.append(torch.tensor(tmp).to(device))
                value_inp_ids.append(tmp_v_inp)
                value_out_ids.append(tmp_v_out)

        self.batch_in = batch_in
        self.raw_in = list(utts)
        self.raw_labels = list(label_lists)
        self.act_labels = acts_map
        self.act_inputs = act_inputs
        self.slot_labels = slots_map
        self.act_slot_pairs = act_slot_ids
        self.value_inps = value_inp_ids
        self.value_outs = value_out_ids
        self.oov_lists = oov_lists


    def __len__(self):
        return len(self.raw_in)

    def __getitem__(self, index):
        batch_in = self.batch_in[index]
        raw_in = self.raw_in[index]
        raw_labels = self.raw_labels[index]
        act_labels = self.act_labels[index]
        act_inputs = self.act_inputs[index]
        slot_labels = self.slot_labels[index]
        act_slot_pair = self.act_slot_pairs[index]
        value_inps = self.value_inps[index]
        value_outs = self.value_outs[index]
        oov_lists = self.oov_lists[index]

        return batch_in, raw_in, raw_labels, act_labels, act_inputs, slot_labels, act_slot_pair, value_inps, value_outs, oov_lists

