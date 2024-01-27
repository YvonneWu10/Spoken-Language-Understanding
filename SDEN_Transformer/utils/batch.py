#-*- coding:utf-8 -*-
import torch


def from_example_list(args, ex_list, device='cpu', train=True):
    # print(ex_list)
    # ex_list = sorted(ex_list, key=lambda x: max([len(i.input_idx) for i in x]), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx

    batch.utt = [[ex.utt for ex in conv] for conv in ex_list]
    input_lens = [max([len(ex.input_idx) for ex in conv]) for conv in ex_list]
    # print(input_lens)
    
    input_ids = []
    for i, conv in enumerate(ex_list):
        max_len = input_lens[i]
        conv_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in conv]
        conv_ids = torch.tensor(conv_ids, dtype=torch.long, device=device)
        input_ids.append(conv_ids)
        # print(i)
        # print(conv_ids.shape)
        # print(conv_ids)

    batch.input_ids = input_ids
    batch.lengths = input_lens
    batch.did = [[ex.did for ex in conv] for conv in ex_list]

    if train:
        batch.labels = [[ex.slotvalue for ex in conv] for conv in ex_list]
        tag_lens = [max([len(ex.tag_id) for ex in conv]) for conv in ex_list]

        tag_ids = []
        tag_mask = []
        for i, conv in enumerate(ex_list):
            max_tag_lens = tag_lens[i]
            conv_tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in conv]
            conv_tag_ids = torch.tensor(conv_tag_ids, dtype=torch.long, device=device)
            tag_ids.append(conv_tag_ids)
            conv_tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in conv]
            conv_tag_mask = torch.tensor(conv_tag_mask, dtype=torch.long, device=device)
            tag_mask.append(conv_tag_mask)
        batch.tag_ids = tag_ids
        batch.tag_mask = tag_mask

    else:
        batch.labels = None
        batch.tag_ids = None
        tag_mask = []
        for i, conv in enumerate(ex_list):
            max_tag_lens = tag_lens[i]
            conv_tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in conv]
            conv_tag_mask = torch.tensor(conv_tag_mask, dtype=torch.long, device=device)
            tag_mask.append(conv_tag_mask)
        batch.tag_mask = tag_mask

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]