import json

from utils.vocab import Vocab, LabelVocab
# from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("../bert")

class Example():

    @classmethod
    def configuration(cls, root, train_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        # cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, flag):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(dataset):
            conversation = []
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', flag)
                conversation.append(ex)
            examples.append(conversation)
        return examples

    def __init__(self, ex: dict, did, flag):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did
        
        assert flag in ['asr', 'manual']
        if flag == "manual":
            self.utt = ex["manual_transcript"]
        else:
            self.utt = ex['asr_1best']

        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)

        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        token = [c for c in self.utt]
        self.input_idx = tokenizer.convert_tokens_to_ids(token)
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
