import json
from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.bert2vec import BertUtils, BERT_WWM_EXT, Roberta
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root=None, train_path=None, word2vec_path=None,
                      spoken_language_select = 'asr_1best', word_embedding = 'Word2vec'):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path,
                               spoken_language_select=spoken_language_select)
        
        if word_embedding == 'Word2vec' :
            cls.word2vec = Word2vecUtils(word2vec_path)  
        elif word_embedding == 'Bert' :
            cls.word2vec = BertUtils(None)
        elif word_embedding == 'WWM':
            cls.word2vec=BERT_WWM_EXT(None)
        elif word_embedding == 'Roberta':
            cls.word2vec=Roberta(None)
        
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, spoken_language_select='asr_1best'):
        
        if isinstance(data_path, str):
            data_path = [data_path]
        if isinstance(spoken_language_select, str):
            spoken_language_select = [spoken_language_select]
        
        examples = []
        for sls in spoken_language_select:
            # assert spoken_language_select in ['asr_1best', 'manual_transcript']
            assert sls in ['asr_1best', 'manual_transcript']
            for path in data_path:
                datas = json.load(open(path, 'r'))
                for data in datas:
                    for utt in data:
                        ex = cls(utt, sls)
                        examples.append(ex)
            
        return examples

    def __init__(self, ex: dict, spoken_language_select):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex[spoken_language_select]

        self.slot = {}
        if 'semantic' in ex:
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
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
    
    def __str__(self):
        return f"vocab seq: {self.utt}\ntag seq: {self.tags}\nvocab seq(index): {self.input_idx}\ntag seq(index): {self.tag_id}"
