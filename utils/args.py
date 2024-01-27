#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--unlabeled_data_path', default='./data/test_unlabelled.json', help='root of the testing unlabeled data')
    arg_parser.add_argument('--labeled_data_path', default='./data/test.json', help='result of the testing unlabeled data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--results_dir', default='./results', help='path for saving results')
    arg_parser.add_argument('--results_save', action='store_true', help='saving results')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    
    arg_parser.add_argument('--trainset_spoken_language_select', default='asr_1best', choices=['manual_transcript', 'asr_1best', 'both'], 
                            help='*sentence used for trainset(asr_1best: with noise; manual_transcript: without noise)')
    arg_parser.add_argument('--trainset_augmentation', action='store_true', help='*used augmented data from lexicon')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
    arg_parser.add_argument('--connection', default='Parallel', choices=['Parallel', 'Serial'], help='how the layers connected')
    arg_parser.add_argument('--CNN', action='store_true', help='use CNN structure')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer_attn', default=2, type=int, help='number of layer of attentions')
    arg_parser.add_argument('--num_layer_rnn', default=2, type=int, help='number of layer of RNN')
    
    #### Word vector model Hyperparams ####
    arg_parser.add_argument('--word_embedding', default='Word2vec', choices=['Word2vec', 'Bert','WWM', 'Roberta'], help='the method to caculate the word vector')
    return arg_parser