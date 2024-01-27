import os
import sys
import logging
import datetime


def make_logger(fn, noStdout=False):
    logFormatter = logging.Formatter('%(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(fn, mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    if not noStdout:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
    return logger


def get_exp_dir_bert(opt):

    dataset_path = 'data_%s' % (opt.dataset)

    # exp_dir_list = [str(datetime.datetime.now())[2:][:-7]]
    exp_dir_list = []
    exp_dir_list.append('dp_%s_%s' % (opt.dropout, opt.bert_dropout))
    lr_str = '%s_%s' % (opt.lr, opt.bert_lr)
    if 'finetune_lr' in opt:
        lr_str += '_%s_%s' % (opt.finetune_lr, opt.finetune_bert_lr)
    exp_dir_list.append('opt_%s_%s_%s' % (
        opt.optim_choice, opt.warmup_proportion, lr_str))
    exp_dir_list.append('seed_%s' % opt.random_seed)
    exp_dir_list.append('encoder_%s' % opt.encoder_cell)

    exp_dir_list.append('n_layers_%s' % opt.n_layers)
    exp_dir_list.append('trans_layers_%s' % opt.trans_layer)
    if not opt.noise:
        exp_dir_list.append('manual')
    else:
        exp_dir_list.append('asr')
    exp_name = '__'.join(exp_dir_list)

    return os.path.join(opt.experiment, dataset_path, exp_name)


def merge_vocabs(v1, v2):
    for word in v2.keys():
        if word not in v1:
            idx = len(v1)
            v1[word] = idx
    return v1


def extend_vocab_with_sep(word2idx):
    sep_token = '<sep>'
    sep_id = len(word2idx)
    word2idx[sep_token] = sep_id
    return word2idx
