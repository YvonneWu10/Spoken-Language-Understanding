import os
import sys
import json
import time
import random
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers.optimization import AdamW

install_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(install_path)

from utils.util import make_logger, get_exp_dir_bert, merge_vocabs
from utils.vocab import *
from utils.fscore import update_f1, compute_f1
from utils.gpu_selection import auto_select_gpu
from utils.bert_xlnet_inputs import prepare_inputs_for_bert_xlnet, \
    prepare_inputs_for_bert_xlnet_act_slot_value
from utils.pos_util import get_sequential_pos
from utils.mask_util import prepare_mask
from models.model import make_model
from models.optimization import BertAdam
import utils.Constants as Constants
from utils.dataset import prepare_dataloader

def parse_arguments():
    parser = argparse.ArgumentParser()

    ######################### model structure #########################
    parser.add_argument('--emb_size', type=int, default=768, help='word embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=512, help='hidden layer dimension')
    parser.add_argument('--max_seq_len', type=int, default=50, help='max sequence length')
    parser.add_argument('--max_n_pos', type=int, default=100, help='max value of position')
    parser.add_argument('--n_layers', type=int, default=3, help='#transformer layers')
    parser.add_argument('--n_dec_layers', type=int, default=1, help='#transformer decoder layers')
    parser.add_argument('--n_head', type=int, default=12, help='#attention heads')
    parser.add_argument('--d_k', type=int, default=64, help='dimension of k in attention')
    parser.add_argument('--d_v', type=int, default=64, help='dimension of v in attention')
    parser.add_argument('--cls_type', default='tf_hd', choices=['tf_hd'], help='classifier type')
    parser.add_argument('--decoder_tied', action='store_true', help='whether to tie the act/slot embedding to their output layers')
    parser.add_argument('--sent_repr', default="cls", help='sentence level representatoin')
    parser.add_argument('--encoder_cell', default='RNN', choices=['LSTM', 'GRU', 'RNN'], help='the type of the network')

    ######################### about hierarchical decoder #########################
    parser.add_argument('--with_ptr', type=bool, default=True, help='whether to apply pointer mechanism')

    ######################### data & vocab #########################
    parser.add_argument('--dataset', default='SJTU', help='<domain>')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--train_file', default='train.json', help='base file name of train dataset')
    parser.add_argument('--valid_file', default='development.json', help='base file name of valid dataset')
    parser.add_argument('--test_file', default='test_unlabelled.json', help='base file name of test dataset')
    parser.add_argument('--ontology_path', default='/home/huangyixin/NLP/data/ontology.json', help='ontology')
    parser.add_argument('--noise', type=bool, default=False)

    ######################## pretrained model (BERT/XLNET) ########################
    parser.add_argument('--bert_model_name', default='../bert-base-chinese',
        choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased', 'bert-base-chinese'])
    parser.add_argument('--fix_bert_model', action='store_true')

    ######################### training & testing options #########################
    parser.add_argument('--testing', type=bool, default=False, help=' test your model (default is training && testing)')
    parser.add_argument('--deviceId', type=int, default=0, help='train model on ith gpu. -1:cpu, 0:auto_select')
    parser.add_argument('--random_seed', type=int, default=999, help='initial random seed')
    parser.add_argument('--l2', type=float, default=0, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='dropout rate for BERT')
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--max_norm', type=float, default=5.0, help='threshold of gradient clipping (2-norm)')
    parser.add_argument('--max_epoch', type=int, default=150, help='max number of epochs to train')
    parser.add_argument('--experiment', default='exp', help='experiment directories for storing models and logs')
    parser.add_argument('--optim_choice', default='bertadam', choices=['schdadam', 'adam', 'adamw', 'bertadam'], help='optimizer choice')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--bert_lr', default=1e-5, type=float, help='learning rate for bert')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='warmup propotion')
    parser.add_argument('--init_type', default='uf', choices=['uf', 'xuf', 'normal'], help='init type')
    parser.add_argument('--init_range', type=float, default=0.2, help='init range, for naive uniform')
    # parser.add_argument('--model_path', type=str, default='/home/huangyixin/NLP/HD_with_Bert/exp/data_SJTU/23-12-31 09:36:56__dp_0.0_0.3__opt_bertadam_0.1_0.0005_1e-05__seed_999__encoder_RNN', help='the directory of the model path to be tested')
    parser.add_argument('--trans_layer', type=int, default=3, help='number of layers in the transformer in the encoder')

    opt = parser.parse_args()

    ######################### option verification & adjustment #########################
    # device definition
    if opt.deviceId >= 0:
        if opt.deviceId > 0:
            opt.deviceId, gpu_name, valid_gpus = auto_select_gpu(assigned_gpu_id=opt.deviceId - 1)
        elif opt.deviceId == 0:
            opt.deviceId, gpu_name, valid_gpus = auto_select_gpu()
        print('Valid GPU list: %s ; GPU %d (%s) is auto selected.' % (valid_gpus, opt.deviceId, gpu_name))
        torch.cuda.set_device(opt.deviceId)
        opt.device = torch.device('cuda')
    else:
        print('CPU is used.')
        opt.device = torch.device('cpu')

    # random seed set
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.random_seed)

    # d_model: just equals embedding size
    opt.d_model = opt.emb_size

    # ontology
    opt.ontology = None if opt.ontology_path is None else \
        json.load(open(opt.ontology_path))

    return opt


def train_epoch(model, data, opt, word_vocab, label_vocab):
    '''Epoch operation in training phase'''

    model.train()
    opt.optimizer.zero_grad()

    TP, FP, FN = 0, 0, 0
    corr, tot = 0, 0
    losses = []
    total_loss_num_pairs = [[0., 0.], [0., 0.], [0., 0.]]

    for step, batch in enumerate(data):
        # prepare data
        batch = sorted(batch, key=lambda batch: len(batch[1]), reverse=True)
        batch_in, raw_in, raw_labels, \
            act_labels, act_inputs, slot_labels, act_slot_pairs, value_inps, value_outs, \
            oov_lists = zip(*batch)
        raw_lens = [len(utt) for utt in raw_in]

        # prepare inputs for BERT/XLNET
        inputs = {}
        pretrained_inputs = prepare_inputs_for_bert_xlnet(
            raw_in, raw_lens, opt.tokenizer,
            cls_token_at_end=False,
            cls_token=opt.tokenizer.cls_token,
            sep_token=opt.tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0,
            device=opt.device
        )
        inputs['pretrained_inputs'] = pretrained_inputs
        masks = prepare_mask(pretrained_inputs)

        pretrained_inputs_hd = prepare_inputs_for_bert_xlnet_act_slot_value(
            act_inputs, act_slot_pairs, value_inps, value_outs, raw_labels,
            word_vocab, label_vocab, opt.tokenizer,
            cls_token_at_end=False,
            cls_token=opt.tokenizer.cls_token,
            sep_token=opt.tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0,
            device=opt.device
        )
        tokens = pretrained_inputs['tokens']
        extend_ids = [tokens[j].unsqueeze(0) for j in range(len(raw_in))]

        pretrained_inputs_hd['enc_batch_extend_vocab_idx'] = extend_ids
        pretrained_inputs_hd['oov_lists'] = oov_lists = [[] for _ in range(len(batch_in))]

        inputs['hd_inputs'] = pretrained_inputs_hd

        # forward
        batch_preds = model(inputs, masks)
        act_scores, slot_scores, value_scores = batch_preds
        batch_gold_values = pretrained_inputs_hd['value_outs']

        batch_loss_num_pairs = [[0., 0.], [0., 0.], [0., 0.]]
        for i in range(len(oov_lists)):
            # act loss
            act_loss = opt.class_loss_function(act_scores[i], act_labels[i].unsqueeze(0))
            batch_loss_num_pairs[0][0] += act_loss
            batch_loss_num_pairs[0][1] += 1

            # NOTE: loss normalization for slot_loss & value_loss
            # slot loss
            if slot_scores[i] is not None:
                # print(slot_scores[i], slot_labels[i])
                slot_loss = opt.class_loss_function(slot_scores[i], slot_labels[i])
                batch_loss_num_pairs[1][0] += slot_loss / slot_scores[i].size(0)
                batch_loss_num_pairs[1][1] += 1

            # value loss
            if value_scores[i] is not None:
                gold_values = batch_gold_values[i]
                sum_value_lens = gold_values.gt(0).sum().item()
                value_loss = opt.nll_loss_function(
                    value_scores[i].contiguous().view(
                        -1, opt.dec_word_vocab_size + len(oov_lists[i]) * opt.with_ptr),
                    gold_values.contiguous().view(-1))
                batch_loss_num_pairs[2][0] += value_loss / sum_value_lens
                batch_loss_num_pairs[2][1] += 1

        total_loss = 0.
        loss_ratios = [1, 1, 1]
        for k, loss_num in enumerate(batch_loss_num_pairs):
            if loss_num[1] > 0:
                total_loss += (loss_num[0] / loss_num[1]) * loss_ratios[k]
        total_loss.backward()

        if (step + 1) % opt.n_accum_steps == 0:
            # clip gradient
            if opt.optim_choice.lower() != 'bertadam' and opt.max_norm > 0:
                params = list(filter(lambda p: p.requires_grad, list(model.parameters())))
                torch.nn.utils.clip_grad_norm_(params, opt.max_norm)

            # update parameters
            if opt.optim_choice.lower() in ['adam', 'bertadam']:
                opt.optimizer.step()
            elif opt.optim_choice.lower() == 'adamw':
                opt.optimizer.step()
                opt.scheduler.step()

            # clear gradients
            opt.optimizer.zero_grad()

        for i in range(len(batch_loss_num_pairs)):
            if isinstance(batch_loss_num_pairs[i][0], float):
                total_loss_num_pairs[i][0] += batch_loss_num_pairs[i][0]
            else:
                total_loss_num_pairs[i][0] += batch_loss_num_pairs[i][0].item()
            total_loss_num_pairs[i][1] += batch_loss_num_pairs[i][1]

        # calculate performance
        batch_preds = model.decode_batch_tf_hd(inputs, masks, word_vocab, label_vocab, opt.device, tokenizer=opt.tokenizer)
        for pred_classes, gold in zip(batch_preds, raw_labels):
            TP, FP, FN = update_f1(pred_classes, gold, TP, FP, FN)
            tot += 1
            if set(pred_classes) == set(gold):
                corr += 1

    act_avg_loss = total_loss_num_pairs[0][0] / total_loss_num_pairs[0][1]
    slot_avg_loss = total_loss_num_pairs[1][0] / total_loss_num_pairs[1][1]
    value_avg_loss = total_loss_num_pairs[2][0] / total_loss_num_pairs[2][1]
    total_avg_loss = act_avg_loss + slot_avg_loss + value_avg_loss
    losses = (act_avg_loss, slot_avg_loss, value_avg_loss, total_avg_loss)
    p, r, f = compute_f1(TP, FP, FN)
    acc = corr / tot * 100

    return losses, (p, r, f), acc


def eval_epoch(model, data, opt, word_vocab, label_vocab, fp, efp, is_test=False):
    '''Epoch operation in evaluating phase'''

    model.eval()

    TP, FP, FN = 0, 0, 0
    corr, tot = 0, 0
    losses = []

    all_cases = []
    err_cases = []
    utt_id = 0

    for j, batch in enumerate(data):
        batch = sorted(batch, key=lambda batch: len(batch[1]), reverse=True)
        batch_in, raw_in, raw_labels, \
            act_labels, act_inputs, slot_labels, act_slot_pairs, value_inps, value_outs, \
            oov_lists = zip(*batch)
        raw_lens = [len(utt) for utt in raw_in]

        # prepare inputs for BERT/XLNET
        inputs = {}
        pretrained_inputs = prepare_inputs_for_bert_xlnet(
            raw_in, raw_lens, opt.tokenizer,
            cls_token_at_end=False,
            cls_token=opt.tokenizer.cls_token,
            sep_token=opt.tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0,
            device=opt.device
        )
        inputs['pretrained_inputs'] = pretrained_inputs
        masks = prepare_mask(pretrained_inputs)

        pretrained_inputs_hd = prepare_inputs_for_bert_xlnet_act_slot_value(
            act_inputs, act_slot_pairs, value_inps, value_outs, raw_labels, word_vocab, label_vocab, opt.tokenizer,
            cls_token_at_end=False,
            cls_token=opt.tokenizer.cls_token,
            sep_token=opt.tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0,
            device=opt.device
        )
        tokens = pretrained_inputs['tokens']
        extend_ids = [tokens[j].unsqueeze(0) for j in range(len(raw_in))]
        pretrained_inputs_hd['enc_batch_extend_vocab_idx'] = extend_ids
        pretrained_inputs_hd['oov_lists'] = oov_lists = [[] for _ in range(len(batch_in))]

        inputs['hd_inputs'] = pretrained_inputs_hd

        # forward
        batch_preds = model.decode_batch_tf_hd(inputs, masks, word_vocab, label_vocab, opt.device, tokenizer=opt.tokenizer)

        # calculate performance
        batch_pred_classes = []
        batch_ids = []
        for pred_classes, gold, raw in zip(batch_preds, raw_labels, raw_in):

            TP, FP, FN = update_f1(pred_classes, gold, TP, FP, FN)
            tot += 1
            if set(pred_classes) == set(gold):
                corr += 1

            batch_pred_classes.append(pred_classes)

            batch_ids.append(utt_id)
            utt_id += 1

            # keep intermediate results
            res_info = '%s\t<=>\t%s\t<=>\t%s\n' % (
                ''.join(raw), ';'.join(pred_classes), ';'.join(gold))
            fp.write(res_info)
            if set(pred_classes) != set(gold):
                efp.write(res_info)
                err_cases.append((raw, pred_classes, gold))
            all_cases.append((raw, pred_classes, gold))

    # mean_loss = np.mean(losses)
    p, r, f = compute_f1(TP, FP, FN)
    acc = corr / tot * 100

    # err_analysis(err_cases)

    if is_test:
        return (p, r, f), acc, all_cases
    else:
        return (p, r, f), acc


def train(model, train_dataloader, valid_dataloader, opt, word_vocab, label_vocab):
    '''Start training'''

    logger = make_logger(os.path.join(opt.exp_dir, 'log.train'))
    t0 = time.time()
    logger.info('Training starts at %s' % (time.asctime(time.localtime(time.time()))))

    best = {'epoch': 0, 'vf': 0., 'tef': 0.}

    for i in range(opt.max_epoch):
        # evaluating train set
        start = time.time()
        train_loss, (trp, trr, trf), tr_acc = train_epoch(model, train_dataloader, opt, word_vocab, label_vocab)
        logger.info('[Train]\tEpoch: %02d\tTime: %.2f\tLoss: (%.2f/%.2f/%.2f/%.2f)\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (i, time.time()-start, *train_loss, trp, trr, trf, tr_acc))

        # evaluating valid set
        with open(os.path.join(opt.exp_dir, 'valid.iter%d'%i), 'w') as fp, \
                open(os.path.join(opt.exp_dir, 'valid.iter%d.err'%i), 'w') as efp:
            start = time.time()
            (vp, vr, vf), v_acc = eval_epoch(model, valid_dataloader, opt, word_vocab, label_vocab, fp, efp)
            logger.info('[Valid]\tEpoch: %02d\tTime: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
                (i, time.time()-start, vp, vr, vf, v_acc))

        # save model
        if vf > best['vf']:
            best['epoch'] = i
            best['vf'] = vf
            best['v_acc'] = v_acc
            model.save_model(os.path.join(opt.exp_dir, 'model.pt'))
            torch.save(opt.optimizer.state_dict(), open(os.path.join(opt.exp_dir, 'optim.pt'),'wb'))
            logger.info('NEW BEST:\tEpoch: %02d\tvalid F1/Acc: %.2f/%.2f\t' % (
                i, vf, v_acc))

    logger.info('Done training. Elapsed time: %s' % timedelta(seconds=time.time() - t0))
    logger.info('BEST RESULT:\tEpoch: %02d\tBest valid F1/Acc: %.2f/%.2f\t' % (
        best['epoch'], best['vf'], best['v_acc']))


def test(model, test_dataloader, opt, word_vocab, label_vocab):
    '''Start testing'''
    logger = make_logger(os.path.join(opt.exp_dir, 'log.test'))
    t0 = time.time()
    logger.info('Testing starts at %s' % (time.asctime(time.localtime(time.time()))))

    with open(os.path.join(opt.exp_dir, 'valid.iter'), 'w') as fp, open(os.path.join(opt.exp_dir, 'valid.iter.err'), 'w') as efp:
        start = time.time()
        (vp, vr, vf), v_acc = eval_epoch(model, valid_dataloader, opt, word_vocab, label_vocab, fp, efp)
        logger.info('[Valid]\tTime: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (time.time()-start, vp, vr, vf, v_acc))

    # evaluating test set
    with open(os.path.join(opt.exp_dir, 'test.eval'), 'w') as fp, \
            open(os.path.join(opt.exp_dir, 'test.eval.err'), 'w') as efp:
        start = time.time()
        (tep, ter, tef), te_acc, test_all_cases = eval_epoch(model, test_dataloader, opt, word_vocab, label_vocab, fp, efp, is_test=True)
        logger.info('[Test]\tTime: %.2f\t(p/r/f): (%.2f/%.2f/%.2f)\tAcc: %.2f' %
            (time.time()-start, tep, ter, tef, te_acc))
        
    print(test_all_cases)
    test_json = json.load(open(os.path.join(opt.dataroot, opt.test_file), 'r'))
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in test_all_cases[ei][1]]

    json.dump(test_json, open(os.path.join(opt.dataroot, 'prediction.json'), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    logger.info('Done testing. Elapsed time: %s' % timedelta(seconds=time.time() - t0))


if __name__ == '__main__':
    opt = parse_arguments()

    # prepare to load the pretrained model
    pretrained_model_class, tokenizer_class = BertModel, BertTokenizer
    # load the pretrained tokenizer and model
    opt.tokenizer = tokenizer_class.from_pretrained(opt.bert_model_name)
    opt.pretrained_model = pretrained_model_class.from_pretrained(
        opt.bert_model_name,
        output_hidden_states=opt.fix_bert_model
    ).to(opt.device)
    opt.tokenizer.add_special_tokens({'bos_token': Constants.BOS_WORD, 'eos_token': Constants.EOS_WORD})
    opt.pretrained_model.resize_token_embeddings(len(opt.tokenizer))
    word_vocab = Vocab(padding=True, unk=True, filepath=os.path.join(opt.dataroot, opt.train_file))
    label_vocab = LabelVocab(opt.dataroot)

    opt.enc_word_vocab_size = len(opt.tokenizer)  # subword-level
    opt.dec_word_vocab_size = len(opt.tokenizer)
    opt.act_vocab_size = len(label_vocab.act2idx.keys())
    opt.slot_vocab_size = len(label_vocab.slot2idx.keys())

    # exp dir
    opt.exp_dir = get_exp_dir_bert(opt)
    if not opt.testing and not os.path.exists(opt.exp_dir):
        os.makedirs(opt.exp_dir)

    # model definition & num of params
    model = make_model(opt)
    model = model.to(opt.device)

    trainable_parameters = list(filter(lambda p: p[1].requires_grad, model.named_parameters()))
    n_params = sum([np.prod(p.size()) for n, p in trainable_parameters])
    bert_parameters = list(filter(lambda n_p: 'bert_encoder' in n_p[0] or 'tf_hd.act_emb' in n_p[0] or 'tf_hd.slot_emb' in n_p[0]
        or 'tf_hd.decoder.embeddings' in n_p[0], trainable_parameters))
    n_bert_params = sum([np.prod(p.size()) for n, p in bert_parameters])
    # print(model)
    print('num params: {}'.format(n_params))
    print('num bert params: {}, {}%'.format(n_bert_params, 100 * n_bert_params / n_params))

    # dataloader preparation
    opt.n_accum_steps = 4 if opt.n_layers == 12 else 1

    train_dataloader = prepare_dataloader(os.path.join(opt.dataroot, opt.train_file), word_vocab, label_vocab, int(opt.batchSize // opt.n_accum_steps), opt.device, opt.noise, shuffle_flag=True)
    valid_dataloader = prepare_dataloader(os.path.join(opt.dataroot, opt.valid_file), word_vocab, label_vocab, int(opt.batchSize // opt.n_accum_steps), opt.device, True, shuffle_flag=False)
    test_dataloader = prepare_dataloader(os.path.join(opt.dataroot, opt.test_file), word_vocab, label_vocab, int(opt.batchSize // opt.n_accum_steps), opt.device, True, shuffle_flag=False)

    # optimizer
    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    named_params = list(model.named_parameters())
    named_params = list(filter(lambda p: p[1].requires_grad, named_params))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    is_bert = lambda name: 'bert_encoder' in name or 'tf_hd.act_emb' in name or 'tf_hd.slot_emb' in name or 'tf_hd.decoder.embeddings' in name
    is_decay = lambda name: not any(nd in name for nd in no_decay)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if is_decay(n) and is_bert(n)], 'weight_decay': 0.01, 'lr': opt.bert_lr},
        {'params': [p for n, p in named_params if not is_decay(n) and is_bert(n)], 'weight_decay': 0.0, 'lr': opt.bert_lr},
        {'params': [p for n, p in named_params if is_decay(n) and not is_bert(n)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if not is_decay(n) and not is_bert(n)], 'weight_decay': 0.0},
    ]

    if opt.optim_choice == 'adam':
        opt.optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.l2)
    elif opt.optim_choice.lower() == 'bertadam':
        num_train_optimization_steps = (len(train_dataloader.dataset) // opt.batchSize + 1) * opt.max_epoch
        opt.optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=opt.lr, warmup=opt.warmup_proportion,
            t_total=num_train_optimization_steps
        )
    elif opt.optim_choice.lower() == 'adamw':
        num_train_optimization_steps = (len(train_dataloader.dataset) // opt.batchSize + 1) * opt.max_epoch
        opt.optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, correct_bias=False)
        opt.scheduler = get_linear_schedule_with_warmup(
            opt.optimizer,
            num_warmup_steps=int(opt.warmup_proportion * num_train_optimization_steps),
            num_training_steps=num_train_optimization_steps
        )  # PyTorch scheduler

    # loss functions
    opt.class_loss_function = nn.BCELoss(reduction='sum')
    opt.nll_loss_function = nn.NLLLoss(reduction='sum', ignore_index=Constants.PAD)

    # training or testing
    if opt.testing:
        model.load_model(os.path.join(opt.exp_dir, 'model.pt'))
        test(model, test_dataloader, opt, word_vocab, label_vocab)
    else:
        train(model, train_dataloader, valid_dataloader, opt, word_vocab, label_vocab)
