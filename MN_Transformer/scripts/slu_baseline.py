# coding: utf-8

import sys, os, time, gc, json
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...", flush=True)
print("Random seed is set to %d" % (args.seed), flush=True)
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device", flush=True)

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path, flag="manual")
dev_dataset = Example.load_dataset(dev_path, flag="asr")
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time), flush=True)
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)), flush=True)

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
# check_point = torch.load(open('best_models/model12.bin', 'rb'), map_location=device)
# model.load_state_dict(check_point["model"])
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

if args.testing:
    check_point = torch.load(open('model.bin', 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path", flush=True)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            for ci, conv in enumerate(current_batch.examples):
                model.clear_memory()
                for ui, ex in enumerate(conv):
                    pred, label, loss = model.decode(Example.label_vocab, current_batch, ci, ui)
                    predictions.append(pred)
                    labels.append(label)
                    total_loss += loss
                    count += 1
                    # if any([l.split('-')[-1] not in current_batch.utt[ci][ui] for l in pred]):
                    #     print(current_batch.utt[ci][ui], pred, label)
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps), flush=True)
    optimizer = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......', flush=True)
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        epoch_loss_count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            batch_loss = 0
            batch_loss_count = 0
            for ci, conv in enumerate(current_batch.examples):
                model.clear_memory()
                for ui, ex in enumerate(conv):
                    output, loss = model(current_batch, ci, ui)
                    batch_loss += loss
                    epoch_loss += loss.item()
                    batch_loss_count += 1
                    epoch_loss_count += 1
            batch_loss /= batch_loss_count
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / epoch_loss_count), flush=True)
        torch.cuda.empty_cache()
        gc.collect()

        # train_metrics, train_loss = decode('train')
        # train_acc, train_fscore = train_metrics['acc'], train_metrics['fscore']
        dev_metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = dev_metrics['acc'], dev_metrics['fscore']
        # print("Train loss: %.4f\tTrain acc: %.2f\tTrain fscore(p/r/f): (%.2f/%.2f/%.2f)" % (train_loss, train_acc, train_fscore['precision'],train_fscore['recall'], train_fscore['fscore']), flush=True)
        print("Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']), flush=True)
        
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('best_models/model15.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']), flush=True)

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']), flush=True)
else:
    train_metrics, train_loss = decode('train')
    train_acc, train_fscore = train_metrics['acc'], train_metrics['fscore']
    dev_metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = dev_metrics['acc'], dev_metrics['fscore']
    predict()
    print("Train loss: %.4f\tTrain acc: %.2f\tTrain fscore(p/r/f): (%.2f/%.2f/%.2f)" % (train_loss, train_acc, train_fscore['precision'],train_fscore['recall'], train_fscore['fscore']), flush=True)
    print("Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']), flush=True)


# start_time = time.time()
# metrics, dev_loss = decode('dev')
# train_acc, train_fscore = metrics['acc'], metrics['fscore']
# print("Evaluation costs %.2fs ; Train loss: %.4f\tTrain acc: %.2f\tTrain fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, train_acc, train_fscore['precision'], train_fscore['recall'], train_fscore['fscore']), flush=True)