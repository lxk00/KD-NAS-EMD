""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np
import random
# import metrics_eval
from scipy.stats import pearsonr, spearmanr
from bert_fineturn.data_processor.glue import glue_compute_metrics as compute_metrics
from sklearn.metrics import matthews_corrcoef, f1_score

def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size())
        for k, v in model.named_parameters()
        if v.requires_grad and not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), isTrain=False, output_modes="classification"):
    """ Computes the precision@k for the specified values of k """
    if output_modes == "classification":
        maxk = max(topk)
        batch_size = target.size(0)
        _, out_classes = output.max(dim=1)
        correct = (out_classes == target).sum()
        correct = correct.float() / batch_size
        return correct
    else:
        correct1 = pearsonr(
            output.reshape(-1).detach().cpu().numpy(),
            target.detach().cpu().numpy())[0]
        correct2 = spearmanr(
            output.reshape(-1).detach().cpu().numpy(),
            target.detach().cpu().numpy())[0]
        return (correct1 + correct2) / 2
    return res


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def json_data(data):
    uids = data['uids']
    predictions = data['scores']
    previous = ""
    rank_list = []
    data_list = []
    for i, uid in enumerate(uids):
        qid = uid.split('-')[0]
        aid = "-".join(uid.split('-')[1:])
        if qid != previous:
            ordered_rank_list = sorted(rank_list, key=lambda x: x[1], reverse=True)
            rank_dict = {}
            for j, value in enumerate(ordered_rank_list):
                rank_dict[value[0]] = j + 1
            for j, value in enumerate(rank_list):
                data_list.append([previous, value[0], rank_dict[value[0]]])
            previous = qid
            rank_list = []
        rank_list.append((aid, predictions[2*i+1]))
    ordered_rank_list = sorted(rank_list, key=lambda x: x[1], reverse=True)
    rank_dict = {}
    for j, value in enumerate(ordered_rank_list):
        rank_dict[value[0]] = j+1
    for j, value in enumerate(rank_list):
        data_list.append([previous, value[0], rank_dict[value[0]]])
    return data_list

def eval_map_mrr(data_list, gold_file):
    dic = {}
    fin = open(gold_file)
    for line in fin:
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')
        if cols[0] == 'QuestionID':
            continue

        q_id = cols[0]
        a_id = cols[4]

        if not q_id in dic:
            dic[q_id] = {}
        dic[q_id][a_id] = [cols[6], -1]
    fin.close()

    for cols in data_list:
        q_id = cols[0]
        a_id = cols[1]
        rank = int(cols[2])
        dic[q_id][a_id][1] = rank

    MAP = 0.0
    MRR = 0.0
    cnt = 0
    for q_id in dic:
        flag = False
        for k,v in dic[q_id].items():
            if v[0] == '1':
                flag = True
        if flag:
            cnt += 1
        else:
            continue

        sort_rank = sorted(dic[q_id].items(), key=lambda asd: asd[1][1], reverse=False)
        correct = 0
        total = 0
        AP = 0.0
        mrr_mark = False
        for i in range(len(sort_rank)):
            # compute MRR
            if sort_rank[i][1][0] == '1' and mrr_mark == False:
                MRR += 1.0 / float(i + 1)
                mrr_mark = True
            # compute MAP
            total += 1
            if sort_rank[i][1][0] == '1':
                correct += 1
                AP += float(correct) / float(total)
        if correct != 0:
            AP /= float(correct)
        else:
            AP = 0.0
        MAP += AP
    MAP /= float(cnt)
    MRR /= float(cnt)
    return MAP, MRR

def eval_qa(data, gold_file):
    data_list = json_data(data=data)
    MAP, MRR = eval_map_mrr(data_list, gold_file)
    return MAP, MRR


def compute_metrics(task_name, preds, labels):
    if 'wikiqa' not in task_name.lower():
        assert len(preds) == len(labels)    
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "trec":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wikiqa":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name in ["wikiqadev", "wikiqatest"]:
        data_list = json_data(preds)
        x = eval_map_mrr(data_list, labels)
        return {"map": x[0], "mrr":x[1]}
    elif task_name == "sick":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sickreg":
        return pearson_and_spearman(preds, labels)
    elif task_name == "agnews":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "dbpedia":
        return {"acc": simple_accuracy(preds, labels)}
    else: 
        raise KeyError(task_name)


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)




def load_embedding_weight(model, path, train=False, device='cpu'):
    pretrain_dict = torch.load(path, map_location=device)
    new_dict = {}
    for key in pretrain_dict.keys():
        if 'embeddings' in key:
            new_k = key
            if 'LayerNorm' in key:
                new_k = new_k.replace('gamma', 'weight')
                new_k = new_k.replace('beta', 'bias')
            if train:
                new_dict[new_k.replace('bert.embeddings', 'net.stem')] = pretrain_dict[key]
            else:
                new_dict[new_k.replace('bert.embeddings', 'stem')] = pretrain_dict[key]
    print("="*10 + " RESTORE EMBEDDING KEYS" + "="*10)
    for k, v in model.named_parameters():
        if k in new_dict:
            print(k)
    model.load_state_dict(new_dict, strict=False)

def load_embedding_weight2(model, path, train=False, device='cpu'):
    pretrain_dict = torch.load(path, map_location=device)
    new_dict = {}
    for key in pretrain_dict.keys():
        if 'embeddings' in key:
            new_key = key
            if 'LayerNorm' in new_key:
                new_key = new_key.replace('gamma', 'weight')
                new_key = new_key.replace('beta', 'bias')
            
            new_dict[new_key] = pretrain_dict[key]
    print("="*10 + " RESTORE EMBEDDING KEYS" + "="*10)
    for k, v in model.named_parameters():
        if k in new_dict:
            print(k)
    model.load_state_dict(new_dict, strict=False)


class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp



def get_acc_from_pred(output_mode, task_name, preds, eval_labels):
    if task_name.lower() not in  ['wikiqadev', 'wikiqatest'] :
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.squeeze(preds)

    result = compute_metrics(task_name.lower(), preds, eval_labels)
    
    acc = 0

    if 'mcc' in result:
        acc = result['mcc']
    elif 'f1' in result:
        acc = result['f1']
    elif 'corr' in result:
        acc = result['corr']
    elif 'map' in result:
        acc = (result['map'] + result['mrr']) / 2 
    else:
        acc = result['acc']

    return result, acc

if __name__ == "__main__":
    top1 = AverageMeter()
    top1.update(0.5, 10)
    print(top1.avg)