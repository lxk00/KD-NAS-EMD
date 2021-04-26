import os
import sys
import time
from datetime import datetime

def train_darts():
    cmds = []
    seed = [612,41326,215]
    training_script = "nohup python model_search.py --batch_size 64 --max_seq_length 64 --use_emd False --alpha_ac_rate 1.0  --w_lr 1e-4 --w_lr_min 1e-4 --use_kd False --train_mode XX --epochs 80 --add_op cnn --datasets WikiQA --alpha_lr 1e-3 --seed {} > DARTS_base_WikiQA_WLR.1e-4_ALR.1e-4_seed.{}.log 2>&1 &"
    # for at in attention_types:
    for s in seed:
        cmds.append(training_script.format(s, s))
    return cmds

def train_time_DARTS():
    cmds = []
    ds = ['SST-2']
    training_script = "nohup python model_search.py --datasets {} --batch_size 32 --use_emd False --alpha_ac_rate 1.0 --w_lr 0.01 --w_lr_min 1e-4 --use_kd False --train_mode XX --epochs 80 --add_op cnn --alpha_lr 0.01 --seed 2 > DARTS_TRAINING_TIME_{}.log 2>&1 &"
    for d in ds:
        cmds.append(training_script.format(d, d))
    return cmds

def train_time_EMD():
    cmds = []
    ds = ['MRPC', 'RTE', 'WikiQA', 'SST-2', 'TREC']
    training_script = "nohup python model_search.py --datasets {} --use_emd {} --batch_size 32 --kd_alpha 0.0 --w_lr 0.01 --w_lr_min 1e-4 --epochs 6 --hidn2attn {} --alpha_optim sgd --alpha_ep 2 --alpha_ac_rate {} --add_op cnn --emd_rate 1 --alpha_lr 0.001 --weight_rate 1.0 --eval_during_train False --update_emd True --cell_norm {} --seed 1234 > EMD_TRAINING_TIME_{}_{}.log 2>&1 &"

    prepare_dict = [
        ["", 'True', 'True',  "0", 'addnorm', "" ,'拆解实验BASE',]]
    for p in prepare_dict:
        for d in ds:
            p[0] = p[5] = d
            cmds.append(training_script.format(*p))
    return cmds


def train_step_2_ablation():
    cmds = []
    seeds = [31278, 25192]
    fs = range(3)
    sx = "nohup python augment.py --datasets TREC --use_kd True --use_emd {} --batch_size 64 --kd_alpha 0.0 --lr 0.025 --epochs 15 --hidn2attn {}  --emd_rate {} --weight_rate {} --eval_during_train True --update_emd True --cell_norm {} --train_eval_time 4 --seed {} --filegt {} --name TREC_{}_seed.{}_gt.{} > step2_TREC_{}_0.1_seed.{}_gt.{}.log 2>&1 &"

    ps = [['True',  'False', "1", 0.1, 'addnorm', '51252', 0 ,'拆解实验ATTN', '31278', 0  ,'拆解实验ATTN', '51252', 0 ],
          ['True',  'False', "1", 0.1, 'addnorm', '51252', 1 ,'拆解实验ATTN', '31278', 1  ,'拆解实验ATTN', '51252', 1 ],
          ['True',  'False', "1", 0.1, 'addnorm', '51252', 2 ,'拆解实验ATTN', '31278', 2  ,'拆解实验ATTN', '51252', 2 ],
    # ps = [['True',  'True',  "1", 10, 'addnorm', '31278', 3 ,'拆解实验BASE', '31278', 0  ,'拆解实验BASE', '31278', 0 ],
    #       ['True',  'True',  "1", 10, 'addnorm', '31278', 4 ,'拆解实验BASE', '31278', 1  ,'拆解实验BASE', '31278', 1 ],
    #       ['True',  'True',  "1", 10, 'addnorm', '31278', 5 ,'拆解实验BASE', '31278', 2  ,'拆解实验BASE', '31278', 2 ],
        #   ['False', 'True',  "0", 10, 'addnorm', '31278', 6 ,'拆解实验EMD',  '31278', 3  ,'拆解实验EMD',  '31278', 3 ],
        #   ['False', 'True',  "0", 10, 'addnorm', '31278', 7 ,'拆解实验EMD',  '31278', 4  ,'拆解实验EMD',  '31278', 4 ],
        #   ['False', 'True',  "0", 10, 'addnorm', '31278', 8 ,'拆解实验EMD',  '31278', 5  ,'拆解实验EMD',  '31278', 5 ],
        #   ['True', 'True',   "1", 10, 'X',       '31278', 9 ,'拆解实验LM',   '31278', 9  ,'拆解实验LM',   '31278', 9 ],
        #   ['True', 'True',   "1", 10, 'X',       '31278', 10,'拆解实验LM',   '31278', 10 ,'拆解实验LM',   '31278', 10],
        #   ['True', 'True',   "1", 10, 'X',       '31278', 11,'拆解实验LM',   '31278', 11 ,'拆解实验LM',   '31278', 11],
          ['True',  'False', "1", 0.1, 'addnorm', '86578', 0 ,'拆解实验ATTN', '25192', 0  ,'拆解实验ATTN', '86578', 0 ],
          ['True',  'False', "1", 0.1, 'addnorm', '86578', 1 ,'拆解实验ATTN', '25192', 1  ,'拆解实验ATTN', '86578', 1 ],
          ['True',  'False', "1", 0.1, 'addnorm', '86578', 2 ,'拆解实验ATTN', '25192', 2  ,'拆解实验ATTN', '86578', 2 ]]
        #   ['True',  'True',  "1", 10, 'addnorm', '25192', 3 ,'拆解实验BASE', '25192', 0  ,'拆解实验BASE', '25192', 0 ],
        #   ['True',  'True',  "1", 10, 'addnorm', '25192', 4 ,'拆解实验BASE', '25192', 1  ,'拆解实验BASE', '25192', 1 ],
        #   ['True',  'True',  "1", 10, 'addnorm', '25192', 5 ,'拆解实验BASE', '25192', 2  ,'拆解实验BASE', '25192', 2 ],]
        #   ['False', 'True',  "0", 10, 'addnorm', '25192', 6 ,'拆解实验EMD',  '25192', 3  ,'拆解实验EMD',  '25192', 3 ],
        #   ['False', 'True',  "0", 10, 'addnorm', '25192', 7 ,'拆解实验EMD',  '25192', 4  ,'拆解实验EMD',  '25192', 4 ],
        #   ['False', 'True',  "0", 10, 'addnorm', '25192', 8 ,'拆解实验EMD',  '25192', 5  ,'拆解实验EMD',  '25192', 5 ],
        #   ['True', 'True',   "1", 10, 'X',       '25192', 9 ,'拆解实验LM',   '25192', 9  ,'拆解实验LM',   '25192', 9 ],
        #   ['True', 'True',   "1", 10, 'X',       '25192', 10,'拆解实验LM',   '25192', 10 ,'拆解实验LM',   '25192', 10],
        #   ['True', 'True',   "1", 10, 'X',       '25192', 11,'拆解实验LM',   '25192', 11 ,'拆解实验LM',   '25192', 11],]
    for p in ps:
        cmds.append(sx.format(*p))
    return cmds


def train_step_2():
    cmds = []
    seeds = [278, 317]
    # fs = 0  #range(3)
    ts = "nohup python augment.py --weight_rate 0.001 --use_emd True --use_kd True --eval_during_train True --lr 0.025 --datasets SST-2 --epochs 15 --cell_norm addnorm --kd_alpha 0.0 --filegt {} --eval_during_train True --train_eval_time 4 --hidn2attn True --name SST-2_CHANGESEED_gt.{}_seed.{} --seed {} --update_emd True --batch_size 64 > step2_SST-2_CHANGESEED_gt.{}_seed.{}.log 2>&1 &"
    for s in seeds:
        for fs in range(6):
            cmds.append(ts.format(fs, fs, s , s, fs, s))
    return cmds

# def train_darts():
#     cmds = []
#     weight_lrs = [1e-3, 1e-2, 1e-4]
#     alpha_lr = [1e-2, 1e-3, 1e-4]
#     training_script = "nohup python model_search.py --batch_size 64 --max_seq_length 64 --use_emd False --alpha_ac_rate 1.0 --datasets TREC --w_lr {} --w_lr_min 1e-4 --use_kd False --train_mode XX --epochs 60 --add_op cnn --alpha_lr {} --seed {} > DARTS_base_TREC_wlr.{}._alr.{}_seed.{}.log 2>&1 &"
#     # for at in attention_types:
#     for a in alpha_lr:
#         for w in weight_lrs:
#             for s in range(3):
#                 cmds.append(training_script.format(w, a, s, w, a, s))
#     return cmds
def train_step_1():
    cmds = []
    emd_rate = [200, 5000, 1000]
    weight_lrs = [0.1, 0.05, 0.02, 0.01]
    alpha_lr = [1e-4, 5e-4, 1e-3, 5e-3]
    datasets = ['TREC']
    training_script = "nohup python model_search.py --datasets {} --use_emd False --alpha_ac_rate 1.0 --batch_size 64 --kd_alpha 0.0 --w_lr {} --w_lr_min 1e-4 --epochs 20 --hidn2attn False --alpha_optim sgd --alpha_ep 2 --max_seq_length 64  --add_op cnn --emd_rate 1 --alpha_lr {} --weight_rate {} --eval_during_train False --update_emd True --cell_norm addnorm --train_eval_time 4 --seed {} > {}调参_WLR.{}_ALR.{}_ER.{}_seed.{}.log 2>&1 &"
    for w in weight_lrs:
        for a in alpha_lr:
            for e in emd_rate:
                for d in datasets:
                    for s in range(3):
                        cmds.append(training_script.format(d, w, a, e, s, d, w, a, e, s))
    return cmds

# def train_step_2():
#     cmds = []
#     f = open("train_genotyps.txt")
#     for index, line in enumerate(f.readlines()):
#         line = line.strip()
#         geno = line
#         for s in [31278, 25192]:
#             cmds.append(
#                 "nohup python augment.py --datasets WikiQA --lr 0.025 --epochs 20 --batch_size 64 --cell_norm x --use_emd False --use_kd False --genotype \"{}\" --seed {} --name step2_WikiQA_DARTS_ids.{}_seed.{} > step2_WikiQA_DARTS_ids.{}_seed.{}.log 2>&1 &"
#                 .format(geno, s, index, s, index, s))
#     return cmds



def train_step_1_Ablation():
    '''
    export TASK_NAME=
    export WEIGHT_LR=
    export ALPHA_LR=
    export EPOCH=
    export TRAIN_EVAL=
    '''
    cmds = []
    seeds = [31278, 25192]
    fs = range(3)
    training_script = "nohup python model_search.py --datasets TREC --use_emd {} --batch_size 64 --kd_alpha 0.0 --w_lr 0.01 --w_lr_min 1e-4 --epochs 15 --hidn2attn {} --alpha_optim sgd --alpha_ep 2 --alpha_ac_rate {} --add_op {} --emd_rate 1 --alpha_lr 0.001 --weight_rate {} --eval_during_train True --update_emd True --cell_norm {} --train_eval_time 4 --seed {} > TREC_SGD_EMD_SEP_{}_WLR.0.01_ALR.0.001_seed.{}.log 2>&1 &"

    prepare_dict = [
        ['True',  'True',  "0", 'cnn', 50000, 'addnorm', '1', '拆解实验BASE', '1'],
        ['True',  'True',  "0", 'cnn', 50000, 'addnorm', '2', '拆解实验BASE', '2'],
        ['True',  'True',  "0", 'cnn', 50000, 'addnorm', '3', '拆解实验BASE', '3'],
        ['False', 'True',  "1", 'cnn', 50000, 'addnorm', '1', '拆解实验EMD',  '1'],
        ['False', 'True',  "1", 'cnn', 50000, 'addnorm', '2', '拆解实验EMD',  '2'],
        ['False', 'True',  "1", 'cnn', 50000, 'addnorm', '3', '拆解实验EMD',  '3'],
        ['True',  'False', "0", 'cnn', 150, 'addnorm', '1', '拆解实验ATTN', '1'],
        ['True',  'False', "0", 'cnn', 150, 'addnorm', '2', '拆解实验ATTN', '2'],
        ['True',  'False', "0", 'cnn', 150, 'addnorm', '3', '拆解实验ATTN', '3'],
        ['True', 'True',  "0", 'cnn', 50000, 'X', '1', '拆解实验LM',  '1'],
        ['True', 'True',  "0", 'cnn', 50000, 'X', '2', '拆解实验LM',  '2'],
        ['True', 'True',  "0", 'cnn', 50000, 'X', '3', '拆解实验LM',  '3']]
    for p in prepare_dict:
        cmds.append(training_script.format(*p))
    return cmds

def transfer_script():
    sx = "nohup python augment.py --datasets {} --use_kd False --use_emd False--batch_size 64 --kd_alpha 0.0 --lr 0.025 --epochs 15 --hidn2attn True --emd_rate 1 --weight_rate {} --eval_during_train True --update_emd True --cell_norm x --train_eval_time 4 --seed 0 --filegt {} --name DARTS_{}_{}_gt.{}_seed.0 > step2_DARTS_{}_TRANSFER_{}_seed.0_gt.{}.log 2>&1 &"
    dataset = ["MRPC", "SST-2", "WikiQA", "RTE", "TREC"]
    hp = {"MRPC": 100000, "SST-2": 0.001, "WikiQA": 10, "RTE": 10, "TREC": 10}
    cmds = []
    for index, d in enumerate(dataset):
        choose = d
        unchoose = set(dataset) - set([choose])
        gts = [index * 3, index * 3 + 1, index * 3 + 2]
        for cd in unchoose:
            for gt in gts: 
                cmds.append(sx.format(cd, hp[d], gt, cd, d, gt, cd, d, gt))
    return cmds

def vdcnn():
    dataset = ["MRPC", "SST-2", "WikiQA", "RTE", "TREC"]
    xx = "nohup python teacher_utils/train_vdcnn.py --datasets {} --seed {} > {}_vdcnn_seed{}.log 2>&1 &"
    cmds = []
    for d in dataset:
        for s in range(3):
            cmds.append(xx.format(d, s, d, s))
    
    return cmds

def trans():
    dataset = ["MRPC", "RTE",]
    xx = "nohup python teacher_utils/finetune.py --datasets {} --seed {} > {}_transformer_seed{}.log 2>&1 &"
    cmds = []
    for d in dataset:
        for s in range(3):
            cmds.append(xx.format(d, s, d, s))
    dataset2 = ["SST-2", "WikiQA", "TREC"]
    xx = "nohup python teacher_utils/finetune.py --datasets {} --learning_rate {} --num_train_epochs {} --seed {} > {}_transformer_lr{}_ep{}_seed{}.log 2>&1 &"
    for d in dataset2:
        for s in range(3):
            for lr in [1e-3, 1e-4, 1e-5, 5e-5, 5e-6]:
                for ep in [15.0]:
                    cmds.append(xx.format(d, lr, ep, s, d, lr, ep, s))
    return cmds

cmd = vdcnn() + trans()
print('\n'.join(cmd))
exit(0)

# print('\n'.join(cmd))


def gpu_info():
    gpu_status = [
        x.strip('|') for x in os.popen('nvidia-smi | grep python').read().strip().split('\n')
    ]
    gpu_status = [x.split() for x in gpu_status if x != '']
    if len(gpu_status) == 0:
        return {}
    device2pid = {int(x[0]): x[1] for x in gpu_status if x[1] not in ['45957', '45958', '45959']}
    return device2pid


def process_info():
    running_process = os.popen('ps -aux | grep python').read().strip().split('\n')
    running_process = [x for x in running_process if ("finetune" in x or "train_vdcnn" in x)]
    running_process = [x.split() for x in running_process]
    return running_process

ALL_GPUS = 8


def narrow_setup(interval=10):
    all_gpus = set(range(ALL_GPUS))
    pid_occupy_history = dict()
    while len(cmd) > 0:
        time.sleep(interval)
        r_pids = process_info()
        if len(r_pids) < ALL_GPUS:
            occupy = gpu_info()
            if len(occupy) < ALL_GPUS:
                using_gpus = occupy.keys()
                waiting_to_load = set([x[1] for x in r_pids]) - set(occupy.values())
                tmp = []
                for x in waiting_to_load:
                    if x in pid_occupy_history:
                        tmp.append(pid_occupy_history[x])
                using_gpus = list(using_gpus) + tmp
                empty_gpu = all_gpus - set(using_gpus)
                empty_gpu = list(empty_gpu)[0]
                running_script = "CUDA_VISIBLE_DEVICES={} ".format(empty_gpu) + cmd.pop()
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")

                print("Running Time =", current_time, running_script)
                sys.stdout.flush()
                os.system(running_script)
                new_pid = set([x[1] for x in process_info()]) - set([x[1] for x in r_pids])
                pid_occupy_history[new_pid.pop()] = empty_gpu


if __name__ == '__main__':
    narrow_setup()