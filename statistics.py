import glob
import os
import numpy as np
import re
import argparse
from numpy import array, float32
from genotypes import Genotype
import sys

fw = open('result.txt', 'w')


def get_step_1_result(path):
    all_filse = sorted(glob.glob(path))
    for f in all_filse:
        print(f)
        fo = open(f)
        flag = 0
        tmp, result, best = [], [], None
        for line in fo.readlines():
            if flag == 1:
                tmp.append(line.strip())
                if "reduce_concat=range(2, 5))}" in line:
                    result.append(eval(''.join(tmp)))
                    tmp = []
            elif flag == 2:
                tmp.append(line.strip())
                if "reduce_concat=range(2, 5))}" in line:
                    best = eval(''.join(tmp))
                    tmp = []
                    break
            if flag == 0 and "==========TRAINING_RESULT============" in line:
                flag = 1
            elif flag == 1 and "====BEST_RESULT====" in line:
                flag = 2
                tmp = []
        if len(result) == 0:
            continue

        alphas = [np.concatenate(x['alpha'], axis=0) for x in result]
        eval_results = [x['eval_result'] for x in result]
        eval_results[0] = -1
        best_ep = eval_results.index(max(eval_results))
        best = result[best_ep]
        alpha_diff, alpha_top2, alpha_none = [], [], []
        for ep_index, alpha in enumerate(alphas):
            alpha, _none = alpha[:, :-1], alpha[:, -1]
            alpha = np.sort(alpha)
            alpha_diff.append(np.average(alpha[:, -1] - alpha[:, 0]))
            alpha_none.append(np.average(_none))
            alpha_top2.append(np.average(alpha[:, -1] - alpha[:, -2]))
        alpha_diff = [round(x, 3) for x in ([alpha_diff[1]] + alpha_diff[::5][1:] + [alpha_diff[-1]])]
        alpha_top2 = [round(x, 3) for x in ([alpha_top2[1]] + alpha_top2[::5][1:] + [alpha_top2[-1]])]
        alpha_none = [round(x, 3) for x in ([alpha_none[1]] + alpha_none[::5][1:] + [alpha_none[-1]])]
        best_ep = best['epoch']
        best_dev = round(best['eval_result'], 4)
        best_gt = best['genotypes']
        best_ep_alpha = np.sort(np.concatenate(best['alpha'], axis=0)[:, :-1])
        best_ep_alpha_diff_max_min = np.average(best_ep_alpha[:, -1] - best_ep_alpha[:, 0])
        best_ep_alpha_diff_top2 = np.average(best_ep_alpha[:, -1] - best_ep_alpha[:, -2])
        print(
            f,
            '\t'.join(f.strip('.log').split('_')),
            best_ep,
            best_dev,
            # best_ep_alpha,
            best_ep_alpha_diff_max_min,
            best_ep_alpha_diff_top2,
            alpha_diff,
            alpha_top2,
            alpha_none,
            best_gt,
            sep='\t',
            file=fw)


def get_step_2_result(path):
    all_filse = sorted(glob.glob(path))
    for file in all_filse:
        sign = 0#not file.startswith("step2_WikiQA")
        with open(file) as f:
            final = 0
            for line in f.readlines():
                if sign == 0:
                    if "Final best Prec@1" in line:
                        final = line.strip().split('=')[-1].replace('%', "")
                else:
                    if line.startswith('dev Loss'):
                        pattern = "\{.*\}"
                        final.append(eval(re.findall(pattern, line.strip())[0]))
            if sign == 0:
                print(
                file.split('/')[-1],
                round(float(final), 5),
                sep='\t',
                file=fw,
                )
            else:
                try:
                    print(
                    file.split('/')[-1],
                    round(float([final['map']]), 5),
                    round(float(final['mrr']), 5),
                    sep='\t',
                    file=fw,
                    )
                except:
                    continue


def get_step_1_EMD_result(path):
    all_files = sorted(glob.glob(path))
    for file in all_files:
        try:
            with open(file) as f:
                genos, emd = [], []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if "EMD loss" in line:
                        line = line.strip().split("EMD loss: ")[-1].strip()
                        emd_loss = float(line)
                        emd.append(emd_loss)
                        f.readline()
                        while True:
                            line = f.readline()
                            if 'Genotype' in line:
                                break
                            if not line:
                                break
                        genos.append(line.strip())
                min_emd_epoch = np.argmin(emd)
                print(
                    file,
                    min_emd_epoch,
                    emd[min_emd_epoch],
                    genos[min_emd_epoch],
                )
        except:
            continue


def draw_step_2_results(path):
    import numpy as np
    import matplotlib.pyplot as plt
    all_accs = {}
    all_files = sorted(glob.glob('./snas_one_step_*.log'))
    for file in all_files:
        print(file)
        with open(file) as f:
            accus = []
            for line in f.readlines():
                if "Final Prec@1" in line:
                    line = line.strip().split("Final Prec@1")[1].split("Loss")[0]
                    line = eval(line)
                    accus.append(line['acc'])
            all_accs[file] = accus
    for k, i in all_accs.items():
        plt.plot(range(len(i)), i, label=k, alpha=0.5)
    plt.legend()
    plt.savefig('darts_result3.png', dpi=400)


def get_step1_geno_from_file(path):
    if not os.path.exists(path):
        files = glob.glob(path)
    else:
        files = [path]
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                if 'Genotype' in line and 'Best' not in line:
                    print(line.strip())


def isnumber(s):
    try:
        float(s)
        return True
    except:
        return False

def step2_error(path):
    all_filse = sorted(glob.glob(path))
    import re
    pattern = "\{.*\}"
    for file in all_filse:
        print(file)
        rs = [[],[]]
        with open(file) as f:
            final = 0
            for line in f.readlines():
                if line.startswith("EVAL RESULT"):
                    final = eval(re.findall(pattern, line.strip())[0])
                    r1 = final['map']
                    r2 = final['mrr']
                    rs[0].append(r1)
                    rs[1].append(r2)
            print(
                file.split('/')[-1],
                round(float(max(rs[0])), 5),
                round(float(max(rs[1])), 5),
                sep='\t',
                file=fw,
            )

def get_training_time(path):
    all_filse = sorted(glob.glob(path))
    for file in all_filse:
        print(file)
        time = []
        f = open(file)
        for line in f.readlines():
            line = line.strip()
            if line.startswith("Current epoch") and 'time' in line:
                time.append(line.split('=')[-1].strip())
        time.insert(0, file)
        print('\t'.join(time), file=fw)

if __name__ == "__main__":
    step2_error("./WikiQA_transformer*.log")
    exit(0)
    # get_training_time('*TIME*')
    # exit()
    # get_step_1_result("SST*.log")
    # get_step_2_result("step2_sst_hidn2attn_NoEMD_1*")
    # func_dict = {1: get_step_1_result, 2: get_step_2_result, 3: get_step_1_EMD_result, 4: draw_step_2_results, 5: get_step1_geno_from_file}
    # func_dict[args.type](args.path)
    if sys.argv[-1] == "2":
        get_step_2_result("/data/lxk/NLP/NASLogs/24/2021-01-07/*.log")
    else:
        get_step_1_result(sys.argv[-1])