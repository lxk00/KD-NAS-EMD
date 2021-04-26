""" Search cell """
import os
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from utils import get_acc_from_pred, param_size, load_embedding_weight, AverageMeter
from data_utils import bert_batch_split, load_glue_dataset
from models.search_cnn import SearchCNNController
from architect import Architect
from config import SearchConfig
import time

import genotypes as gt
from kdTool import Emd_Evaluator, softmax
from dist_util_torch import init_gpu_params, set_seed, FileLogger
from tool_blocks import convert_to_attn

acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte", 'books', 'dbpedia', 'agnews']
corr_tasks = ["sts-b"]
mcc_tasks = ["cola"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SearchModel():

    def __init__(self):
        self.config = SearchConfig()
        self.writer = None
        if self.config.tb_dir != "":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.tb_dir, flush_secs=20)
        init_gpu_params(self.config)
        set_seed(self.config)
        self.logger = FileLogger('./log', self.config.is_master, self.config.is_master)
        self.load_data()
        self.logger.info(self.config)
        self.model = SearchCNNController(self.config, self.n_classes, self.output_mode)
        self.load_model()
        self.init_kd_component()
        if self.config.n_gpu > 0:
            self.model.to(device)
        if self.config.n_gpu > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.config.local_rank], find_unused_parameters=True)
        self.model_to_print = self.model if self.config.multi_gpu is False else self.model.module
        self.architect = Architect(self.model, self.teacher_model, self.config, self.emd_tool)
        mb_params = param_size(self.model)
        self.logger.info("Model size = {:.3f} MB".format(mb_params))
        self.eval_result_map = []
        self.init_optim()

    def init_kd_component(self):
        from transformers import BertForSequenceClassification
        self.teacher_model, self.emd_tool = None, None
        if self.config.use_kd:
            self.teacher_model = BertForSequenceClassification.from_pretrained(
                self.config.teacher_model, return_dict=False)
            self.teacher_model = self.teacher_model.to(device)
            self.teacher_model.eval()
            if self.config.use_emd:
                self.emd_tool = Emd_Evaluator(
                    self.config.layers,
                    12,
                    device,
                    weight_rate=self.config.weight_rate,
                    add_softmax=self.config.add_softmax)

    def load_data(self):
        # set seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = True
        self.task_name = self.config.datasets
        self.train_loader, self.arch_loader, self.eval_loader, _, self.output_mode, self.n_classes, self.config, self.eval_sids = load_glue_dataset(
            self.config)
        self.logger.info(f"train_loader length {len(self.train_loader)}")

    def init_optim(self):
        no_decay = ["bias", "LayerNorm.weight"]
        self.w_optim = torch.optim.SGD([
            p for n, p in self.model.named_parameters()
            if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' not in n
        ],
                                       self.config.w_lr,
                                       momentum=self.config.w_momentum,
                                       weight_decay=self.config.w_weight_decay)
        if self.config.alpha_optim.lower() == 'adam':
            self.alpha_optim = torch.optim.Adam([
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' in n
            ],
                                                self.config.alpha_lr,
                                                weight_decay=self.config.alpha_weight_decay)
        elif self.config.alpha_optim.lower() == 'sgd':
            self.alpha_optim = torch.optim.SGD([
                p for n, p in self.model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad and 'alpha' in n
            ],
                                               self.config.alpha_lr,
                                               weight_decay=self.config.alpha_weight_decay)
        else:
            raise NotImplementedError("no such optimizer")

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optim, self.config.epochs, eta_min=self.config.w_lr_min)

    def load_model(self):
        if self.config.restore != "":
            old_params_dict = dict()
            for k, v in self.model.named_parameters():
                old_params_dict[k] = v
            self.model.load_state_dict(torch.load(self.config.restore), strict=False)
            for k, v in self.model.named_parameters():
                if torch.sum(v) != torch.sum(old_params_dict[k]):
                    print(k + " not restore")
            del old_params_dict
        else:
            load_embedding_weight(self.model, 'teacher_utils/bert_base_uncased/pytorch_model.bin', True, device)

    def save_checkpoint(self, dump_path, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.model.module if hasattr(self.model, "module") else self.model
        state_dict = mdl_to_save.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if 'alpha' in k}
        torch.save(state_dict, os.path.join(dump_path, checkpoint_name))

    def main(self):
        # training loop
        best_top1 = 0.
        is_best = False
        for epoch in range(self.config.epochs):
            lr = self.lr_scheduler.get_last_lr()[-1]

            self.logger.info("Epoch {}".format(epoch))
            self.logger.info("Learning Rate {}".format(lr))
            start_time = time.time()

            # for k, v in self.model.named_parameters():
            #     print(k, torch.sum(v).item())

            if self.config.train_mode == 'sep':
                self.train_sep(lr, epoch)
            else:    
                self.train(lr, epoch)
            self.logger.info("Current epoch training time = {}".format(time.time() - start_time))

            self.model_to_print.print_alphas(self.logger)
            self.lr_scheduler.step()

            self.logger.info('valid')
            cur_step = (epoch + 1) * len(self.train_loader)
            
            dev_start_time = time.time()
            top1 = self.validate(epoch, cur_step, "val")
            self.logger.info("Current epoch vaildation time = {}".format(time.time() - dev_start_time))
            self.logger.info("Current epoch Total time = {}".format(time.time() - start_time))

            # genotype
            genotypes = self.model_to_print.genotype()
            # if config.is_master:
            self.logger.info("========genotype========\n" + str(genotypes))
            # # save
            is_best = best_top1 <= top1
            if is_best:
                best_top1 = top1
                best_genotype = genotypes
            self.logger.info("Present best Prec@1 = {:.4%}".format(best_top1))
        if self.config.tb_dir != "":
            self.writer.close()
        # self.logger.info("Final best Prec@1 = " + str(best_top1))
        # self.logger.info("Best Genotype = " + str(best_genotype))
        self.logger.info("==========TRAINING_RESULT============")
        for x in self.eval_result_map:
            self.logger.info(x)
        evals = [x['eval_result'] for x in self.eval_result_map]
        evals[0] = -1
        best_ep = evals.index(max(evals))
        self.logger.info("==========BEST_RESULT============")
        self.logger.info(self.eval_result_map[best_ep])

    def train(self, lr, epoch):
        top1 = AverageMeter()
        losses = AverageMeter()

        self.model.train()
        total_num_step = len(self.train_loader)
        cur_step = epoch * len(self.train_loader)
        valid_iter = iter(self.arch_loader)

        point = [int(total_num_step * i / self.config.train_eval_time) for i in range(1, self.config.train_eval_time + 1)][:-1]

        for step, data in enumerate(self.train_loader):
            trn_X, trn_y = bert_batch_split(data, self.config.local_rank, device)
            try:
                v_data = next(valid_iter)
            except StopIteration:
                valid_iter = iter(self.arch_loader)
                v_data = next(valid_iter)
            val_X, val_y = bert_batch_split(v_data, self.config.local_rank, device)

            trn_t, val_t = None, None
            if self.config.use_kd:
                with torch.no_grad():
                    teacher_logits, teacher_reps = self.teacher_model(
                        input_ids=trn_X[0], attention_mask=trn_X[1], token_type_ids=trn_X[2])
                    trn_t = (teacher_logits, teacher_reps)

                    if self.config.one_step is not True:
                        v_teacher_logits, v_teacher_reps = self.teacher_model(
                            input_ids=val_X[0], attention_mask=val_X[1], token_type_ids=val_X[2])
                        val_t = (v_teacher_logits, v_teacher_reps)

            N = trn_X[0].size(0)

            self.alpha_optim.zero_grad()
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, trn_t, val_t, lr, self.w_optim)
            self.alpha_optim.step()
            if self.config.multi_gpu:
                torch.distributed.barrier()
            self.w_optim.zero_grad()

            logits = self.model(trn_X)

            if self.config.use_emd:
                logits, s_layer_out = logits
            loss = self.model_to_print.crit(logits, trn_y)

            loss.backward()

            # gradient clipping
            # if not self.config.alpha_only:
            clip = clip_grad_norm_(self.model_to_print.weights(), self.config.w_grad_clip)
            self.w_optim.step()
            # if self.config.one_step and update_alpha:
            #     self.alpha_optim.step()
            if self.config.tb_dir != "":
                ds, ds2 = self.model.format_alphas()
                for layer_index, dsi in enumerate(ds):
                    self.writer.add_scalars(f'layer-{layer_index}-alpha', dsi, global_step=cur_step)
                for layer_index, dsi in enumerate(ds2):
                    self.writer.add_scalars(
                        f'layer-{layer_index}-softmax_alpha', dsi, global_step=cur_step)
                self.writer.add_scalar('loss', loss, global_step=cur_step)
                # self.writer.add_scalar("EMD", rep_loss, global_step=cur_step)
                self.writer.add_scalar("l1 loss", l1_loss, global_step=cur_step)

            preds = logits.detach().cpu().numpy()
            result, train_acc = get_acc_from_pred(self.output_mode, self.task_name, preds,
                                                  trn_y.detach().cpu().numpy())

            losses.update(loss.item(), N)
            top1.update(train_acc, N)
            # model.print_alphas(logger)

            if self.config.eval_during_train:
                if step + 1 in point:
                    self.model_to_print.print_alphas(self.logger)
                    self.logger.info("CURRENT Training Step [{:02d}/{:02d}] ".format(step, total_num_step ))
                    self.validate(epoch, cur_step, mode="train_dev")
                    genotypes = self.model_to_print.genotype()
                    self.logger.info("========genotype========\n" + str(genotypes))

            if step % self.config.print_freq == 0 or step == total_num_step - 1:
                self.logger.info(
                    "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f}, Prec@(1,5) {top1.avg:.1%}"
                    .format(
                        epoch + 1,
                        self.config.epochs,
                        step,
                        total_num_step - 1,
                        losses.avg,
                        top1=top1))
            cur_step += 1
        self.logger.info("{:.4%}".format(top1.avg))

    def train_sep(self, lr, epoch):
        top1 = AverageMeter()
        losses = AverageMeter()
        self.model_to_print.train()
        total_num_step = len(self.train_loader)
        self.logger.info(total_num_step)
        cur_step = epoch * len(self.train_loader)
        # valid_iter = iter(self.arch_loader)
        train_component = False
        point = [int(total_num_step * i / self.config.train_eval_time) for i in range(1, self.config.train_eval_time + 1)][:-1]
        self.logger.info(f"TRAINING ALPHA: {train_component}")
        for step, data in enumerate(self.train_loader):
            trn_X, trn_y = bert_batch_split(data, self.config.local_rank, device)
            N = trn_X[0].size(0)

            if self.config.multi_gpu:
                torch.distributed.barrier()
            loss = 0.0
            self.alpha_optim.zero_grad()
            self.w_optim.zero_grad()
            logits = self.model_to_print(trn_X)
            
            if self.config.use_emd:
                logits, s_layer_out = logits
            
            if epoch % 2 == 1:
                if self.config.use_emd:
                    with torch.no_grad():
                        teacher_logits, teacher_reps = self.teacher_model(
                            input_ids=trn_X[0], attention_mask=trn_X[1], token_type_ids=trn_X[2])
                    if self.config.hidn2attn:
                        s_layer_out = convert_to_attn(s_layer_out, trn_X[1])
                        teacher_reps = convert_to_attn(teacher_reps, trn_X[1])
                    if self.config.skip_mapping:
                        rep_loss = 0
                        teacher_reps = teacher_reps[1:][2::3]
                        for s_layerout, teacher_rep in zip(s_layer_out, teacher_reps):
                            rep_loss += nn.MSELoss()(s_layerout, teacher_rep)
                    else:
                        rep_loss, flow, distance = self.emd_tool.loss(
                            s_layer_out, teacher_reps, return_distance=True)
                        if self.config.update_emd:
                            self.emd_tool.update_weight(flow, distance)
                else:
                    rep_loss = 0.0
                loss = rep_loss * self.config.emd_rate + self.model_to_print.crit(
                    logits, trn_y) * self.config.alpha_ac_rate
                loss.backward()
                self.alpha_optim.step()
            else:
                loss = self.model_to_print.crit(logits, trn_y)
                loss.backward()
                # gradient clipping
                clip = clip_grad_norm_(self.model_to_print.weights(), self.config.w_grad_clip)
                self.w_optim.step()
            preds = logits.detach().cpu().numpy()
            result, train_acc = get_acc_from_pred(self.output_mode, self.task_name, preds, trn_y.detach().cpu().numpy())
            losses.update(loss.item(), N)
            top1.update(train_acc, N)

            if self.config.eval_during_train and self.config.local_rank == 0:
                if step + 1 in point:
                    self.model_to_print.print_alphas(self.logger)
                    self.logger.info("CURRENT Training Step [{:02d}/{:02d}] ".format(step, total_num_step))
                    self.validate(epoch, step, mode="train_dev")
                    genotypes = self.model_to_print.genotype()
                    self.logger.info("========genotype========\n" + str(genotypes))
                    self.model.train()
            if self.config.multi_gpu:
                torch.distributed.barrier()

            if step % 50 == 0 and self.config.local_rank == 0:
                self.logger.info(
                    "Train: , [{:2d}/{}] Step {:03d}/{:03d} Loss {:.3f}, Prec@(1,5) {top1.avg:.1%}"
                    .format(
                        epoch + 1,
                        self.config.epochs,
                        step,
                        total_num_step - 1,
                        losses.avg,
                        top1=top1))
                if epoch % self.config.alpha_ep != 0 and self.config.update_emd and self.config.use_emd:
                    self.logger.info("s weight:{}".format(self.emd_tool.s_weight))
                    self.logger.info("t weight:{}".format(self.emd_tool.t_weight))
            cur_step += 1
        self.logger.info("{:.4%}".format(top1.avg))

    def validate(self, epoch, cur_step, mode="dev"):
        eval_labels = []
        preds = []
        self.model_to_print.eval()

        total_loss, total_emd_loss = 0, 0
        task_name = self.task_name
        with torch.no_grad():
            for step, data in enumerate(self.eval_loader):
                X, y= bert_batch_split(data, self.config.local_rank, device)
                N = X[0].size(0)
                logits = self.model(X, train=False)
                rep_loss = 0
                if self.config.use_emd:
                    logits, s_layer_out = logits
                    teacher_logits, teacher_reps = self.teacher_model(
                        input_ids=X[0], attention_mask=X[1], token_type_ids=X[2])
                    if self.config.hidn2attn:
                        s_layer_out = convert_to_attn(s_layer_out, X[1])
                        teacher_reps = convert_to_attn(teacher_reps, X[1])
                    rep_loss, flow, distance = self.emd_tool.loss(
                        s_layer_out, teacher_reps, return_distance=True)
                    total_emd_loss += rep_loss.item()
                loss = self.model_to_print.crit(logits, y)
                total_loss += loss.item()
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                eval_labels.extend(y.detach().cpu().numpy())
            preds = preds[0]

            if self.task_name.lower() == 'wikiqa':
                preds = {"uids": self.eval_sids['dev'], 'scores':np.reshape(np.array([softmax(x) for x in preds]), -1)}
                eval_labels = "data/WikiQA/WikiQA-dev.tsv"
                task_name += 'dev'

            result, acc = get_acc_from_pred(self.output_mode, task_name, preds, eval_labels)

        self.logger.info(mode + ": [{:2d}/{}] Final Prec@1 {} Loss {}, EMD loss: {}".format(
            epoch + 1, self.config.epochs, result, total_loss, total_emd_loss))
        alpha_soft, alpha_ori = self.model_to_print.get_current_alphas()
        self.eval_result_map.append({'mode':mode, "epoch":epoch, "step":cur_step, "eval_result":acc, "alpha":alpha_soft, "alpha_ori":alpha_ori,"genotypes":self.model_to_print.genotype()})
        return acc


if __name__ == "__main__":
    architecture = SearchModel()
    architecture.main()
