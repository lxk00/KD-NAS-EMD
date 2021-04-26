from torch.utils.data import TensorDataset
import torch
import numpy as np
import random
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import os


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id
        self.guid = guid

def get_tensor_data(output_mode, features, ):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    s_ids = [f.guid for f in features]
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_seq_lengths)
    return tensor_data, s_ids


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode, is_master=True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and is_master:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index == 0 and is_master:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: {}".format(example.label))
            print("label_id: {}".format(label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                seq_length=seq_length,
                guid=example.guid))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def load_glue_dataset(config):
    from bert_fineturn.data_processor.glue import glue_processors as processors
    from bert_fineturn.data_processor.glue import glue_output_modes as output_modes
    from transformers import BertConfig, BertTokenizer

    task_name = config.datasets
    config.is_master = True
    config.multi_gpu = False
    processor = processors[task_name.lower()]()
    output_mode = output_modes[task_name.lower()]
    label_list = processor.get_labels()
    if output_mode == 'classification':
        n_classes = len(label_list)
    else:
        n_classes = 1
    sids = dict()
    tokenizer = BertTokenizer.from_pretrained('teacher_utils/bert_base_uncased', do_lower_case=True)

    train_examples = processor.get_train_examples(config.data_src_path)
    train_features = convert_examples_to_features(train_examples, label_list,
                                                        config.max_seq_length, tokenizer,
                                                        output_mode, config.is_master)
    train_data, train_sids = get_tensor_data(output_mode, train_features)

    eval_examples = processor.get_dev_examples(config.data_src_path)
    eval_features = convert_examples_to_features(eval_examples, label_list,
                                                    config.max_seq_length, tokenizer,
                                                    output_mode, config.is_master)
    eval_data, eval_sids = get_tensor_data(output_mode, eval_features)

    test_examples = processor.get_test_examples(config.data_src_path)
    test_features = convert_examples_to_features(test_examples, label_list,
                                                    config.max_seq_length, tokenizer,
                                                    output_mode, config.is_master)
    test_data, test_sids = get_tensor_data(output_mode, test_features)

    train_eval_data, _ = get_tensor_data(output_mode, eval_features)

    if not config.multi_gpu:
        train_sampler = RandomSampler(train_data)
        train_eval_sampler = RandomSampler(train_eval_data)
    else:
        train_sampler = DistributedSampler(train_data)
        train_eval_sampler = DistributedSampler(train_eval_data)
    eval_sampler = SequentialSampler(eval_data)
    test_sampler = SequentialSampler(test_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=config.batch_size)
    train_eval_dataloader = DataLoader(train_eval_data, sampler=train_eval_sampler, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=config.batch_size)
    bert_config = BertConfig.from_pretrained("teacher_utils/bert_base_uncased/config.json")
    config.bert_config = bert_config

    sids = {"train": train_sids, "test": test_sids, "dev": eval_sids}

    return train_dataloader, train_eval_dataloader, eval_dataloader, test_dataloader, output_mode, n_classes, config, sids


from torch.utils.data.sampler import Sampler
class OrderdedSampler(Sampler):
    def __init__(self, dataset, order):
        self._dataset = dataset
        self._train_data_list = order
        self._train_data_list
    def __len__(self):
        return len(self._dataset)
    def __iter__(self):
        random.shuffle(self._train_data_list)
        for index in self._train_data_list:
            yield self._dataset[index]

def check_data_vaild(data1, data2):
    # data1, data2 = next(iter(data1)), next(iter(data2))
    def pad_replace(x):
        x = np.array(x)
        pad_mask = np.array([not(i == '[PAD]' or i == "<pad>") for i in x])
        new_x = x[pad_mask].tolist() + [f'[PAD] * { - sum(pad_mask - 1)}']
        return new_x
    def mask_replace(x):
        t = sum(x)
        new_x = f"1 * {t}, 0 * {len(x) - t}"
        return new_x
    with open('/data/lxk/NLP/github/darts-KD/data/MRPC-nas/embedding/vocab.txt') as f:
        vocab1 = {i:x.strip() for i, x in enumerate(f.readlines())}
    with open('/data/lxk/NLP/github/darts-KD/teacher_utils/teacher_model/MRPC/vocab.txt') as f:
        vocab2 = {i:x.strip() for i, x in enumerate(f.readlines())}

    sent_words = torch.split(data1[0], 1, dim=1)
    sent_words = [torch.squeeze(x, dim=1) for x in sent_words]

    mask = [x.ne(0) for x in sent_words]
    if len(mask) > 1:
        mask = torch.logical_or(mask[0], mask[1])
    else:
        mask = mask[0]

    print("SENT1:", pad_replace([vocab1[x.item()] for x in data1[0][0][0]]))
    if data1[0].shape[1] == 2:
        print("SENT2:", pad_replace([vocab1[x.item()] for x in data1[0][0][1]]))

    print("MASK:", mask_replace(mask[0]))

    print("LABEL:", data1[2][0].item())

    input_ids, input_mask, segment_ids, label_ids, seq_lengths = data2


    print("TEACHER SENT:", pad_replace([vocab2[x.item()] for x in input_ids[0]]))
    print("TEACHER MASK", mask_replace(input_mask[0]))
    print("TEACHER LABEL", label_ids[0].item())

class RandomSamplerByOrder(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples
    
def bert_batch_split(data, rank, device=None):
    if device == torch.device('cpu'):
        data = [x for x in data]
    else:
        data = [x.to(f"cuda:{rank}", non_blocking=True) for x in data]
    input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
    X = [input_ids, input_mask, segment_ids, seq_lengths]
    Y = label_ids
    return X, Y
