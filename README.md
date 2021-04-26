# KD_EMD

## Requirements

- python 3
- pytorch >= 0.4.1
- numpy

## Run example

Adjust the batch size if out of memory (OOM) occurs. It dependes on your gpu memory size and genotype.

- Download glue data to dir ./data/glue_data
- Prepare finetuned BERT model

```shell
python teacher_utils/finetune.py --datasets MRPC
```

- Search

```shell
python3 search.py --datasets MRPC
```

- Augment

```shell
python3 augment.py --datasets MRPC --genotype "Genotype(normal=[[('conv_3x3', 0)], [('highway', 0)], [('conv_3x3', 2)], [('conv_5x5', 1)], [('conv_3x3', 3)], [('conv_3x3', 1)]], normal_concat=range(1, 7), reduce=[], reduce_concat=range(1, 7))"
```