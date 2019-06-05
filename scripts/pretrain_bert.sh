#!/bin/bash

RANK=0
WORLD_SIZE=1

python3.6 -u pretrain_bert.py \
    --batch-size 32 \
    --tokenizer-type BertWordPieceTokenizer \
    --cache-dir cache_dir \
    --tokenizer-model-type bert-large-uncased \
    --vocab-size 30522 \
    --presplit-sentences \
    --train-data /megatron_workspace/Megatron/Megatron-LM/AA_wiki.json \
    --loose-json \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --num-layers 24 \
    --hidden-size 1024 \
    --intermediate-size 4096 \
    --num-attention-heads 16 \
    --hidden-dropout 0.1 \
    --fp16 \
