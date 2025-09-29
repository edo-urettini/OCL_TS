#!/bin/bash

methods=(ogd er derpp fsnet onenet natsr)
datas=(ETTh1 ETTh2 ECL WTH ETTm1 ETTm2 Traffic)
pred_lens=(1 24 48)

for method in ${methods[*]}; do
for pred_len in ${pred_lens[*]}; do
for data in ${datas[*]}; do
if [ "$method" = "natsr" ]; then
    opt=sgd
else
    opt=adam
fi
if [[ "$data" = "Traffic" && "$pred_len" = "48" ]]; then
    # skip this case due to memory limit
    continue
fi
if [ "$data" = "ETTh1" ]; then
    CUDA_VISIBLE_DEVICES=1 python main/main.py --data=WTH --method=$method --pred_len=$pred_len --opt=$opt --online_hpo
else
    CUDA_VISIBLE_DEVICES=1 python main/main.py --data=$data --method=$method --pred_len=$pred_len --opt=$opt
fi
done
done
done    