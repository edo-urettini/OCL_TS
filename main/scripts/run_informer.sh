#!/bin/bash

methods=(ogd_informer er_informer)
datas=(ETTh1 ETTh2 ECL WTH ETTm1 ETTm2) # Traffic)
#pred_lens=(1 24 48)
pred_lens=(48) #24 48)

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
echo "Processing: method: $method, data: $data, pred_len: $pred_len"

if [ "$data" = "ETTh1" ]; then
    python main/main_run.py --data=ETTh1 --method=$method --pred_len=$pred_len --opt=$opt --online_hpo --itr=1 >output_informer/res_tune_${method}_${data}_${pred_len}.txt
else
    python main/main_run.py --data=$data --method=$method --pred_len=$pred_len --opt=$opt >output_informer/res_${method}_${data}_${pred_len}.txt
fi
done
done
done    