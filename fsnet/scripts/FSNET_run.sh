## M

i=1
ns=(1 )
bszs=(1 )
lens=(24 )
methods=('fsnet')
lrs=(1e-3 1e-5 1e-1)
alpha_emas=(0.5 0.1)
reguls=(0.1 10)

for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do
for lr in ${lrs[*]}; do
for alpha_ema in ${alpha_emas[*]}; do
for regul in ${reguls[*]}; do
CUDA_VISIBLE_DEVICES=1 python -u fsnet/main.py  --method $m --root_path /data/e.urettini/DATA/ --n_inner $n --test_bsz $bsz --data WTH --features M --seq_len 60 --label_len 0 --pred_len $len --des 'Exp' --itr $i --train_epochs 6 --learning_rate $lr --online_learning 'full' --OCAR_alpha_ema $alpha_ema --OCAR_regul $regul
done
done
done
done
done
done
done