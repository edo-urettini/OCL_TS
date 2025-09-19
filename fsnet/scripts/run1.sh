D=1
d=50
M=fsnet
o=adam
h=1

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=onenet
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=er
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=derpp
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=ogd
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h





h=48
M=fsnet
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=onenet
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=er
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=derpp
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h

M=ogd
o=adam

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h






