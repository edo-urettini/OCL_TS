D=0
M=ocar_temp
o=adam
d=50

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=o --online_hpo
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=o 
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=o
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=o
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=o
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=o
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=o
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=o
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=o
