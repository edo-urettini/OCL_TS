D=0

M=ocar
i=3
h=1
d=50
o=sgd

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=WTH --method=$M --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm1 --method=$M --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Traffic --method=$M --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=Toy --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h --itr=$i
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ToyG --method=$M --features=S --deg_f=$d --opt=$o --pred_len=$h --itr=$i

