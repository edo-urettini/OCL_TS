CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh2 --method=ocar --online_hpo --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh1 --method=ocar --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=WTH --method=ocar --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ECL --method=ocar --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTm1 --method=ocar --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Traffic --method=ocar --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Toy --method=ocar --features=S --deg_f=20 --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ToyG --method=ocar --features=S --deg_f=20 --ng_only_last


CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh2 --method=ocar --online_hpo --deg_f=20 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh1 --method=ocar --deg_f=20 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=WTH --method=ocar --deg_f=20
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ECL --method=ocar --deg_f=20 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTm1 --method=ocar --deg_f=20 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Traffic --method=ocar --deg_f=20 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Toy --method=ocar --features=S --deg_f=20 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ToyG --method=ocar --features=S --deg_f=20 