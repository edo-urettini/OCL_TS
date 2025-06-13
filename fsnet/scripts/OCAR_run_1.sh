CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ETTh2 --method=ocar --online_hpo
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ETTh1 --method=ocar 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=WTH --method=ocar 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ECL --method=ocar 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ETTm1 --method=ocar 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=Traffic --method=ocar 

