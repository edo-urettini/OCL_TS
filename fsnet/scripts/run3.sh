CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh2 --method=ocar_fsnet --online_hpo 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh1 --method=ocar_fsnet  
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=WTH --method=ocar_fsnet 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ECL --method=ocar_fsnet 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTm1 --method=ocar_fsnet 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Traffic --method=ocar_fsnet 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Toy --method=ocar_fsnet --features=S 
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ToyG --method=ocar_fsnet --features=S 

CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh2 --method=ocar_fsnet --online_hpo --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh1 --method=ocar_fsnet --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=WTH --method=ocar_fsnet --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ECL --method=ocar_fsnet --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTm1 --method=ocar_fsnet --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Traffic --method=ocar_fsnet --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Toy --method=ocar_fsnet --features=S --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ToyG --method=ocar_fsnet --features=S --ng_only_last

CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh2 --method=ocar_derpp --online_hpo --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTh1 --method=ocar_derpp --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=WTH --method=ocar_derpp --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ECL --method=ocar_derpp --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ETTm1 --method=ocar_derpp --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Traffic --method=ocar_derpp --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=Toy --method=ocar_derpp --features=S --ng_only_last
CUDA_VISIBLE_DEVICES=1 python fsnet/main.py --data=ToyG --method=ocar_derpp --features=S --ng_only_last