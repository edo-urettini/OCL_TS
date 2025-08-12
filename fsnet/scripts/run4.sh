CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ETTh2 --method=derpp --online_hpo
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ETTh1 --method=derpp 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=WTH --method=derpp 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ECL --method=derpp 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ETTm1 --method=derpp 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=Traffic --method=derpp 
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=Toy --method=derpp --features=S
CUDA_VISIBLE_DEVICES=0 python fsnet/main.py --data=ToyG --method=derpp --features=S

