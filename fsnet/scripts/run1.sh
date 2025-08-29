D=1
M=ocar_temp
d=500

M=er
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTm2 --method=$M --deg_f=$d --opt=adam

