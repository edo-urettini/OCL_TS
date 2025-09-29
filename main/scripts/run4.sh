D=0
d=50
M=ocar
o=sgd
h=24

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h --OCAR_alpha_ema=0.1 --OCAR_alpha_ema_grad=1.0
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h --OCAR_alpha_ema=0.1 --OCAR_alpha_ema_grad=1.0

CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ETTh2 --method=$M  --deg_f=$d --opt=$o --pred_len=$h --OCAR_alpha_ema=1.0 --OCAR_alpha_ema_grad=1.0
CUDA_VISIBLE_DEVICES=$D python fsnet/main.py --data=ECL --method=$M --deg_f=$d --opt=$o --pred_len=$h --OCAR_alpha_ema=1.0 --OCAR_alpha_ema_grad=1.0

