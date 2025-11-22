#!/bin/bash


MAX_JOBS=3  # Adjust this to control parallelism

reguls=(0.01 0.5 0.9 1.5)
emas=(0.1 0.3 0.7 0.99)
degrees=(10 25 250 500)
buff_sizes=(2 4 16 32)
datas=(ETTh1 ETTm1 WTH)

function run_job {
    local regul=$1
    local ema=$2
    local deg=$3
    local buff_size=$4
    local data=$5
    echo "Processing: data: $data, regul: $regul, ema: $ema, deg: $deg, buff size: $buff_size"
    python main/main_run.py --data=$data --method=natsr --pred_len=24 --opt=sgd --itr=1 --NatSR_regul=$regul --NatSR_alpha_ema=$ema --deg_f=$deg --replay_buff_size=$buff_size >out_params_sens/results_${data}_regul_${regul}_ema_${ema}_deg_${deg}_buff_${buff_size}.txt
}

# Track background jobs and limit concurrency
main_regul=0.14054211103075812
main_ema=0.5548965149124905
main_deg=50
main_buff=8

job_count=0
for data in "${datas[@]}"; do
    run_job "$main_regul" "$main_ema" "$main_deg" "$main_buff" "$data" &
    ((job_count++))

    if [ "$job_count" -ge "$MAX_JOBS" ]; then
        wait -n  # Wait for any one job to finish
        ((job_count--))
    fi
    for ema in "${emas[@]}"; do
        run_job "$main_regul" "$ema" "$main_deg" "$main_buff" "$data" &
        ((job_count++))

        if [ "$job_count" -ge "$MAX_JOBS" ]; then
            wait -n  # Wait for any one job to finish
            ((job_count--))
        fi
    done
    for deg in "${degrees[@]}"; do
        run_job "$main_regul" "$main_ema" "$deg" "$main_buff" "$data" &
        ((job_count++))

        if [ "$job_count" -ge "$MAX_JOBS" ]; then
            wait -n  # Wait for any one job to finish
            ((job_count--))
        fi
    done
    for buff in "${buff_sizes[@]}"; do
        run_job "$main_regul" "$main_ema" "$main_deg" "$buff" "$data" &
        ((job_count++))

        if [ "$job_count" -ge "$MAX_JOBS" ]; then
            wait -n  # Wait for any one job to finish
            ((job_count--))
        fi
    done
    for regul in "${reguls[@]}"; do
        run_job "$regul" "$main_ema" "$main_deg" "$main_buff" "$data" &
        ((job_count++))

        if [ "$job_count" -ge "$MAX_JOBS" ]; then
            wait -n  # Wait for any one job to finish
            ((job_count--))
        fi
    done
done
wait