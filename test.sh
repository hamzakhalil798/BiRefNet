#!/bin/bash
# Run inference & evaluation script

method="$1"
devices=${2:-0}
pred_root=${3:-e_preds}
resolutions=${4:-"config.size"}

task=$(python3 config.py --print_task)
ckpt_folder="/workspace/drive/MyDrive/train_birefnet/ckpts/${method}"

# Create pred_root early so eval doesn't fail
mkdir -p ${pred_root}

echo Inference started at $(date)

for resolution in ${resolutions}; do
    CUDA_VISIBLE_DEVICES=${devices} python inference.py \
        --pred_root ${pred_root} \
        --resolution ${resolution} \
        --ckpt_folder ${ckpt_folder}
done

echo Inference finished at $(date)

# Evaluation
log_dir=e_logs && mkdir -p ${log_dir}

testsets=$(python3 config.py --print_testsets)
testsets=(`echo ${testsets} | tr ',' ' '`) && testsets=${testsets[@]}

for testset in ${testsets}; do
    python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testset} --metrics 'all' > ${log_dir}/eval_${testset}.out
done

echo Evaluation finished at $(date)
