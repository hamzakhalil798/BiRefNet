#!/bin/bash
# Run script
# Settings of training & test for different tasks.
method="$1"
task=$(python3 config.py --print_task)
case "${task}" in
    'DIS5K') epochs=500 && val_last=50 && step=5 ;;
    'COD') epochs=150 && val_last=50 && step=5 ;;
    'HRSOD') epochs=150 && val_last=50 && step=5 ;;
    'Custom') epochs=$((244+250)) && val_last=250 && step=5 ;;
    'General-2K') epochs=250 && val_last=30 && step=2 ;;
    'Matting') epochs=150 && val_last=50 && step=5 ;;
esac

# Train
devices=$2
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)

to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)
resume_weights_path='/workspace/weights/cv/BiRefNet-general-epoch_244.pth'
ckpt_dir="/workspace/drive/MyDrive/train_birefnet/ckpts/${method}"
if [ ${to_be_distributed} == "True" ]
then
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --standalone --nproc_per_node $((nproc_per_node+1)) \
    train.py --ckpt_dir ${ckpt_dir} --epochs ${epochs} \
        --dist ${to_be_distributed} \
        --resume ${resume_weights_path} \
        --use_accelerate
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --ckpt_dir ${ckpt_dir} --epochs ${epochs} \
        --dist ${to_be_distributed} \
        --resume ${resume_weights_path} \
        --use_accelerate
fi

echo Training finished at $(date)
