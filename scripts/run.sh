# This contains the sample script of running NTK-SAP

args=(
    --dataset cifar10 \
    --model resnet20 \
    --model-class lottery \
    --optimizer sgd \
    --train-batch-size 128 \
    --lr 0.1 \
    --lr-drops 80 120 \
    --weight-decay 1e-4 \
    --post-epochs 160 \
    --pruner NTKSAP \
    --prune-epochs 20 \
    --experiment multishot \
    --expid NTKSAP-resnet20 \
    --level-list 1 \
    --compression-list 18 16 14 12 10 8 6 4 2 \
    --prune-train-mode True \
    --ntksap_R 5 \
)

python main.py "${args[@]}"