CUDA_VISIBLE_DEVICES=0 python trainer.py \
    --dataset MUTAG \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0 \
    --optimizer adam \
    --scheduler log \
    --model-type GNN \
    --config-path experiments/GNN.yaml \
    --exp_name GNN-batchnorm

