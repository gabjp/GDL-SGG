CUDA_VISIBLE_DEVICES=1 python trainer.py \
    --dataset MUTAG \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0 \
    --optimizer adam \
    --scheduler log \
    --model-type MLP \
    --config-path experiments/MLPGraphs.yaml \
    --exp_name MLPGraphs-batchnorm

