CUDA_VISIBLE_DEVICES=1 python trainer.py \
    --dataset MNIST \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0 \
    --optimizer adam \
    --scheduler log \
    --model-type MLP \
    --config-path experiments/Grids/MLPGrids.yaml \
    --exp_name MLPGrids-batchnorm

