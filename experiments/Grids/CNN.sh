CUDA_VISIBLE_DEVICES=0 python trainer.py \
    --dataset MNIST \
    --epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0 \
    --optimizer adam \
    --scheduler log \
    --model-type CNN \
    --config-path experiments/Grids/CNN.yaml \
    --exp_name CNN-batchnorm

