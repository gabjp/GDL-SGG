CUDA_VISIBLE_DEVICES=0 python trainer.py \
    --dataset ModelNet10 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --weight-decay 0 \
    --optimizer adam \
    --scheduler log \
    --model-type MLP \
    --config-path experiments/MLPSets.yaml \
    --exp_name mlp-set-batchnorm

