CUDA_VISIBLE_DEVICES=0 python trainer.py \
    --dataset ModelNet10 \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001 \
    --weight-decay 0 \
    --optimizer adam \
    --scheduler log \
    --model-type DeepSet \
    --config-path experiments/DeepSets.yaml \
    --exp_name deep-set-batchnorm

