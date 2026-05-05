export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffPTS.py \
   --dataset_type="ETTm2" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --e_layers=2 \
   --lr=0.0001 \
   --diffusion_steps=1 \
   --pred_len=192 \
   --dropout=0.05 \
   --cond_backbone="patchtst" \
   --l2_weight_decay=0.0005 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'
# export PYTHONPATH=./
# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ./src/experiments/DiffPTS.py \
#    --dataset_type="ETTm2" \
#    --device="cuda:1" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len=192 \
#    --windows=168 \
#    --diff_steps=20 \
#    --dropout=0.00 \
#    config_wandb ProbForecastBase \
#    runs --seeds='[1, 2, 3]'
