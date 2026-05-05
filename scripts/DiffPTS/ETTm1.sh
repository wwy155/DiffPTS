export PYTHONPATH=./
# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ./src/experiments/ODiff0.py \
#    --dataset_type="ETTm1" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len=192 \
#    --windows=168 \
#    config_wandb ProbForecastBase \
#    runs --seeds='[1, 2, 3]'


# python3 ./src/experiments/ODiff0.py \
#    --dataset_type="ETTm1" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --diffusion_steps=50 \
#    --pred_len=192 \
#    --windows=168 \
#    config_wandb ProbForecastBase \
#    runs --seeds='[1, 2, 3]'


export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffPTS.py \
   --dataset_type="ETTm1" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   --cond_backbone="patchtst" \
   config_wandb ProbForecastBase \
   runs --seeds='[1, 2, 3]'


# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ./src/experiments/ODiff0.py \
#    --dataset_type="ETTm1" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len=192 \
#    --windows=168 \
#    --diffusion_steps=20 \
#    config_wandb ProbForecastBase \
#    runs --seeds='[1, 2, 3]'

# CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ./src/experiments/ODiff0.py \
#    --dataset_type="ETTm1" \
#    --device="cuda:0" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len=192 \
#    --windows=168 \
#    --diffusion_steps=10 \
#    config_wandb ProbForecastBase \
#    runs --seeds='[1, 2, 3]'
