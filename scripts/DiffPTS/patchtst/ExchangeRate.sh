export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
# python3 ./src/experiments/ODiff0.py \
#    --dataset_type="ExchangeRate" \
#    --device="cuda:1" \
#    --batch_size=32 \
#    --horizon=1 \
#    --pred_len=192 \
#    --windows=168 \
#    config_wandb ProbForecastBase \
#    runs --seeds='[1, 2, 3]'

python3 ./src/experiments/DiffPTS.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --cond_backbone="patchtst" \
   --pred_len=192 \
   --windows=168 \
   config_wandb ProbForecastBase \
   runs --seeds='[1, 2, 3]'


# python3 ./src/experiments/ODiff0.py \
#    --dataset_type="ExchangeRate" \
#    --device="cuda:1" \
#    --batch_size=32 \
#    --horizon=1 \
#    --beta_end=0.02 \
#    --diffusion_steps=100 \
#    --pred_len=192 \
#    --windows=168 \
#    runs --seeds='[111221, 2, 3]'
