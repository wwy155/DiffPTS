export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/RDIT.py \
   --dataset_type="Traffic" \
   --device="cuda:1" \
   --batch_size=1 \
   --pos=False \
   --d_model=2 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   config_wandb ProbForecastBase \
   runs --seeds='[1, 2, 3]'
