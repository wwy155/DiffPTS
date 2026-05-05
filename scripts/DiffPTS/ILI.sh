export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffPTS.py \
   --dataset_type="ILI" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=32 \
   --windows=168 \
   config_wandb ProbForecastBase \
   runs --seeds='[1, 2, 3]'









