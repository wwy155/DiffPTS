export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffPTS.py \
   --dataset_type="SolarEnergy" \
   --device="cuda:3" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   config_wandb ProbForecastBase \
   runs --seeds='[1, 2, 3]'
