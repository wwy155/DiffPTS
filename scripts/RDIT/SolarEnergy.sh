export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/RDIT.py \
   --dataset_type="SolarEnergy" \
   --device="cuda:2" \
   --batch_size=1 \
   --pos=False \
   --d_model=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'
