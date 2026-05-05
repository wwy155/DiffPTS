export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiffPE.py \
   --dataset_type="SolarEnergy" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --diffusion_steps=20 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'
