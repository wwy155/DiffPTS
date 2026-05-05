
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffPTS.py \
   --dataset_type="ETTh2" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --lr=0.001 \
   --diffusion_steps=20 \
   --pred_len=192 \
   --dropout=0.05 \
   --l2_weight_decay=0.0005 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'
