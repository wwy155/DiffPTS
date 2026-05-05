export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/DiffPTS.py \
   --dataset_type="ETTh2" \
   --device="cuda:1" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --cond_backbone="patchtst" \
   --windows=168 \
   runs --seeds='[1, 2, 3]'

