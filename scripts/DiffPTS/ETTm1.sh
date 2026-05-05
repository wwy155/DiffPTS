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
   runs --seeds='[1, 2, 3]'

