CUDA_VISIBLE_DEVICES=3 python3 main.py --method=vrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=100 --erm_loader --resume=./erm_small_ratio_0.5_seed_0.pt --sampling=small --small_ratio=0.5 --seed=0

CUDA_VISIBLE_DEVICES=3 python3 main.py --method=vrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=100 --erm_loader --resume=./erm_small_ratio_0.5_seed_1.pt --sampling=small --small_ratio=0.5 --seed=1

CUDA_VISIBLE_DEVICES=3 python3 main.py --method=vrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=100 --erm_loader --resume=./erm_small_ratio_0.5_seed_2.pt --sampling=small --small_ratio=0.5 --seed=2