python3 main.py --method=crex --model=ConvTransformer                                   \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4     \
        --wd=5e-3 --lr=1e-3 --dropout=0.1  --max_epoch=1000 --annealing_epoch=500 \
        --sbp_beta=1 --dbp_beta=0.1 --erm_loader --C1=1e-4