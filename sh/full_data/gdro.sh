python3 main.py --method=dro --model=ConvTransformer                                  \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --wd=1e-1 --lr=1e-3 --dropout=0.1 --erm_loader --max_epoch=1000       \
        --robust_step_size=0.01