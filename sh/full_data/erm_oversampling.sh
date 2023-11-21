## BEST
python3 main.py --method=erm --model=ConvTransformer --sampling=upsampling            \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000