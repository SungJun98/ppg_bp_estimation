for C1 in 5e-6 # 1e-2 1e-3 1e-4 1e-5 5e-6 1e-6
do
	CUDA_VISIBLE_DEVICES=0 python main.py --method=ours-1 --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=100 --erm_loader --resume="/home/lsj9862/pakdd_etri/11-13_05-39-35_epoch499_erm_all_normal_0.005_0.001_0.1_64_2.pt" --C1=${C1} --mixup
done