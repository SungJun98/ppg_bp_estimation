for sbp_beta in 0.5 1 5 1e+1 1e+2 1e+3
do
	for dbp_beta in 1e-1 1 1e+1
	do
		python main.py --method=vrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=${sbp_beta} --dbp_beta=${dbp_beta} --save_every=100 --erm_loader --resume="./11-13_05-39-35_epoch499_erm_all_normal_0.005_0.001_0.1_64_2.pt"
	done
done
