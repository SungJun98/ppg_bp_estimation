for wd in 1e-2 5e-3 1e-3
do
	for lr in 1e-4 5e-3 1e-3 0.075 0.05
	do
		for dropout in 0.0 0.1 0.5
		do
			python main.py --method=erm --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --wd=${wd} --lr=${lr} --dropout=${dropout}
		done
	done
done
