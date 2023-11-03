for small_ratio in 0.5
do
	for seed in 0 1 2
	do
		CUDA_VISIBLE_DEVICES=1 python3 main.py --method=erm --sampling=small --small_ratio=$small_ratio --seed=$seed --save_pkl --exit
	done
done

for small_ratio in 0.5
do
	for seed in 0 1 2
	do
		CUDA_VISIBLE_DEVICES=2 python3 main.py --method=erm --sampling=small --small_ratio=$small_ratio --up_small --seed=$seed --save_pkl --exit
	done
done

for small_ratio in 0.5
do
	for seed in 0 1 2
	do
		CUDA_VISIBLE_DEVICES=1 python3 main.py --method=erm --sampling=small --small_ratio=$small_ratio --down_small --seed=$seed --save_pkl --exit
	done
done
