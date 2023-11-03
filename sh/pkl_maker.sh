for seed in 0 1 2 3 4
do
	python3 main.py --method=erm --seed=$seed --DFR --save_pkl --exit
	python3 main.py --method=vrex --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --sampling=upsampling --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --sampling=downsampling --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --mode=normal --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --mode=hyper2 --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --mode=hypo --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --mode=crisis --seed=$seed --save_pkl --exit
	python3 main.py --method=erm --mode=prehyper --seed=$seed --save_pkl --exit

done
