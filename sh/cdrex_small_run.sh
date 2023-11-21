# for C1 in 1e-3 # 1e-5
# do
# for C21 in 1e-3 1e-5
# do
# for C22 in 1e-3 1e-5
# do
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=3 python3 main.py --method=cdrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
# --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=500 --erm_loader --resume="/home/lsj9862/ppg_bp_estimation/pre_trained/erm_small_ratio_0.5_seed_${seed}.pt" \
# --C1=${C1} --tukey --C21=${C21} --C22=${C22} --sampling=small --small_ratio=0.5 --seed=${seed}
# done
# done
# done
# done


# for C1 in 5e-3 5e-4
# do
# for C21 in 5e-3 1e-3 5e-4
# do
# for C22 in 5e-3 1e-3 5e-4
# do
# for seed in 0 1 2
# do
# CUDA_VISIBLE_DEVICES=2 python3 main.py --method=cdrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
# --wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=500 --erm_loader --resume="/home/lsj9862/ppg_bp_estimation/pre_trained/erm_small_ratio_0.5_seed_${seed}.pt" \
# --C1=${C1} --tukey --C21=${C21} --C22=${C22} --sampling=small --small_ratio=0.5 --seed=${seed}
# done
# done
# done
# done



for C1 in 1e-4 5e-4 5e-5    # 5e-4
do
for C21 in 1e-3 5e-4 1e-4   # 5e-4
do
for C22 in 5e-3 1e-3        # 5e-3
do
for seed in 0 1 2
do
CUDA_VISIBLE_DEVICES=2 python3 main.py --method=cdrex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
--wd=5e-3 --lr=1e-3 --dropout=0.1 --annealing_epoch=0 --sbp_beta=1 --dbp_beta=0.1 --save_every=500 --erm_loader --resume="/home/lsj9862/ppg_bp_estimation/pre_trained/erm_small_ratio_0.5_seed_${seed}.pt" \
--C1=${C1} --tukey --C21=${C21} --C22=${C22} --sampling=small --small_ratio=0.5 --seed=${seed}
done
done
done
done