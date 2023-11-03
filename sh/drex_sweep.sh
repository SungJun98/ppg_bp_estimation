for lr in 1e-2 1e-3 1e-4
do
for wd in 5e-3
do
for drop_out in 0.1 0.3
do
for sbp_beta in 1e-3 1e-1
do
for dbp_beta in 1e-1 # 1e-3 1e-1
do
for C21 in 1e-2 # 1e-4 1e-2
do
for C22 in 1e-2 # 1e-4 1e-2
do
CUDA_VISIBLE_DEVICES=0 python main.py --method=drex --max_epoch=500 --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
--wd=${wd} --lr=${lr} --dropout=${drop_out} --annealing_epoch=0 \
--sbp_beta=${sbp_beta} --dbp_beta=${dbp_beta} --save_every=100 --erm_loader \
--resume="/home/lsj9862/ppg_bp_estimation/pre_trained/ERM_500epoch_SEED_0.pt" \
--tukey --C21=${C21} --C22=${C22}
done
done
done
done
done
done                    
done
