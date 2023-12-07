# ConvTransform + CD_REx + Time_Cutmix
for sbp_beta in 1e-4 1e-2 1 
do
for dbp_beta in 1e-4 1e-2 1 
do
for C1 in 1e-4 1e-2 1e-1 1
do
for C21 in 1e-4 1e-2 1e-1 1
do
for C22 in 1e-4 1e-2 1e-1 1
do
CUDA_VISIBLE_DEVICES=0 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
--num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
--max_epochs 100 --method cdrex_time \
--sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
--C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey --group_avg
done
done
done
done
done