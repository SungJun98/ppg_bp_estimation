# ConvTransform + CD_REx + Time_Cutmix
for num_filters in 4 8 12 16
do
for num_heads in 2
do
for d_model in 96      
do
for dropout in 0.1 0.2
do
for num_layer in 4  # 2 3 4
do
CUDA_VISIBLE_DEVICES=0 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
--num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
--max_epochs 100 --method cdrex_time \
--sbp_beta 0.1 --dbp_beta 0.1 \
--C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey \
--group_avg
done
done
done
done
done


