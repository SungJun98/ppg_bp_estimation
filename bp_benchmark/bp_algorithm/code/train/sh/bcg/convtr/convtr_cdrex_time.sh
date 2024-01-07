for sbp_beta in 1e-1 1e-3 1e-5
do
for dbp_beta in 1e-1 1e-3 1e-5
do
for C1 in 1e-1 1e-3 1e-5
do
for C21 in 1e-5 # 1e-1 1e-3 1e-5
do
for C22 in 1e-5 # 1e-1 1e-3 1e-5
do
CUDA_VISIBLE_DEVICES=2 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
--num_filters 6 --num_heads 2 --d_model 64 --dropout 0.1 --num_layer 2 \
--lr 1e-3 --wd 1e-2 --max_epochs 100 --method cdrex_time \
--sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
--C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
done
done
done
done
done

