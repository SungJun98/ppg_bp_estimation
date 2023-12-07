# ConvTransform + CD_REx + Time_Cutmix
# for num_filters in 4 8 12 # BEST : 12
# do
# for num_heads in 2 4      # BEST : 2
# do
# for d_model in 64 96      
# do
# for dropout in 0.1 0.2    # BEST : 0.1
# do
# for num_layer in 2 3 4    # BEST : 4
# do
# CUDA_VISIBLE_DEVICES=2 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
# --num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta 0.1 --dbp_beta 0.1 \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done
# done
# done
# done
# done


# for sbp_beta in 1e-5 1e-3 1e-1 1
# do
# for dbp_beta in 1e-5 1e-3 1e-1 1
# do
# CUDA_VISIBLE_DEVICES=4 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
# --num_filters 12 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 4 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done
# done

# for sbp_beta in 1e-4 5e-4 1e-3 5e-3 1e-2 # BEST : 5e-3 1e-3
# do
# for dbp_beta in 5e-6 1e-5 5e-5 1e-4       # BEST : 5e-6 1e-5
# do
# CUDA_VISIBLE_DEVICES=4 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
# --num_filters 12 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 4 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done
# done


# for sbp_beta in 1e-3
# do
# for dbp_beta in 1e-5
# do
# for C1 in 1e-4 1e-2 1e-1 1
# do
# for C21 in 1e-4 1e-2 1e-1 1
# do
# for C22 in 1e-4 1e-2 1e-1 1
# do
# CUDA_VISIBLE_DEVICES=4 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
# --num_filters 12 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 4 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
# done
# done
# done
# done
# done


# for sbp_beta in 1e-5
# do
# for dbp_beta in 0.1
# do
# for C1 in 1e-4 1e-2 1e-1 1
# do
# for C21 in 1e-4 1e-2 1e-1 1
# do
# for C22 in 1e-4 1e-2 1e-1 1
# do
# CUDA_VISIBLE_DEVICES=7 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
# --num_filters 12 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 4 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
# done
# done
# done
# done
# done


for sbp_beta in 1e-5
do
for dbp_beta in 0.1
do
for C1 in 5e-2 1e-1 5e-1
do
for C21 in 1e-5 5e-4 1e-4 5e-3 1e-3
do
for C22 in 5e-1 # 5e-3 1e-2 5e-2 1e-1 5e-1
do
CUDA_VISIBLE_DEVICES=3 python train.py --config ./core/config/dl/convtr/convtr_bcg.yaml \
--num_filters 12 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 4 \
--max_epochs 100 --method cdrex_time \
--sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
--C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
done
done
done
done
done