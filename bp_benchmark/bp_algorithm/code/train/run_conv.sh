# ConvTransform + CD_REx + Time_Cutmix
# for num_filters in 4 8    # BEST : 8
# do
# for num_heads in 2 4      # BEST : 2
# do
# for d_model in 32 64 96   # BEST : 96
# do
# for dropout in 0.1 0.2    # BEST : 0.1
# do
# for num_layer in 2 3      # BEST : 3
# do
# CUDA_VISIBLE_DEVICES=3 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta 0.1 --dbp_beta 0.1 \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done
# done
# done
# done
# done


# for sbp_beta in 1e-4 1e-3 1e-2 1e-1 1   # BEST : 1e-4
# do
# for dbp_beta in 1e-4 1e-3 1e-2 1e-1 1   # BEST : 1e-3
# do
# CUDA_VISIBLE_DEVICES=3 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done
# done


# for wd in 1e-5 1e-4 1e-3 5e-3 1e-2 # BEST: 0
# do
# CUDA_VISIBLE_DEVICES=3 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta 1e-4 --dbp_beta 1e-3 \
# --wd ${wd} \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done

# for lr in  1e-5 1e-4 1e-3 5e-3 1e-2  # BEST : 1e-3
# do
# CUDA_VISIBLE_DEVICES=4 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta 1e-4 --dbp_beta 1e-3 \
# --lr ${lr} \
# --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
# done

# for C1 in 1e-5 1e-3 1e-1 1
# do
# for C21 in 1e-5 1e-3 1e-1 1
# do
# for C22 in 1e-5 1e-3 1e-1 1
# do
# CUDA_VISIBLE_DEVICES=7 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta 1e-4 --dbp_beta 1e-3 \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
# done
# done
# done


# for C1 in 5e-2 1e-1 0.5
# do
# for C21 in 0.5 1 1.5
# do
# for C22 in 0.05 0.1 0.5
# do
# CUDA_VISIBLE_DEVICES=0 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta 1e-4 --dbp_beta 1e-3 \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
# done
# done
# done
# done

#######################################################
# for num_filters in 8    # BEST : 8
# do
# for num_heads in 2      # BEST : 2
# do
# for d_model in 96   # BEST : 96
# do
# for dropout in 0.1    # BEST : 0.1
# do
# for num_layer in 3      # BEST : 3
# do
# for sbp_beta in 1e-4 1e-2 1
# do
# for dbp_beta in 1e-4 1e-2 1
# do
# for C1 in 1e-4 1e-2 1
# do
# for C21 in 1e-4 1e-2 1
# do
# for C22 in 1e-4 1e-2 1
# do
# for beta in 0.3 0.7
# do
# CUDA_VISIBLE_DEVICES=4 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
# --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta ${beta} --tukey
# done
# done
# done
# done
# done
# done
# done
# done
# done
# done
# done
# done
# done


CUDA_VISIBLE_DEVICES=4 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
--num_filters 8 --num_heads 2 --d_model 96 --dropout 0.1 --num_layer 3 \
--max_epochs 100 --method cdrex_time \
--sbp_beta 1e-2 --dbp_beta 1 \
--C1 1e-4 --C21 1 --C22 1e-4 --beta 0.7 --tukey