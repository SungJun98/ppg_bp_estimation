# for num_filters in 6 8 12
# do
# for num_heads in 2 4
# do
# for d_model in 32 64 96
# do
# for dropout in 0.1 0.2 0.4
# do
# for num_layer in 2 3 4
# do
# for wd in 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=2 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
# --lr 1e-3 --wd ${wd} --max_epochs 100 --method erm
# done
# done
# done
# done
# done
# done

# for num_filters in 6 8 12
# do
# for num_heads in 2 4
# do
# for d_model in 32 64 96
# do
# for dropout in 0.1 0.2 0.4
# do
# for num_layer in 2 3 4
# do
# for wd in 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=2 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
# --lr 1e-3 --wd ${wd} --max_epochs 100 --method erm
# done
# done
# done
# done
# done
# done


# for num_filters in 6 8 12
# do
# for num_heads in 2 4
# do
# for d_model in 32 64 96
# do
# for dropout in 0.1 0.2 0.4
# do
# for num_layer in 2 3 4
# do
# for wd in 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=2 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters ${num_filters} --num_heads ${num_heads} --d_model ${d_model} --dropout ${dropout} --num_layer ${num_layer} \
# --lr 1e-3 --wd ${wd} --max_epochs 100 --method erm
# done
# done
# done
# done
# done
# done

# for wd in 5e-3 1e-3 5e-4 1e-4 5e-5
# do
# CUDA_VISIBLE_DEVICES=0 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
# --num_filters 12 --num_heads 4 --d_model 96 --dropout 0.1 --num_layer 4 \
# --lr 1e-3 --wd ${wd} --max_epochs 100 --method erm
# done

CUDA_VISIBLE_DEVICES=0 python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml \
--num_filters 12 --num_heads 4 --d_model 96 --dropout 0.1 --num_layer 4 \
--lr 1e-3 --wd 5e-4 --max_epochs 100 --method erm
