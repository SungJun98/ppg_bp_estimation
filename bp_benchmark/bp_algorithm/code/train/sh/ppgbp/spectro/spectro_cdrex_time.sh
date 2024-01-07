# PPGBP
# for wd in 1e-2 1e-3 1e-4
# do
# for sbp_beta in 1e-1 1e-2 1e-3 1e-4
# do
# for dbp_beta in 1e-1 1e-2 1e-3 1e-4
# do
# for C1 in 1e-1 1e-2 1e-3 1e-4
# do
# for C21 in 1e-1 1e-2 1e-3 1e-4 
# do
# for C22 in 1e-1 1e-2 1e-3 1e-4
# do
# CUDA_VISIBLE_DEVICES=4 python train.py --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml \
# --lr 1e-3 --wd ${wd} --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
# done
# done
# done
# done
# done
# done

# for wd in 1e-1 5e-2 1e-3 5e-3
# do
# for sbp_beta in 5e-4 1e-4 5e-5 1e-5
# do
# for dbp_beta in 5e-2 1e-2 5e-3
# do
# for C1 in 5e-4 1e-4 5e-5 1e-5
# do
# for C21 in 1 5e-1 1e-1 5e-2
# do
# for C22 in 5e-2 1e-2 5e-3
# do
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml \
# --lr 1e-3 --wd ${wd} --max_epochs 100 --method cdrex_time \
# --sbp_beta ${sbp_beta} --dbp_beta ${dbp_beta} \
# --C1 ${C1} --C21 ${C21} --C22 ${C22} --beta 0.5 --tukey
# done
# done
# done
# done
# done
# done


CUDA_VISIBLE_DEVICES=2 python train.py --config_file core/config/dl/spectroresnet/spectroresnet_ppgbp.yaml \
--lr 1e-3 --wd 1e-3 --max_epochs 100 --method cdrex_time \
--sbp_beta 1e-4 --dbp_beta 1e-2 \
--C1 5e-5 --C21 1e-1 --C22 1e-2 --beta 0.5 --tukey
