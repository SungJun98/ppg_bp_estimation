# ConvTransform + CD_REx + Time_Cutmix

python train.py --config ./core/config/dl/convtr/convtr_ppgbp.yaml --max_epochs 1 --method cdrex_time --sbp_beta 0.1 --dbp_beta 0.1 --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey