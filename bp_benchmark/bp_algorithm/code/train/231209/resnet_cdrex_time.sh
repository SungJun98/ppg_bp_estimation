# BCG
python train.py --config_file core/config/dl/resnet/resnet_bcg.yaml --lr 1e-3 --wd 1e-3 --no_result_save --max_epochs 100 --method cdrex_time --sbp_beta 0.1 --dbp_beta 0.1 --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey

# PPGBP
python train.py --config_file core/config/dl/resnet/resnet_ppgbp.yaml --lr 1e-3 --wd 1e-3 --no_result_save --max_epochs 100 --method cdrex_time --sbp_beta 0.1 --dbp_beta 0.1 --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey

# Sensors
python train.py --config_file core/config/dl/resnet/resnet_sensors.yaml --lr 1e-3 --wd 1e-3 --no_result_save --max_epochs 100 --method cdrex_time --sbp_beta 0.1 --dbp_beta 0.1 --C1 0.1 --C21 1.0 --C22 0.1 --beta 0.5 --tukey
