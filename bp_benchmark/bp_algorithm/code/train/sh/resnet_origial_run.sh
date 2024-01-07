# BCG
python train.py --config_file core/config/dl/resnet/resnet_bcg.yaml --lr 1e-3 --wd 0 --max_epochs 100 --method erm

# PPGBP
python train.py --config_file core/config/dl/resnet/resnet_ppgbp.yaml --lr 1e-3 --wd 0 --max_epochs 100 --method erm

# Sensors
python train.py --config_file core/config/dl/resnet/resnet_sensors.yaml --lr 1e-3 --wd 0 --max_epochs 100 --method erm
