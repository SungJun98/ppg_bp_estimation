# ## ERM - ConvTransformer
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=erm --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/ERM/ERM_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=erm --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/ERM/ERM_SEED_${seed}_groupbest.pt" --rmse
# done


# ## ERM - Undersampling
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=erm --sampling=downsampling --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/ERM_Down/ERM_down_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=erm --sampling=downsampling --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/ERM_Down/ERM_down_SEED_${seed}_groupbest.pt" --rmse
# done


# ## ERM - Oversampling
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=erm --sampling=upsampling --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/ERM_Up/ERM_up_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=erm --sampling=upsampling --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/ERM_Up/ERM_up_SEED_${seed}_groupbest.pt" --rmse
# done


# ## GDRO
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=dro --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/GDRO/GDRO_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=dro --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/GDRO/GDRO_SEED_${seed}_groupbest.pt" --rmse
# done




# ## V-REx
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=vrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/VREx/VREX_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=vrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/pakdd_etri/best_models/VREx/VREX_SEED_${seed}_groupbest.pt" --rmse
# done



## C-REx (Ours-1)
# MAE
for seed in 0 1 2
do
    python test.py --method=crex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/C-REx/crex_SEED_${seed}_groupbest.pt"
done

# RMSE
for seed in 0 1 2
do
    python test.py --method=crex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/C-REx/crex_SEED_${seed}_groupbest.pt" --rmse
done



# ## D-REx (Ours-2)
# # MAE
# for seed in 0 1 2
# do
#     CUDA_VISIBLE_DEVICES=7 python test.py --method=drex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/D-REx/drex_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     CUDA_VISIBLE_DEVICES=7 python test.py --method=drex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/D-REx/drex_SEED_${seed}_groupbest.pt" --rmse
# done


# ## CD-REx (Ours-both)
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD-REx/cdrex_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD-REx/cdrex_SEED_${seed}_groupbest.pt" --rmse
# done


# ## CD-REx (Ours-both) + TC
# # MAE
# for seed in 0 1 2
# do
#     python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD_REX_TC/cdrex_tc_SEED_${seed}_groupbest.pt"
# done

# # RMSE
# for seed in 0 1 2
# do
#     python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD_REX_TC/cdrex_tc_SEED_${seed}_groupbest.pt" --rmse
# done