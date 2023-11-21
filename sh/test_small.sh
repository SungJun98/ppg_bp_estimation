## C-REx (Ours-1)
# MAE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=crex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/C-REx/small_ratio=0.5/crex_SEED_${seed}_groupbest_small_ratio_0.5.pt"
done

## RMSE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=crex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/C-REx/small_ratio=0.5/crex_SEED_${seed}_groupbest_small_ratio_0.5.pt" \
    --rmse
done


## D-REx (Ours-2)
# MAE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=drex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/D-REx/small_ratio=0.5/drex_SEED_${seed}_groupbest_small_ratio_0.5.pt"
done

## RMSE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=drex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/D-REx/small_ratio=0.5/drex_SEED_${seed}_groupbest_small_ratio_0.5.pt" \
    --rmse
done


## CD-REx (Ours-both)
# MAE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD-REx/small_ratio=0.5/cdrex_SEED_${seed}_groupbest_small_ratio_0.5.pt"
done

## RMSE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD-REx/small_ratio=0.5/cdrex_SEED_${seed}_groupbest_small_ratio_0.5.pt" \
    --rmse
done


## CD-REx (Ours-both) + TC
# MAE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD_REX_TC/small_ratio=0.5/cdrex_tc_SEED_${seed}_groupbest_small_ratio_0.5.pt"
done

## RMSE
for seed in 0 1 2
do
    CUDA_VISIBLE_DEVICES=7 python test.py --method=cdrex --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
    --sampling=small --small_ratio=0.5 \
    --seed=${seed} --resume="/home/lsj9862/ppg_bp_estimation/best_models/CD_REX_TC/small_ratio=0.5/cdrex_tc_SEED_${seed}_groupbest_small_ratio_0.5.pt" \
    --rmse
done