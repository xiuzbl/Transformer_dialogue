gpuid=5

mark=$1
epoch_id=$2
pred_mark=$3

full_mark=${mark}_${epoch_id}_${pred_mark}

output=outputs/$mark
bigstore_path=../../bigstore_socialNetConv/
gt_file=data/data_v15_d1g10f1_for_base_ori/weibo_78wu_v14_d1_35.wotrain.t1k_as_test.filter_byn2v_as_test221_d1g10f1_s10p25_0p3_0p5_for_base_ori.src
checkpoint=${output}/checkpoint${epoch_id}.pt

mkdir -p $bigstore_path"/"$output
mkdir -p loggs results

function generate() {
CUDA_VISIBLE_DEVICES=${gpuid} \
python main.py \
    --run-mode test \
    --load-checkpoint $checkpoint \
    --pred_output_file results/res$full_mark \
    > loggs/logg$full_mark 2> loggs/errg$full_mark
}

function evaluate() {
    python scripts/bleu.py $gt_file results/res$full_mark 0 > results/res${full_mark}.bleu
}

generate
evaluate
