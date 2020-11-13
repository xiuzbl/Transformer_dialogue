gpuid=7

mark=$1
epoch_id=$2
pred_mark=$3

full_mark=${mark}_${epoch_id}_${pred_mark}

output=outputs/$mark
bigstore_path=../../bigstore_socialNetConv/
gt_file=data/data_v15_d1g10_for_base_transductive_ori/weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_t3000test.ori.tgt
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
