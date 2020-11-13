dict=data/data_v01_d2/term2id.dict
#ids=results/resdv012_debug10m_d512_bs64_d8_1_test1k_d5
ids=data/data_v01_d2/weibo_78w_v01_d2_12kdict.test1k.src
terms=${ids}.txt

awk 'NR==FNR{d[$2]=$1} NR!=FNR{for(i=1;i<NF;i++){if($i in d){printf("%s ", d[$i])}} if($NF in d){print(d[$NF])}}' $dict $ids > $terms
