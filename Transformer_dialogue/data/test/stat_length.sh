src=data_v00/weibo_78w_v00_12kdict.src
tgt=data_v00/weibo_78w_v00_12kdict.tgt

awk -v s=0 '{s+=NF} END{print s/NR}' $src > $src".avg_length" &
awk -v s=0 '{s+=NF} END{print s/NR}' $tgt > $tgt".avg_length" &
wait
