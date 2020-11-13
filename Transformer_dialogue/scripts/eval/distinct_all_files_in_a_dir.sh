input_dir=$1
#input_dir=results/dgn36un_datav14d1g10_pret160e4_accgrad0_fconemb_ada_vtg218

files=$(ls ${input_dir}/res*k)
#files=$(ls ${input_dir}/*.hyp)

for file in ${files[@]}; 
do
    echo $file
    sh scripts/eval/distinct.sh $file
done
#file=
#sh scripts/eval/distinct.sh $file
