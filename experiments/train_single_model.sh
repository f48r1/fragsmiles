#!/bin/bash

ARGS=$(getopt -o '' --longoptions hl:,hu:,bs:,lr:,es:,savefr:,epochs:,aug:,notation:,device:,fold:,gamma:,dataset: -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$ARGS"

es=300
hu=256
bs=512
hl=2
lr=0.001
gamma=1.0
epochs=12
notation=

savefr=1
aug=
device=cpu
fold=
dataset=grisoni


while true; do
  case "$1" in
    --hl ) hl="$2"; shift 2 ;;
    --hu) hu="$2"; shift 2 ;;
    --bs) bs="$2"; shift 2 ;;
    --es) es="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --savefr) savefr="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --aug) aug="Aug${2}"; shift 2 ;;
    --notation) notation="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --fold) fold="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --gamma) gamma="$2"; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done


if [ $notation == 'fragsmiles' ]; then
    model=word_rnn
    experiment=${hl}hl_${hu}hu_${bs}bs_${es}es_${lr}lr
    embargument="--embedding_size ${es}"
else
    model=char_rnn
    experiment="${hl}hl_${hu}hu_${bs}bs_${lr}lr"
fi

case $notation in
fragsmiles) dir="${dataset}${aug}_fragSMILES-RNN"; ;;
selfies) dir="${dataset}${aug}_SELFIES-RNN"; ;;
smiles) dir="${dataset}${aug}_SMILES-RNN"; ;;
esac

if [[ "${fold}" ]]; then
    foldStr="_${fold}"
    testargument="--val_load ../data/${dataset}_test${aug}${foldStr}.tar.xz"
fi

train=../data/${dataset}_train${aug}${foldStr}.tar.xz
prefixes=${model}${foldStr}


mkdir -p ${dir}/${experiment}
cd $dir

echo $notation starting ${dir}/${experiment}/${prefixes} on $(date "+%Y/%m/%d %H:%M:%S")

## Here, if required, you can set loading of your python/conda environment
# source ~/.venv/bin/activate
# conda activate venv

python ../scripts/train.py ${model} --train_load ${train} $testargument --config_save ./${experiment}/${prefixes}_config.pt --log_file ./${experiment}/${prefixes}_log.txt --save_frequency ${savefr} --lr ${lr} --n_batch ${bs} --num_layers ${hl} --hidden ${hu} ${embargument} --train_epochs ${epochs} --gamma ${gamma} --notation ${notation} --vocab_save ./${experiment}/${prefixes}_vocab.pt --model_save ./${experiment}/${prefixes}_model.pt --device ${device}
echo $notation finished ${dir}/${experiment}/${prefixes} on $(date "+%Y/%m/%d %H:%M:%S")