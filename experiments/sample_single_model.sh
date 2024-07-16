#!/bin/bash

ARGS=$(getopt -o '' --longoptions hl:,hu:,bs:,lr:,es:,epoch:,aug:,notation:,device:,fold:,gamma:,samples:,temp:,batch:,seed:,dataset:,onlyNovels -- "$@")

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

epoch=
aug=
device=cpu
fold=
seed=0
onlyNovels=
dataset=grisoni

samples=6000
T=1.0
batch=1000

while true; do
  case "$1" in
    --hl ) hl="$2"; shift 2 ;;
    --hu) hu="$2"; shift 2 ;;
    --bs) bs="$2"; shift 2 ;;
    --es) es="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --aug) aug="Aug${2}"; shift 2 ;;
    --notation) notation="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --fold) fold="$2"; shift 2 ;;
    --gamma) gamma="$2"; shift 2 ;;
    --samples) samples="$2"; shift 2 ;;
    --temp) T="$2"; shift 2 ;;
    --batch) batch="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --onlyNovels) onlyNovels="--onlyNovels"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ $notation == 'fragsmiles' ]; then
    model=word_rnn
    experiment=${hl}hl_${hu}hu_${bs}bs_${es}es_${lr}lr
    embargument="--embedding_size ${es}"
    len=40
else
    model=char_rnn
    experiment="${hl}hl_${hu}hu_${bs}bs_${lr}lr"
    embargument=
    len=100
fi

case $notation in
fragsmiles) dir="${dataset}${aug}_fragSMILES-RNN"; ;;
selfies) dir="${dataset}${aug}_SELFIES-RNN"; ;;
smiles) dir="${dataset}${aug}_SMILES-RNN"; ;;
esac

if [[ "${fold}" ]]; then
    foldStr="_${fold}"
fi

prefixes=${model}${foldStr}

if [[ "${epoch}" ]]; then
    epochPad=$(printf "%03d" $epoch)
    epoch=_$((10#${epochPad}))
    epochPad=_$epochPad
fi

if [ $seed != '0' ]; then
    seedStr=_seed$seed
fi

if [ "${onlyNovels}" ]; then
    novelsStr=novels
fi

echo $notation started ${dir}/${experiment}/${prefixes}_generated${samples}${novelsStr}${epoch}${seedStr}_T${T}.csv $(date "+%Y/%m/%d %H:%M:%S")

cd ${dir}

## Here, if required, you can set loading of your python/conda environment
# source ~/.venv/bin/activate
# conda activate venv

python ../scripts/sample.py ${model} --device ${device} --model_load ./${experiment}/${prefixes}_model${epochPad}.pt --gen_save ./${experiment}/${prefixes}_generated${samples}${novelsStr}${epoch}${seedStr}_T${T}.csv --n_samples ${samples} --temp ${T} --max_len ${len} --n_batch ${batch} --config_load ./${experiment}/${prefixes}_config.pt --vocab_load ./${experiment}/${prefixes}_vocab.pt --notation $notation --seed ${seed} ${onlyNovels}

echo $notation finished ${dir}/${experiment}/${prefixes}_generated${samples}${novelsStr}${epoch}${seedStr}_T${T}.csv $(date "+%Y/%m/%d %H:%M:%S")