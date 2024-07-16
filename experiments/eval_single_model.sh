#!/bin/bash

ARGS=$(getopt -o '' --longoptions hl:,hu:,bs:,lr:,es:,epoch:,aug:,notation:,njobs:,fold:,gamma:,samples:,temp:,seed:,dataset:,onlyNovels,device: -- "$@")

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
njobs=1

samples=6000
T=1.0
batch=512
dataset=grisoni

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
    --fold) fold="$2"; shift 2 ;;
    --gamma) gamma="$2"; shift 2 ;;
    --samples) samples="$2"; shift 2 ;;
    --temp) T="$2"; shift 2 ;;
    --batch) batch="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --njobs) njobs="$2"; shift 2 ;;
    --onlyNovels) onlyNovels="novels"; shift ;;
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

# qui l'augmented e` importante perche` determina la cartella dell'esperimento
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

#Augmented dataset is not involved for the evualuation phase because we need only canonical sampled smiles
train="../data/${dataset}_train${foldStr}.tar.xz"
test="../data/${dataset}_test${foldStr}.tar.xz"

if [ $dataset == 'moses' ]; then
    #'--ptest_path ../data/moses_train_stats.npz'
    # ptestStr=
    # testStr=
    testStr="--test_path ${test}"
else
    testStr="--test_path ${train}"
fi

cd ${dir}

echo $notation started $dir/${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv $(date "+%Y/%m/%d %H:%M:%S")

## Here, if required, you can set loading of your python/conda environment
# source ~/.venv/bin/activate
# conda activate venv

python ../scripts/eval.py --train_path $train ${testStr} ${ptestStr} --device ${device} --n_jobs $njobs --gen_path ./${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv

echo $notation finished $dir/${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv $(date "+%Y/%m/%d %H:%M:%S")