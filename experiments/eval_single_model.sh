#!/bin/bash

ARGS=$(getopt -o '' --longoptions hl:,hu:,bs:,lr:,es:,epoch:,aug:,notation:,njobs:,fold:,gamma:,samples:,temp:,seed:,dataset:,onlyNovels,cuda: -- "$@")

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
cuda=cpu
fold=
seed=0
onlyNovels=
njobs=1

samples=6000
T=1.0
batch=1000
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
    --cuda) cuda="cuda:$2"; shift 2 ;;
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
    suffix="_${fold}"
fi

prefixes=${model}${suffix}

if [[ "${epoch}" ]]; then
    epochPad=$(printf "%03d" $epoch)
    epoch=_$((10#${epochPad}))
    epochPad=_$epochPad
fi

if [ $seed != '0' ]; then
    seedStr=_seed$seed
fi

#Non metto mai augmented per il train perche' tanto e' quello x1 che devo caricare in ogni caso ...
train=../data/${dataset}_train${suffix}.tar.xz

cd ~/DeNovo/experiments/${dir}

echo $notation started $dir/${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv $(date "+%Y/%m/%d %H:%M:%S")

## modificato per il moses ... mettiamo la normale valutaizone basata sul test e test scaffold !!
# python3.11 ../scripts/eval.py --train_path $train --test_path $train --device ${cuda} --n_jobs $njobs --gen_path ./${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv

python3.11 ../scripts/eval.py --train_path $train --device ${cuda} --n_jobs $njobs --gen_path ./${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv

echo $notation finished $dir/${experiment}/${prefixes}_generated${samples}${onlyNovels}${epoch}${seedStr}_T${T}.csv $(date "+%Y/%m/%d %H:%M:%S")