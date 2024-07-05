#!/bin/bash

params="--hl 2 --hu 512 --lr 0.001 --bs 512 --es 300 "

# for fold in {3..4}; do
#     for notation in smiles selfies; do
#         bash sample_single_model.sh --notation $notation --fold ${fold} --onlyNovels --epoch 10  ${params}
#         bash eval_single_model.sh --notation $notation --fold ${fold} --njobs 10 --onlyNovels --epoch 10  ${params}
#     done
# done

for fold in {0..4}; do
    bash sample_single_model.sh --notation fragsmiles --fold ${fold} --onlyNovels --epoch 16  ${params}
    bash eval_single_model.sh --notation fragsmiles --fold ${fold} --njobs 7 --onlyNovels --epoch 16  ${params}
done
wait