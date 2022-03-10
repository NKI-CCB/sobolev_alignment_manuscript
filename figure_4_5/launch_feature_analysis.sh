#!/bin/sh

python=python

script_folder=./sobolev_alignment/
output_folder=./output/
data_folder=./
tmp_folder=/tmp
mkdir ${output_folder}
mkdir ${output_folder}/scripts/
mkdir ${output_folder}/GSEA_null/
mkdir ${tmp_folder}

# Train VAE
${python} train_VAE.py -o $output_folder -d $data_folder -n 10000000 -t $tmp_folder -j 50;
rm -r ${tmp_folder}'*'

${python} null_model_PV_sim.py -o $output_folder -d $data_folder -n 10000000 -t $tmp_folder -j 50 -i 2 -p 0;
rm -r ${tmp_folder}'*'

${python} null_model_GSEA.py -o $output_folder -d $data_folder -n 10000000 -t $tmp_folder -j 50 -p 100;
rm -r ${tmp_folder}'*'

${python} taylor_feature_attribution.py -o $output_folder -d $data_folder -n 10000000 -t $tmp_folder -j 50 -i 1;
rm -r ${tmp_folder}'*'