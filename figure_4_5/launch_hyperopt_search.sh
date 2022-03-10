#!/bin/sh
output_folder=./output/
data_folder=.
python=python

${python} kim_hyperopt_search.py -o $output_folder -d $data_folder -c 0.1 -m 100
${python} kinker_hyperopt_search.py -o $output_folder -d $data_folder -c 0.1 -m 100
${python} combined_hyperopt_search.py -o $output_folder -d $data_folder -c 0.1 -m 100