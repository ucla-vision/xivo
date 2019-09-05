#!/bin/bash
# Evaluation on TUM-VI room sequences.
# Usage:
# In project root directory, execute the following in terminal:
# OUTPUT=/output/directory/of/your/choice misc/run_all.sh

# TMPDIR=$(mktemp -d -p .)
# TUMVIROOT="/local2/Data/tumvi/exported/euroc/512_16"
TUMVIROOT="/home/feixh/Data/tumvi/exported/euroc/512_16"

mkdir $OUTPUT
python scripts/run_and_eval_pyxivo.py -out_dir $OUTPUT -root $TUMVIROOT -seq room1 ${1}
python scripts/run_and_eval_pyxivo.py -out_dir $OUTPUT -root $TUMVIROOT -seq room2 ${1}
python scripts/run_and_eval_pyxivo.py -out_dir $OUTPUT -root $TUMVIROOT -seq room3 ${1}
python scripts/run_and_eval_pyxivo.py -out_dir $OUTPUT -root $TUMVIROOT -seq room4 ${1}
python scripts/run_and_eval_pyxivo.py -out_dir $OUTPUT -root $TUMVIROOT -seq room5 ${1}
python scripts/run_and_eval_pyxivo.py -out_dir $OUTPUT -root $TUMVIROOT -seq room6 ${1}
