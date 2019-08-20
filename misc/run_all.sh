#!/bin/bash
# TMPDIR=$(mktemp -d -p .)
TUMVIROOT="/local2/Data/tumvi/exported/euroc/512_16"

#python scripts/quick_run.py -out_dir $TMPDIR -root $TUMVIROOT -seq room1
#python scripts/quick_run.py -out_dir $TMPDIR -root $TUMVIROOT -seq room2
#python scripts/quick_run.py -out_dir $TMPDIR -root $TUMVIROOT -seq room3
#python scripts/quick_run.py -out_dir $TMPDIR -root $TUMVIROOT -seq room4
#python scripts/quick_run.py -out_dir $TMPDIR -root $TUMVIROOT -seq room5
#python scripts/quick_run.py -out_dir $TMPDIR -root $TUMVIROOT -seq room6

mkdir $CFG
python scripts/run_and_eval_pyxivo.py -out_dir $CFG -root $TUMVIROOT -seq room1 -cfg test_cfg/$CFG.json ${1} &
python scripts/run_and_eval_pyxivo.py -out_dir $CFG -root $TUMVIROOT -seq room2 -cfg test_cfg/$CFG.json ${1} &
python scripts/run_and_eval_pyxivo.py -out_dir $CFG -root $TUMVIROOT -seq room3 -cfg test_cfg/$CFG.json ${1} &
python scripts/run_and_eval_pyxivo.py -out_dir $CFG -root $TUMVIROOT -seq room4 -cfg test_cfg/$CFG.json ${1} &
python scripts/run_and_eval_pyxivo.py -out_dir $CFG -root $TUMVIROOT -seq room5 -cfg test_cfg/$CFG.json ${1} &
python scripts/run_and_eval_pyxivo.py -out_dir $CFG -root $TUMVIROOT -seq room6 -cfg test_cfg/$CFG.json ${1} &
