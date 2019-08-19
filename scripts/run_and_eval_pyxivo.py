import numpy as np
import argparse
import os, sys
from shutil import copyfile

TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'

parser = argparse.ArgumentParser()
parser.add_argument(
    '-root', default=KIF_ROOT, help='root directory of the tumvi dataset')
parser.add_argument(
    '-cfg', default='cfg/estimator.json', help='path to the estimator configuration')
parser.add_argument(
    '-seq', default='room6', help='short tag for the seuqence name')
parser.add_argument(
    '-cam_id', default=0, type=int, help='specify which camera to use')
parser.add_argument(
    '-out_dir', default='.', help='output directory to save results')
parser.add_argument(
    '-use_viewer', default=False, action='store_true', help='visualize if set')
parser.add_argument(
    '-stdout', default=False, action='store_true', help='write to stdout instead of a benchmark file if set')
args = parser.parse_args()

if __name__ == '__main__':

    ########################################
    # BACKUP CONFIGURATION
    ########################################
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    copyfile(args.cfg, os.path.join(args.out_dir, os.path.basename(args.cfg)))

    ########################################
    # RUN THE FILTER
    ########################################
    cmd = 'python scripts/pyxivo.py \
-root {root:} \
-cfg {cfg:} \
-seq {seq:} \
-cam_id {cam_id:} \
-out_dir {out_dir:} \
{use_viewer:}'.format(
        root=args.root, cfg=args.cfg, seq=args.seq, cam_id=args.cam_id, out_dir=args.out_dir,
        use_viewer='-use_viewer' if args.use_viewer else '')
    print('*** COMMAND TO BE EXECUTED ***')
    print(cmd)
    os.system(cmd)

    result_file = os.path.join(args.out_dir, 'tumvi_{}'.format(args.seq))
    groundtruth_file = os.path.join(args.out_dir, 'tumvi_{}_gt'.format(
        args.seq))
    benchmark_file = os.path.join(args.out_dir, 'tumvi_{}_bench'.format(args.seq))

    os.system('echo {} >> {}'.format(args.seq, benchmark_file))

    ########################################
    # COMPUTE ATE
    ########################################
    cmd = 'python scripts/tum_rgbd_benchmark_tools/evaluate_ate.py \
--max_difference 0.001 \
--verbose \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
        groundtruth_file=groundtruth_file,
        result_file=result_file,
        write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
    print('*** COMMAND TO BE EXECUTED ***')
    print(cmd)
    os.system(cmd)

    ########################################
    # COMPUTE RPE
    ########################################
    cmd = 'python scripts/tum_rgbd_benchmark_tools/evaluate_rpe.py \
--fixed_delta \
--delta_unit s \
--delta 1 \
--verbose \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
        groundtruth_file=groundtruth_file,
        result_file=result_file,
        write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
    print('*** COMMAND TO BE EXECUTED ***')
    print(cmd)
    os.system(cmd)
