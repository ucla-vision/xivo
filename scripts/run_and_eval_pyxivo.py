import numpy as np
import argparse
import os, sys
from shutil import copyfile

TP_ROOT = '/home/feixh/Data/tumvi/exported/euroc/512_16'
KIF_ROOT = '/local2/Data/tumvi/exported/euroc/512_16'
double_fusion = False

parser = argparse.ArgumentParser()
parser.add_argument(
    '-root', default=KIF_ROOT, help='root directory of the tumvi dataset')
parser.add_argument(
    '-seq', default='room6', help='short tag for the seuqence name')
parser.add_argument(
    '-out_dir', default='.', help='output directory to save results')
parser.add_argument(
    '-use_viewer', default=False, action='store_true', help='visualize if set')

# parser.add_argument(
#     '-double-fusion', default=False, action='store_true', help='if ture, take average of two trajectories in the same coordinate system')
#
parser.add_argument(
    '-stdout', default=False, action='store_true', help='write to stdout instead of a benchmark file if set')
args = parser.parse_args()

if __name__ == '__main__':

    ########################################
    # BACKUP CONFIGURATION
    ########################################
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    cam_ids = [0, 1] if double_fusion else [0]

    for cam_id in cam_ids:
        cfg = 'cfg/tumvi_cam{}.json'.format(cam_id)

        copyfile(cfg, os.path.join(args.out_dir, os.path.basename(cfg)))

        ########################################
        # RUN THE FILTER
        ########################################
        cmd = 'python3 scripts/pyxivo.py \
-root {root:} \
-cfg {cfg:} \
-seq {seq:} \
-cam_id {cam_id:} \
-dump {out_dir:} \
{use_viewer:}'.format(
            root=args.root, cfg=cfg, seq=args.seq, cam_id=cam_id, out_dir=args.out_dir,
            use_viewer='-use_viewer' if args.use_viewer else '')
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)

        result_file = os.path.join(args.out_dir, 'tumvi_{}_cam{}'.format(args.seq, cam_id))
        groundtruth_file = os.path.join(args.out_dir, 'tumvi_{}_gt'.format(args.seq))
        benchmark_file = os.path.join(args.out_dir, 'tumvi_{}_bench'.format(args.seq))

        if cam_id == 0:
            os.system('echo tumvi sequence {} >> {}'.format(args.seq, benchmark_file))

        os.system('echo camera {} >> {}'.format(cam_id, benchmark_file))

        ########################################
        # COMPUTE ATE
        ########################################
        cmd = 'python3 scripts/tum_rgbd_benchmark_tools/evaluate_ate.py \
--max_difference 0.001 \
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
        cmd = 'python3 scripts/tum_rgbd_benchmark_tools/evaluate_rpe.py \
--fixed_delta \
--delta_unit s \
--delta 1 \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
            groundtruth_file=groundtruth_file,
            result_file=result_file,
            write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)

    if double_fusion:
        # fuse trajectories
        cmd = 'python3 scripts/double_fusion.py \
-root {root:} \
-working-dir {workdir:}  \
-seq {seq:}'.format( root=args.root, workdir=args.out_dir, seq=args.seq)
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)

        os.system('echo double-fusion >> {}'.format(benchmark_file))

        # evaluate the fused trajectory
        result_file = os.path.join(args.out_dir, 'tumvi_{}_fused'.format(args.seq))
        cmd = 'python3 scripts/tum_rgbd_benchmark_tools/evaluate_ate.py \
--max_difference 0.001 \
{groundtruth_file:} \
{result_file:} {write_to:}'.format(
            groundtruth_file=groundtruth_file,
            result_file=result_file,
            write_to='>> {}'.format(benchmark_file) if not args.stdout else '')
        print('*** COMMAND TO BE EXECUTED ***')
        print(cmd)
        os.system(cmd)






