# quick script to run and evaluate vio on a given dataset
import argparse
import os, sys
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-log_dir', default='log',
        help='directory to save logs')
parser.add_argument('-root', default='/home/feixh/Data/tumvi/exported/euroc/512_16',
        help='root directory of the tumvi dataset')
parser.add_argument('-dataset', default='tumvi',
        help='type of dataset, tumvi|euroc')
parser.add_argument('-seq', default='room6',
        help='short tag for the seuqence name')
parser.add_argument('-cam_id', default=0, type=int,
        help='specify which camera to use')
parser.add_argument('-resolution', default=0.001, type=float,
        help='resolution to pair poses and compute RPE')
parser.add_argument('-out_dir', default='.',
        help='directory to hold results')

args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    result_file = os.path.join(args.out_dir, '{}_{}'.format(args.dataset, args.seq))
    benchmark_file = os.path.join(args.out_dir, 'benchmark')
    copyfile('cfg/estimator.json', os.path.join(args.out_dir, 'estimator.json'))

    # run
    cmd = 'GLOG_logtostderr=0 GLOG_log_dir={log_dir:} GLOG_v=0 \
msckf/bin/st_vio \
-cfg cfg/vio.json \
-root {root:} \
-dataset {dataset:} \
-seq {seq:} \
-cam_id {cam_id:} \
-out {out:}'.format(
            log_dir=args.log_dir,
            root=args.root,
            dataset=args.dataset,
            seq=args.seq,
            cam_id=args.cam_id,
            out=result_file)
    print('***Executing the following command***\n{}'.format(cmd))
    os.system(cmd)

    # evaluation
    os.system('echo {} >> {}'.format(args.seq, benchmark_file))
    cmd = 'msckf/bin/eval \
-root {root:} \
-dataset {dataset:} \
-seq {seq:} \
-cam_id {cam_id:} \
-result {result:} \
-resolution {resolution:} >> {benchmark_file:}'.format(
            log_dir=args.log_dir,
            root=args.root,
            dataset=args.dataset,
            seq=args.seq,
            cam_id=args.cam_id,
            result=result_file,
            resolution=args.resolution,
            benchmark_file=benchmark_file)
    print('***Executing the following command***\n{}'.format(cmd))
    os.system(cmd)
