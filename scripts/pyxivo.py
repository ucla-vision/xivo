import argparse
import os, glob

import sys
sys.path.insert(0, 'lib')
import pyxivo
import savers



parser = argparse.ArgumentParser()
parser.add_argument("-root", default="/media/data1/Data/tumvi/exported/euroc/512_16",
    help="location of VIO dataset")
parser.add_argument("-dump", default=".",
    help="location of xivo's output data from a dataset")
parser.add_argument("-dataset", default="tumvi",
    help="name of a (supported) VIO dataset [tumvi|cosyvio|alphred|xivo|void]")
parser.add_argument("-seq", default="room1",
    help="short tag for sequence name")
parser.add_argument("-cam_id", default=0, type=int,
    help="camera from stereo camera pair (only used for tumvi dataset)")
parser.add_argument('-cfg', default='cfg/tumvi_cam0.json',
    help='path to the estimator configuration')
parser.add_argument('-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
parser.add_argument('-mode', default='eval',
    help='[eval|dump|dumpCov|runOnly] mode to handle the state estimates. eval: save states for evaluation; dump: save to json file for further processing')
parser.add_argument(
    '-save_full_cov', default=False, action='store_true',
    help='save the entire covariance matrix, not just that of the motion state, if set')


def main(args):
    if not os.path.exists(args.dump):
        os.makedirs(args.dump)

    ########################################
    # CHOOSE SAVERS
    ########################################
    if args.mode == 'eval':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIEvalModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIOEvalModeSaver(args)
        elif args.dataset == "xivo":
            saver = savers.XIVOEvalModeSaver(args)
        elif args.dataset == "void":
            saver = savers.VOIDEvalModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaEvalModeSaver(args)
    elif args.mode == 'dump':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIDumpModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIODumpModeSaver(args)
        elif args.dataset == "xivo":
            saver = savers.XIVODumpModeSaver(args)
        elif args.dataset == "void":
            saver = savers.VOIDDumpModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaDumpModeSaver(args)
    elif args.mode == 'dumpCov':
        if args.dataset == 'tumvi':
            saver = savers.TUMVICovDumpModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIOCovDumpModeSaver(args)
        elif args.dataset == "xivo":
            saver = savers.XIVOCovDumpModeSaver(args)
        elif args.dataset == "void":
            saver = savers.VOIDCovDumpModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaCovDumpModeSaver(args)
    elif args.mode == 'runOnly':
        pass
    else:
        raise ValueError('mode=[eval|dump|dumpCov|runOnly]')

    ########################################
    # LOAD DATA
    ########################################
    if args.dataset == 'tumvi':
        img_dir = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                               'mav0', 'cam{}'.format(args.cam_id), 'data')

        imu_path = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                'mav0', 'imu0', 'data.csv')
    elif args.dataset == 'cosyvio':
        img_dir = os.path.join(args.root, 'data', args.sen, args.seq, 'frames')
        imu_path = os.path.join(args.root, 'data', args.sen, args.seq, 'data.csv')
    elif args.dataset in ['xivo', 'carla', 'void']:
        img_dir = os.path.join(args.root, args.seq, 'cam0', 'data')
        imu_path = os.path.join(args.root, args.seq, 'imu0', 'data.csv')
    else:
        raise ValueError('unknown dataset argument; choose from tumvi, xivo, cosyvio, carla')

    data = []

    if args.dataset in ['tumvi', 'xivo', 'carla', 'void']:
        for p in glob.glob(os.path.join(img_dir, '*.png')):
            ts = int(os.path.basename(p)[:-4])
            data.append((ts, p))
    elif args.dataset == 'cosyvio':
        img_filelist = os.path.join(img_dir, 'data.csv')
        with open(img_filelist, 'r') as fid:
            for l in fid.readlines():
                if l[0].isdigit():
                    larr = l.strip().split(',')
                    ts = int(larr[0])
                    png_file = os.path.join(img_dir, larr[1])
                    data.append((ts,png_file))

    with open(imu_path, 'r') as fid:
        for l in fid.readlines():
            if l[0].isdigit():
                v = l.strip().split(',')
                ts = int(v[0])
                w = [float(x) for x in v[1:4]]
                t = [float(x) for x in v[4:]]
                data.append((ts, (w, t)))

    data.sort(key=lambda tup: tup[0])

    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    viewer_cfg = ''
    if args.use_viewer:
        if args.dataset == 'tumvi':
            viewer_cfg = os.path.join('cfg', 'viewer.json')
        elif args.dataset == "xivo":
            viewer_cfg = os.path.join('cfg', 'phab_viewer.json')
        elif args.dataset == 'void':
            viewer_cfg = os.path.join('cfg', 'void_viewer.json')
        elif args.dataset == 'cosyvio':
            if args.sen == 'tango_top':
                viewer_cfg = os.path.join('cfg', 'phab_viewer.json')
            elif args.sen == 'tango_bottom':
                viewer_cfg = os.path.join('cfg', 'cosyvio_tango_bottom_viewer.json')
        elif args.dataset == 'carla':
            viewer_cfg = os.path.join('cfg', 'carla_viewer.json')

    #########################################
    # RUN ESTIMATOR AND SAVE DATA
    #########################################
    # this is wrapped in a try/finally block so that data will save even when
    # we hit an exception (namely, KeyboardInterrupt)
    try:
        estimator = pyxivo.Estimator(args.cfg, viewer_cfg, args.seq, False)
        for i, (ts, content) in enumerate(data):
            if i > 0 and i % 1000 == 0:
                print('{:6}/{:6}'.format(i, len(data)))
            if isinstance(content, tuple):
                gyro, accel = content
                estimator.InertialMeas(ts, gyro[0], gyro[1], gyro[2], accel[0],
                                    accel[1], accel[2])
            else:
                estimator.VisualMeas(ts, content)
                if estimator.UsingLoopClosure():
                    estimator.CloseLoop()
                estimator.Visualize()
                if (args.mode != 'runOnly') and (estimator.VisionInitialized()):
                    saver.onVisionUpdate(estimator, datum=(ts, content))

    finally:
        if args.mode != 'runOnly':
            saver.onResultsReady()


if __name__ == '__main__':
    main(args=parser.parse_args())
