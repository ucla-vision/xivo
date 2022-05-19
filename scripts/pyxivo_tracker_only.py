import argparse
import os, glob, json, re

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
    help="name of a (supported) VIO dataset [tumvi|cosyvio|alphred|xivo]")
parser.add_argument("-seq", default="room1",
    help="short tag for sequence name")
parser.add_argument("-cam_id", default=0, type=int,
    help="camera from stereo camera pair (only used for tumvi dataset)")
parser.add_argument('-cfg', default='cfg/tumvi_tracker_only_cam0.json',
    help='path to the estimator configuration')
parser.add_argument('-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
parser.add_argument('-mode', default='dumpTracker',
    help='[eval|dump|dumpCov|runOnly|dumpTracker] mode to handle the state estimates. eval: save states for evaluation; dump: save to json file for further processing')


def main(args):
    if not os.path.exists(args.dump):
        os.makedirs(args.dump)

    ########################################
    # CHOOSE SAVERS
    ########################################
    if args.mode == 'dumpTracker':
        if args.dataset == 'tumvi':
            saver = savers.TUMVITrackerDumpModeSaver(args)
        elif args.dataset == 'cosyvio':
            saver = savers.COSYVIOCovDumpModeSaver(args)
        elif args.dataset == 'xivo':
            saver = savers.XIVOTrackerDumpModeSaver(args)
        elif args.dataset == 'carla':
            saver = savers.CarlaTrackerDumpModeSaver(args)

    elif args.mode == 'runOnly':
        pass
    else:
        raise ValueError('mode=[eval|dump|dumpCov|runOnly|dumpTracker]')

    ########################################
    # LOAD DATA
    ########################################
    if args.dataset == 'tumvi':
        img_dir = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                               'mav0', 'cam{}'.format(args.cam_id), 'data')
    elif args.dataset == 'cosyvio':
        img_dir = os.path.join(args.root, 'data', args.sen, args.seq, 'frames')
    elif args.dataset in ['xivo', 'carla']:
        img_dir = os.path.join(args.root, args.seq, 'cam0', 'data')
    else:
        raise ValueError('unknown dataset argument; choose from tumvi, xivo, cosyvio, carla')

    data = []

    if args.dataset in ['tumvi', 'xivo', 'carla']:
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

    data.sort(key=lambda tup: tup[0])

    # Open the file and remove all comments 
    json_file = open(args.cfg)
    json_string = ''.join(re.sub('//.*','',line) for line in json_file)
        
    # Parse as json object
    json_data = json.loads(json_string)

    # Store the descriptor type
    descriptor_type = json_data['tracker_cfg']['descriptor']
    
    # Determine the format for writing descriptor values based on descriptor type
    descriptor_format = "%f" if descriptor_type in ["SIFT", "SURF"] else "%d"

    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    viewer_cfg = ''
    if args.use_viewer:
        if args.dataset == 'tumvi':
            viewer_cfg = os.path.join('cfg', 'viewer_tracker_only.json')
        elif args.dataset == 'xivo':
            viewer_cfg = os.path.join('cfg', 'phab_viewer_tracker_only.json')
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
        estimator = pyxivo.Estimator(args.cfg, viewer_cfg, args.seq, True)
        for i, (ts, content) in enumerate(data):
            if i > 0 and i % 1000 == 0:
                print('{:6}/{:6}'.format(i, len(data)))
            estimator.VisualMeasTrackerOnly(ts, content)
            estimator.Visualize()
            if args.mode != 'runOnly':
                saver.onVisionUpdate(estimator, datum=(ts, content), descriptor_format=descriptor_format)

    finally:
        if args.mode != 'runOnly':
            saver.onResultsReady()


if __name__ == '__main__':
    main(args=parser.parse_args())
