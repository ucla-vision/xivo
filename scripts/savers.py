import numpy as np
import os
import json
from transforms3d.quaternions import mat2quat


class BaseSaver:
    def __init__(self, args):
        self.results = []
        self.resultsPath = os.path.join(args.out_dir, 'tumvi_{}_cam{}'.format(args.seq, args.cam_id))
    def onVisionUpdate(self, estimator, datum):
        pass
    def onResultsReady(self):
        pass


class EvalModeSaver(BaseSaver):
    """ Callback functions used in eval mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)
        # parse mocap and save gt in desired format
        mocapPath = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                  'mav0', 'mocap0', 'data.csv')
        groundtruthPath = os.path.join(args.out_dir, 'tumvi_{}_gt'.format(args.seq))
        self.saveMocapAs(mocapPath, groundtruthPath)

    def onVisionUpdate(self, estimator, datum):
        now = estimator.now()
        gsb = np.array(estimator.gsb())
        Tsb = gsb[:, 3]

        # print gsb[:3, :3]
        try:
            q = mat2quat(gsb[:3, :3])  # [w, x, y, z]
            # format compatible with tumvi rgbd benchmark scripts
            self.results.append(
                [now * 1e-9, Tsb[0], Tsb[1], Tsb[2], q[1], q[2], q[3], q[0]])
        except np.linalg.linalg.LinAlgError:
            pass

    def onResultsReady(self):
        np.savetxt(
            self.resultsPath,
            self.results,
            fmt='%f %f %f %f %f %f %f %f')

    def saveMocapAs(self, mocapPath, groundtruthPath):
        gt = []
        with open(mocapPath, 'r') as fid:
            for l in fid.readlines():
                if l[0] != '#':
                    v = l.strip().split(',')
                    ts = int(v[0])
                    t = [float(x) for x in v[1:4]]
                    q = [float(x) for x in v[4:]]  # [w, x, y, z]
                    gt.append(
                        [ts * 1e-9, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

        np.savetxt(
            groundtruthPath,
            gt,
            fmt='%f %f %f %f %f %f %f %f')


class DumpModeSaver(BaseSaver):
    """ Callback functions used by dump mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum):
        ts, content = datum
        now = estimator.now()
        g = np.array(estimator.gsc())
        T = g[:, 3]

        if np.linalg.norm(T) > 0:
            try:
                q = mat2quat(g[:3, :3])  # [w, x, y, z]
                # format compatible with tumvi rgbd benchmark scripts
                entry = dict()
                entry['ImagePath'] = str(content)
                entry['Timestamp'] = ts
                entry['TranslationXYZ'] = [T[0], T[1], T[2]]
                entry['QuaternionWXYZ'] = [q[0], q[1], q[2], q[3]]
                self.results.append(entry)

                with open(self.resultsPath, 'w') as fid:
                    json.dump(self.results, fid, indent=2)
            except np.linalg.linalg.LinAlgError:
                pass

    def onResultsReady(self):
        with open(self.resultsPath, 'w') as fid:
            json.dump(self.results, fid, indent=2)
