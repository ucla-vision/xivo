import sys, os, glob, shutil

if __name__ == '__main__':
    bagpath = sys.argv[1]
    outdir = sys.argv[2]
    os.system('kalibr_bagextractor --image-topics /cam0/image_raw --imu-topics /imu0 --output-folder {outdir:} --bag {bagpath:}'.format(outdir=outdir, bagpath=bagpath))

    imu_dir = os.path.join(outdir, 'imu0')
    os.makedirs(imu_dir)
    shutil.move(os.path.join(outdir, 'imu0.csv'), os.path.join(imu_dir, 'data.csv'))

    img_dir = os.path.join(outdir, 'cam0')
    imgs = glob.glob(os.path.join(img_dir, '*png'))
    imgs.sort()

    img_data_dir = os.path.join(img_dir, 'data')
    os.makedirs(img_data_dir)

    csv = []
    for each in imgs:
        basename = os.path.basename(each)
        shutil.move(each, os.path.join(img_data_dir, basename))
        csv.append(basename[:-4])

    with open(os.path.join(img_dir, 'data.csv'), 'w') as fid:
        fid.write('timestamp (ns), filename\n')
        for each in csv:
            fid.write('{},{}.png\n'.format(each, each))
