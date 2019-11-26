TUMVIROOT="/home/feixh/Data/tumvi/exported/euroc/512_16"
OUTPUT="./output/"

mkdir output

for index in {1..6}
do
  python scripts/pyxivo.py -use_viewer -root $TUMVIROOT -cfg cfg/tumvi_cam0.json -seq room$index -out_dir $OUTPUT -mode dump
  python scripts/undistort.py $OUTPUT/tumvi_room${index}_cam0 $OUTPUT/undistorted/room$index
done

