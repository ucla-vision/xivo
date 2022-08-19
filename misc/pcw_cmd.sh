

#python scripts/pyxivo_pcw.py \
#    -npts 1000 \
#    -motion_type lissajous \
#    -xlim -20 20 \
#    -ylim -20 20 \
#    -zlim -20 20 \
#    -vision_dt 0.01 \
#    -noise_vision 1.0 \
#    -noise_accel 1e-4 \
#    -noise_gyro 1e-4 \
#    -cfg cfg/pcw_loops.json \
#    -use_viewer \
#    -viewer_cfg cfg/pcw_loops_viewer.json


python scripts/pyxivo_pcw.py \
    -motion_type calib_traj \
    -noise_accel 1e-4 \
    -noise_gyro 1e-4 \
    -vision_dt 0.01 \
    -noise_vision 0.5 \
    -noise_accel 1e-4 \
    -noise_gyro 1e-4 \
    -cfg cfg/pcw.json \
    -use_viewer \
    -viewer_cfg cfg/pcw_viewer.json
