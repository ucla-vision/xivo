python scripts/pyxivo_pcw.py \
    -npts 1000 \
    -motion_type lissajous \
    -xlim -20 20 \
    -ylim -20 20 \
    -zlim -20 20 \
    -init_Vsb 0 0.7 8 \
    -vision_dt 0.01 \
    -noise_accel 1e-9 \
    -noise_gyro 1e-9 \
    -cfg cfg/pcw.json \
    -use_viewer


# trefoil init Vsb: 0 10 3
# lissajous init Vsb: 0 0.7 8