function qout = slerp(q0, q1, t)
% Spherical linear interpolation for moving from quaternion `q0` to
% quaternion `q1`. `t` is in `[0,1]`.

q1_q0inv = qmul(q1, qinv(q0));
dq = qpow(q1_q0inv, t);

qout = qmul(dq, q0);


end