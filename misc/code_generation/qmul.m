function q_out = qmul(q1, q2)
% Quaternion multiplication of two normalized quaternions

w1 = q1(1);
x1 = q1(2);
y1 = q1(3);
z1 = q1(4);

w2 = q2(1);
x2 = q2(2);
y2 = q2(3);
z2 = q2(4);

w_out = w1*w2 - x1*x2 - y1*y2 - z1*z2;
x_out = w1*x2 + x1*w2 + y1*z2 - z1*y2;
y_out = w1*y2 - x1*z2 + y1*w2 + z1*x2;
z_out = w1*z2 + x1*y2 - y1*x2 + z1*w2;

q_out = [ w_out, x_out, y_out, z_out ];


end