function qp = qpow(q,t)
% Raise a normalized WXYZ quaternion `q` to a scalar power `t`.

w = q(1);
v = q(2:4);

phi = acos(w);
nhat = v / norm(v);

qp = [ cos(t*phi), nhat*sin(t*phi) ];

end