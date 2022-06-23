function nq = qinv(q)
% Inverse of a normalized WXYZ quaternion `q`

nq = [ q(:,1), -q(:,2), -q(:,3), -q(:,4) ];


end