function V = cx(v)
%cx construct 3x3 skew-symmetric matrix from vector of length 3

V = [ ...
        0, -v(3),  v(2); ...
     v(3),     0, -v(1); ...
    -v(2),  v(1),     0 ];

end

