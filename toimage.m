function [ I ] = toimage( V )
%TOIMAGE Converts N length column vector into an sqrt(N) x sqrt(N) image.

[N, ~] = size(V);
I = reshape(V, sqrt(N), sqrt(N));

end

