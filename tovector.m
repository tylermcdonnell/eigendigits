function [ V ] = tovector( I )
%TOVECTOR Converts a 2D M x N image into a 1D column vector length M x N.

[M, N] = size(I);
V = reshape(I, M*N, 1);

end

