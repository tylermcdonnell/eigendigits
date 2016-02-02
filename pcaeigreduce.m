function [ m, V ] = pcaeigreduce( A )
%HW1FINDEIGENDIGITS Summary of this function goes here
%   Takes an (x by k) matrix A where x is the feature vector and k
%   is the number of examples.

[X, K] = size(A);

% Mean normalize the input.
[m, A] = meannormalize(A);

% Compute the eigenvectors of the covariance matrix.
[U, v] = eig(A'*A);

% Extend to the larger space.
V = A * U;


end

