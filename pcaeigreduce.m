function [ m, PC ] = pcaeigreduce( A )
%HW1FINDEIGENDIGITS Summary of this function goes here
%   Takes an (x by k) matrix A where x is the feature vector and k
%   is the number of examples.

% Mean normalize the input.
[m, A] = meannormalize(A);

% Compute eigenvectors of reduced covariance matrix.
% The eigenvectors are the Principal Components = PC.
[~, K] = size(A);
[PC, V] = eig(1 / (K - 1) * (A' * A));

% Expand from reduced space into full space.
PC = A * PC;

% Sort eigenvectors in descending order.
V = diag(V);
[~, I] = sort(-1*V);
PC = PC(:,I);

% Normalize output.
PC = normc(PC);

end

