function [ m, V ] = pcaeig( A )
%HW1FINDEIGENDIGITS Summary of this function goes here
%   Takes an (x by k) matrix A where x is the feature vector and k
%   is the number of examples.

[X, K] = size(A);

% Apply mean normalization to input.
A_mean = mean(A, 2);
A = double(A) - repmat(A_mean, 1, K);

% Calculate the covariance matrix.
A_covariance = 1 / (K - 1) * A * A';

% Compute the eigenvectors, or principle components.
[PC, V] = eig(A_covariance);

% Sort the eigenvectors in decreasing order of eigenvalue.
V = diag(V);
[B, I] = sort(-1*V); % Descending order.
V = V(I);
PC = PC(:,I);

% Mean value of examples.
m = A_mean;
% Normalize eigenvectors for output.
V = PC; 

end

