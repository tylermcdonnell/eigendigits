function [ m, V ] = pcasvd ( A )

[X, K] = size(A);

% Apply mean normalization to input.
A_mean = mean(A, 2);
A = double(A) - repmat(A_mean, 1, K);

% Calculate the covariance matrix.
A_covariance = 1 / (K - 1) * (A * A');

% Compute the eigenvectors, or principle components.
[U, S, PC] = svd(A_covariance);

%m = PC' * A;

m = A_mean;
V = PC;

end
