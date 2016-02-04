function [ m, PC ] = pcasvd ( A )

[~, K] = size(A);

% Mean normalize the input.
[m, A] = meannormalize(A);

% Calculate the covariance matrix.
Acov = 1 / (K - 1) * (A * A');

% Compute the eigenvectors, or principle components.
[~, ~, PC] = svd(Acov);

% Normalize output.
PC = normc(PC);

end
