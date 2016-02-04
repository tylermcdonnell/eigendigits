function [ m, PC ] = pcaeig( A )
%HW1FINDEIGENDIGITS Principal Components Analysis using eigenvectors.
%   Takes A, an (X by K) matrix, where X = [ dimensionality of features ]
%   and K = [ number of examples ] and computes the principal components,
%   or K eigenvectors of A. The result is an (X by K) eigenmatrix.

[~, K] = size(A);

% Mean Normalize the input.
[m, A] = meannormalize(A);

% Calculate the covariance matrix.
Acov = 1 / (K - 1) * (A * A');

% Compute the eigenvectors, or principle components.
[PC, V] = eig(Acov);

% Sort the eigenvectors in decreasing order of eigenvalue.
V = diag(V);
[~, I] = sort(-1*V); % Descending order.
PC = PC(:,I);

% Normalize resultant matrix.
PC = normc(PC);

% If we are also interested in the eigenvalues, we could sort using this.
%V = V(I);

end

