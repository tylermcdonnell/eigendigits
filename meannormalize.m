function [ m, A ] = meannormalize( A )
%MEANNORMALIZE Normalizes input features.
%   Computes the a mean vector mu, representing the mean 
%   value of each feature in the input matrix. Subtracts
%   this mean vector from the features of each example.

[~, K] = size(A);
m = mean(A, 2);
A = A - repmat(m, 1, K);

end

