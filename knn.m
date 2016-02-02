function [ N ] = knn( X, x, k )
%MEANNORMALIZE Finds the k nearest neighbors in X of x.
%   Given a matrix an m x n matrix X, where columns in X correspond
%   to features of a single observation and there are n observations,
%   finds the k nearest neighbors in X to x using the Euclidean Distance
%   metric.
%   
%   N -- A vector indicating the k columns (in increasing distance) in X
%        corresponding to the k nearest neighbors of x in R^(m).

% knnsearch expects features in rows.
N = knnsearch(X', x', 'k', k);

end

