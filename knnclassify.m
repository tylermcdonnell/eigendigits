function [ c ] = knnclassify( X, X_l, x, k )
%KNNCLASSIFY Classifies an observation x by the majority of its neighbors.
%   Given an m x n matrix X, where m is the dimensionality of the features
%   and n is the number of examples, and an 1 x n vector X_l, containing
%   labels for each of the examples in X, and a new observation x, an
%   m by 1 vector of features, classifies x according to the majority of 
%   its k nearest neighbors. Returns the classified value.
%   
%   c = Classification label of x.

% Find the nearest neighbors.
nearestNeighbors = knn(X, x, k);

% Find the most frequent label among nearest neighbors.
labels = zeros(1, k);
for n = 1:k
    labels(1,n) = X_l(nearestNeighbors(n));
    
c = mode(labels);

end

