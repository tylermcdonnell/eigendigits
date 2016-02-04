function [ a ] = runbaseline( X, XLabels, x, xLabels, K, N )
%RUNBASELINE KNN classification in full data space.
%   X       -- [C by R] full data.
%   XLabels -- [1 by R] labels for training data.
%   x       -- 2D test data. Columns are features.
%   xLabels -- Labels for test data.
%   K       -- Number of nearest neighbors for KNN.
%   N       -- Number of training examples to use.

correct = 0;
for i = 1:length(xLabels)
    i
    c = knnclassify(X(:,1:N), XLabels(:,1:N), x(:,i), K);
    if c == xLabels(:,i)
        correct = correct + 1;
    end
end

a = double(correct) / length(xLabels);
end

