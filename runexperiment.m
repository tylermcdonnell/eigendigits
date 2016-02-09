function [ a ] = runexperiment ( X, XLabels, x, xLabels, K, T, N )
%RUNEXPERIMENT Runs a PCA experiment.
%   Performs Principal Components Analysis for the given values of K, T,
%   and N on the training set X. Classifies all column vectors in x by
%   KNN and returns a single number, the classification accuracy.
%
%   X       -- [C by R] training data. C features, R examples.
%   XLabels -- [1 by R] labels for training data.
%   x       -- 2D test data. Columns are features.
%   xLabels -- Gold standard labels for test data.
%   K       -- Number of nearest neighbors to use for KNN.
%   T       -- Number of eigenvectors to use for projection.
%   N       -- Number of training examples to use from X. N <= R.

thisTrainingSet = X(:,1:N);
thisTrainingLabels = XLabels(:,1:N);

% Compute eigenmatrix.
[m, PC] = pcaeig(thisTrainingSet);

% Take only top T eigenvectors.
PC = PC(:,1:T);

% Project data into eigenspace.
[~, thisTrainingSet] = meannormalize(thisTrainingSet);
projectedData = PC' * thisTrainingSet;

correct = 0;
[~, tests] = size(x);

for i = 1:tests
   thisTest = x(:,i);
   thisLabel = xLabels(1,i);
   
   % Project into eigenspace.
   projection = PC' * (thisTest - m);
   
   % Classify.
   c = knnclassify(projectedData, thisTrainingLabels, projection, K);
   if c == thisLabel
       correct = correct + 1;
   end
end

a = double(correct) / tests;
end