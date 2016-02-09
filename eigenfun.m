function [ accuracy ] = eigenfun( X, XL, x, xl, K, T)
%EIGENFUN Summary of this function goes here
%   Detailed explanation goes here

% Number of features.
[F, ~] = size(X);

% Unique classifications.
classifications = unique(XL);

% Size of classification space.
cSize = length(classifications);

% First, let's find the example for which we have the fewest
% classifications in the training set.
classifications = unique(XL);
counts          = zeros(length(classifications), 1);
for i = 1:length(classifications)
   counts(i) = length(find(XL == classifications(i)));
end
cMin = min(counts);

% -------------------------------
% For each classification, let's construct an eigenspace.
% -------------------------------
% Eigenspaces.
Us = zeros(F, T, cSize);
% Training Data projected into eigenspaces.
Ps = zeros(T, cMin, cSize); 
% Means.
Ms = zeros(F, cSize);
for i = 1:length(classifications)
    % Find training examples with this classification.
    indices = find(XL==classifications(i));
    thisClassification = X(:,indices');
    thisClassification = thisClassification(:,1:cMin);
    
    % Build the eigenspace.
    [m, U] = pcaeig(thisClassification);
    U = normc(U(:,1:T));
    
    % Save relevant data.
    Us(:,:,i) = U;
    Ps(:,:,i) = U' * thisClassification;
    Ms(:,i)   = m;
end

% -------------------------------
% Project each test example x into all of the eigenspaces and find the
% mean distance between x and the K nearest neighbors in that eigenspace.
% Classify the example x by the lowest mean distance across eigenspaces.
% -------------------------------
correct = 0;
for i = 1:length(xl)
    i
    thisTest = x(:,i);
    
    eigenspaceDistances = zeros(cSize, 1);
    
    for j = 1:cSize        
        % Project into eigenspace.
        projection = Us(:,:,j)' * thisTest;
        indices    = knn(Ps(:,:,j), projection, K);
        neighbors  = Ps(:,indices,j);
      
        % Calculate mean distance of neighbors.
        numberOfNeighbors = length(indices);
        distances         = zeros(numberOfNeighbors, 1);
        for k = 1:numberOfNeighbors
            thisNeighbor = neighbors(:,k);
            %Euclidian distance
            distances(k) = sqrt(sum((thisNeighbor-projection).^2)); 
        end
        eigenspaceDistances(j) = mean(distances);
    end
    c = classifications(find(eigenspaceDistances==min(eigenspaceDistances)));
    if c == xl(i)
       correct = correct + 1; 
    end 
end

accuracy = double(correct) / length(xl);
end