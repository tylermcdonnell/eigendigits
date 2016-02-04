% This script is an introduction to Principal Components Analysis by
% way of analysis and classification of handwritten digits.
% 
% I evaluate and compare several variations of PCA that may offer benefits
% in different situations and compare them to baseline classification in
% the non-compressed, original data space. 
%
% Standard PCA. 
%    1. Mean Normalize input. 
%    2. Calculate the covariance matrix of the input, AA'.
%    3. Compute the eigenvectors of the covariance matrix.
%    4. Sort the eigenvectors in descending order.
%    5. Normalize the final eigenvectors.
%
% Modified PCA
% A slight modification of standard PCA useful when the dimensionality of
% the input data is much larger than the number of training examples (and
% computationally intractable. Here, we modify (2) and insert a new (4).
%    2. Calculate the reduced covariance matrix of the input, A'A.
%       Note: dimensionality of covariance matrix is N X N, where N is the
%       number of training examples. In standard PCA, the dimensionality of
%       the covariance matrix is M x M, where M is the dimensionality of
%       the input data.
%    4. Multiply original input matrix by reduced eigenmatrix to yield the
%       eigenvectors in the full data space.

% -------------------------------
% Startup
% -------------------------------
disp('Loading test and training data...');
load digits.mat;
disp('Done.');

% -------------------------------
% Data Preprocessing
% -------------------------------
% Reshape data sets to be 2D X by K matrices, where X is the length
% of the features, in this case 28x28=784 pixels, and K is the number
% of examples in the data set.
features    = 784;
trainImages = reshape(trainImages, features, 60000);
testImages  = reshape(testImages, features, 10000);

% For consistent computations throughout.
trainImages = double(trainImages);
testImages  = double(testImages);

% % -------------------------------
% % Example PCA Analysis #1
% % -------------------------------
% % Here I apply standard and modified PCA for different values of N.
% %    N : number of training examples
% disp('About to begin PCA Example #1. Press Enter to continue');
% pause;
% 
% [~, numberOfTestImages] = size(testImages);
% exampleImage = testImages(:,1);
% 
% % Prepare data structures.
% testValues = [ 2 50 500 2500 ];
% standardReconstructions = zeros(features, length(testValues));
% modifiedReconstructions = zeros(features, length(testValues));
% 
% for i = 1:length(testValues)
%    N = testValues(i);
%    thisTrainingSet = trainImages(:,1:N);
%    
%    % Compute eigenmatrix.
%    [mStandard, PCStandard] = pcaeig(thisTrainingSet);
%    % Project into eigenspace.
%    standardProjection = PCStandard' * (exampleImage - mStandard);
%    % Reconstruct.
%    standardReconstructions(:,i) = PCStandard * standardProjection;
% 
%    % Compute eigenmatrix.
%    [mModified, PCModified] = pcaeigreduce(thisTrainingSet);
%    % Project into eigenspace.
%    modifiedProjection = PCModified' * (exampleImage - mModified);
%    % Reconstruct.
%    modifiedReconstructions(:,i) = PCModified * modifiedProjection;
% end
% displayimages([ standardReconstructions modifiedReconstructions ]);
% 
% % -------------------------------
% % Example PCA Analysis #2
% % -------------------------------
% % Here I apply standard and modified PCA for different values of T.
% %    T : number of eigenvectors
% disp('About to begin PCA Example #2. Press Enter to continue.');
% pause;
% 
% % Prepare data structures.
% testValues = [2 50 500 784];
% numberOfTrainingImages = 2500;
% thisTrainingSet = trainImages(:,1:numberOfTrainingImages);
% standardReconstructions = zeros(features, length(testValues));
% modifiedReconstructions = zeros(features, length(testValues));
% 
% for i = 1:length(testValues)
%    T = testValues(i);
%     
%    [mStandard, PCStandard] = pcaeig(thisTrainingSet);
%    % Take top T eigenvectors.
%    PCStandard = PCStandard(:,1:T);
%    % Project into eigenspace.
%    standardProjection = PCStandard' * (exampleImage - mStandard);
%    % Reconstruct.
%    standardReconstructions(:,i) = PCStandard * standardProjection;
% 
%    [mModified, PCModified] = pcaeigreduce(thisTrainingSet);
%    % Take top T eigenvectors.
%    PCModified = PCModified(:,1:T);
%    % Project into eigenspace.
%    modifiedProjection = PCModified' * (exampleImage - mModified);
%    % Reconstruct.
%    modifiedReconstructions(:,i) = PCModified * modifiedProjection;
% end
% displayimages([ standardReconstructions modifiedReconstructions ]);

% -------------------------------
% PCA Experiments
% -------------------------------
% For the following experiments, I will classify test images using KNN
% in the eigenspace using standard and modified PCA. I will compare these
% to a baseline KNN classification in the original data space. There are
% three variables at play here, and in the following three experiments, I
% will vary one variable at a time to see its effect on classification
% accuracy.
%    K : number of nearest neighbors for KNN
%    N : number of training examples
%    T : number of eigenvectors

easyTest = testImages(:,1:5000);
easyLabels = testLabels(:,1:5000);
[~,easyTestLength] = size(easyTest);
hardTest = testImages(:,9500:10000);
hardLabels = testLabels(:,1:500);
[~,hardTestLength] = size(hardTest);

%runexperiment(trainImages, trainLabels, easyTest, easyLabels, 9, 20, 5000)
runbaseline(trainImages, trainLabels, easyTest, easyLabels, 9, 5000)


% % -------------------------------
% % PCA Experiment #1 - K
% % -------------------------------
% % Here I experiment with K's impact on classification accuracy.
% disp('About to begin PCA Experiment #1. Press Enter to continue');
% pause;
% 
% % Number of training examples.
% N = 2500;
% thisTrainingSet = trainImages(:,1:N);
% thisTrainingLabels = trainLabels(:,1:N);
% 
% % Range of K we will be testing.
% testRange = 1:10;
% 
% % Standard PCA.
% [mStandard, PCStandard] = pcaeig(thisTrainingSet);
% % Project data.
% standardDataProjection = PCStandard' * thisTrainingSet;
% 
% % Modified PCA.
% [mModified, PCModified] = pcaeig(thisTrainingSet);
% % Project data.
% modifiedDataProjection = PCStandard' * thisTrainingSet;
% 
% % Statistics we will compute.
% standardEasy = zeros(1,length(testRange));
% modifiedEasy = zeros(1,length(testRange));
% originalEasy = zeros(1,length(testRange));
% 
% for K = testRange
%     sCorrect = 0;
%     mCorrect = 0;
%     oCorrect = 0;
%     for i = 1:easyTestLength
%         % Test sample.
%         [K, i]
%         
%         x = easyTest(:,i);
%         xLabel = easyTestLabels(1,i);
%         
%         % KNN in original data space.
%         c = knnclassify(thisTrainingSet, thisTrainingLabels, x, K);
%         if c == xLabel
%            oCorrect = oCorrect + 1; 
%         end
%         
%         % Standard PCA.
%         projection = PCStandard' * x;
%         c = knnclassify(standardDataProjection, thisTrainingLabels, projection, K);
%         if c == xLabel
%             sCorrect = sCorrect + 1;
%         end
%         
%         % Modified PCA.
%         projection = PCModified' * x;
%         c = knnclassify(modifiedDataProjection, thisTrainingLabels, projection, K);
%         if c == xLabel
%            mCorrect = mCorrect + 1; 
%         end
%     end
%     standardEasy(1,K) = double(sCorrect) / easyTestLength;
%     modifiedEasy(1,K) = double(mCorrect) / easyTestLength;
%     originalEasy(1,K) = double(oCorrect) / easyTestLength;
% end
% plot(testRange, standardEasy, testRange, originalEasy);
% legend('Base', 'PCA');
% xlabel('K : # neighbors');
% ylabel('Accuracy');
% 
% % -------------------------------
% % PCA Experiment #2 - N
% % -------------------------------
% % Here I experiment with N's impact on classification accuracy.
% disp('About to begin PCA Experiment #1. Press Enter to continue');
% pause;
% 
% % Hold K for KNN constant.
% K = 6;
% 
% testRange = [ 25 50 500 2500 5000];
% 
% % Statistics we will compute.
% standardEasy = zeros(1,length(testRange));
% modifiedEasy = zeros(1,length(testRange));
% originalEasy = zeros(1,length(testRange));
% 
% for count = 1:length(testRange)
%     N = testRange(count);
%     
%     % Generate training set.
%     thisTrainingSet = trainImages(:,1:N);
%     thisTrainingLabels = trainLabels(:,1:N);
%     
%     % Standard PCA.
%     [mStandard, PCStandard] = pcaeig(thisTrainingSet);
%     % Project data.
%     standardDataProjection = PCStandard' * thisTrainingSet;
% 
%     % Modified PCA.
%     [mModified, PCModified] = pcaeig(thisTrainingSet);
%     % Project data.
%     modifiedDataProjection = PCStandard' * thisTrainingSet;
% 
%     sCorrect = 0;
%     mCorrect = 0;
%     oCorrect = 0;
%     for i = 1:easyTestLength
%         % Test sample.
%         [N, i]
%         
%         x = easyTest(:,i);
%         xLabel = easyTestLabels(1,i);
%         
%         % KNN in original data space.
%         c = knnclassify(thisTrainingSet, thisTrainingLabels, x, K);
%         if c == xLabel
%            oCorrect = oCorrect + 1; 
%         end
%         
%         % Standard PCA.
%         projection = PCStandard' * x;
%         c = knnclassify(standardDataProjection, thisTrainingLabels, projection, K);
%         if c == xLabel
%             sCorrect = sCorrect + 1;
%         end
%         
%         % Modified PCA.
%         projection = PCModified' * x;
%         c = knnclassify(modifiedDataProjection, thisTrainingLabels, projection, K);
%         if c == xLabel
%            mCorrect = mCorrect + 1; 
%         end
%     end
%     standardEasy(1,count) = double(sCorrect) / easyTestLength;
%     modifiedEasy(1,count) = double(mCorrect) / easyTestLength;
%     originalEasy(1,count) = double(oCorrect) / easyTestLength;
% end
% plot(testRange, standardEasy, testRange, originalEasy);
% legend('Base', 'PCA');
% xlabel('N : # training examples');
% ylabel('Accuracy');
% 
% % -------------------------------
% % PCA Experiment #3 - T
% % -------------------------------
% % Here I experiment with N's impact on classification accuracy.
% disp('About to begin PCA Experiment #1. Press Enter to continue');
% pause;
% 
% % Hold K and N constant.
% K = 6;
% N = 5000;
% 
% testRange = [ 2 5 25 50 100 250 500];
% 
% % Statistics we will compute.
% standardEasy = zeros(1,length(testRange));
% modifiedEasy = zeros(1,length(testRange));
% originalEasy = zeros(1,length(testRange));
% 
% % Generate training set.
% thisTrainingSet = trainImages(:,1:N);
% thisTrainingLabels = trainLabels(:,1:N);
% 
% for count = 1:length(testRange)
%     T = testRange(count);
%    
%     % Standard PCA.
%     [mStandard, PCStandard] = pcaeig(thisTrainingSet);
%     % Only use top T eigenvectors.
%     PCStandard = PCStandard(:,1:T);
%     % Project data.
%     standardDataProjection = PCStandard' * thisTrainingSet;
% 
%     % Modified PCA.
%     [mModified, PCModified] = pcaeig(thisTrainingSet);
%     % Only use top T eigenvectors.
%     PCModified = PCModified(:,1:T);
%     % Project data.
%     modifiedDataProjection = PCStandard' * thisTrainingSet;
% 
%     sCorrect = 0;
%     mCorrect = 0;
%     oCorrect = 0;
%     for i = 1:easyTestLength
%         % Test sample.
%         [T, i]
%         x = easyTest(:,i);
%         xLabel = easyLabels(1,i);
%         
%         % KNN in original data space.
%         c = knnclassify(thisTrainingSet, thisTrainingLabels, x, K);
%         if c == xLabel
%            oCorrect = oCorrect + 1; 
%         end
%         
%         % Standard PCA.
%         projection = PCStandard' * x;
%         c = knnclassify(standardDataProjection, thisTrainingLabels, projection, K);
%         if c == xLabel
%             sCorrect = sCorrect + 1;
%         end
%         
%         % Modified PCA.
%         projection = PCModified' * x;
%         c = knnclassify(modifiedDataProjection, thisTrainingLabels, projection, K);
%         if c == xLabel
%            mCorrect = mCorrect + 1; 
%         end
%     end
%     standardEasy(1,count) = double(sCorrect) / easyTestLength;
%     modifiedEasy(1,count) = double(mCorrect) / easyTestLength;
%     originalEasy(1,count) = double(oCorrect) / easyTestLength;
% end
% plot(testRange, standardEasy, testRange, originalEasy);
% legend('Base', 'PCA');
% xlabel('T : # eigenvectors');
% ylabel('Accuracy');


