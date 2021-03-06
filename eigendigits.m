% This script is an introduction to Principal Components Analysis by
% way of analysis and classification of handwritten digits.
% 
% I evaluate and compare several variations of PCA that may offer benefits
% in different situations.
%
% Standard PCA. 
%    1. Mean Normalize input. 
%    2. Calculate the covariance matrix of the input, AA'.
%    3. Compute the eigenvectors of the covariance matrix.
%    4. Sort the eigenvectors in descending order.
%    5. Normalize the final eigenvectors.
%
% SVD PCA.
%    - Uses the Singular Value Decomposition of the covariance matrix,
%      rather than eigenvalue decomposition.
%
% Training-Dominated PCA.
%    - My informal name for the algebraic variant of PCA, which calculates
%      the covariance of A'A, rather than AA', where A is the input
%      training data. The primary advantage of this variant of PCA is for
%      situations in which the dimensionality of the data is very high
%      and perhaps computationally intractable. Rather than computing the
%      X by X covariance matrix (where X is the dimensionaltiy of the
%      data), we calculate A'A, whose dimensionality is dependent on the
%      number of training examples, which we can tune.
%
% I also apply these PCA variations as part of a classification algorithm
% for new handwritten digit samples and compare to a baseline KNN
% classification in the original data space.
% 
% Standard PCA KNN Classification.
%    - Using each of the approaches above, I project new examples into the
%      eigenspace generated by PCA, perform KNN in the eigenspace, and
%      classify the example according to the simple majority among the
%      nearest neighbors.
% 
% Eigenfun.
%    - Another made-up term. With PCA, we construct an eigenspace for our
%      training data and can project new examples into it. With this
%      variant, intended for classification, I instead construct a separate
%      eigenspace for each classification. I then project examples into
%      each eigenspace, perform KNN within each eigenspace, and classify
%      the example according to the minimum mean Euclidean Distance of
%      nearest neighbors among all the eigenspaces.

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

% -------------------------------
% Example PCA Analysis #1
% -------------------------------
% Here I apply standard PCA and observe the "eigendigits", or eigenvectors
% generated by PCA from the training examples.

disp('About to begin PCA Analysis #1. Press ENTER to continue.');
pause;

% Compute eigenmatrix.
[m, PC] = pcaeig(trainImages);

% Display first 25 eigendigits.
displayimages(255*PC(:,1:5));

% -------------------------------
% Example PCA Analysis #2
% -------------------------------
% Here I apply standard PCA and training-dominated PCA to reconstruct a digit 
% using reduced eigenspaces of varying dimensionality.

disp('About to begin PCA Analysis #2. Press ENTER to continue.');
pause;

[m, PC] = pcaeig(trainImages);

testImage = testImages(:,1);
p1   = PC(:,1:2)' * (testImage - m);
p1_r = PC(:,1:2) * p1;
p2   = PC(:,1:20)' * (testImage - m);
p2_r = PC(:,1:20) * p2;
p3   = PC(:,1:100)' * (testImage - m);
p3_r = PC(:,1:100) * p3;
p4   = PC(:,1:700)' * (testImage - m);
p4_r = PC(:,1:700) * p4; 

standard = [testImage p1_r p2_r p3_r p4_r];

[m, PC] = pcaeigreduce(trainImages(:,1:5000));

testImage = testImages(:,1);
p1   = PC(:,1:2)' * (testImage - m);
p1_r = PC(:,1:2) * p1;
p2   = PC(:,1:20)' * (testImage - m);
p2_r = PC(:,1:20) * p2;
p3   = PC(:,1:100)' * (testImage - m);
p3_r = PC(:,1:100) * p3;
p4   = PC(:,1:700)' * (testImage - m);
p4_r = PC(:,1:700) * p4; 

displayimages([ standard testImage p1_r p2_r p3_r p4_r]);

% -------------------------------
% Experiment #1
% -------------------------------
% Here I compare KNN in PCA and original data space for various N.

disp('About to begin Experiment 1. Press ENTER to continue.');
disp('Warning: This takes a LONG time!');
pause;

test        = testImages(:,1:500);
testLabels  = testLabels(1:1:500);

N_Tests = [ 20 50 500 1000 5000 60000 ];
testResults = zeros(length(N_Tests), 1);
baseResults = zeros(length(N_Tests), 1);

for i = 1:length(N_Tests)
    N = N_Tests(i);
    T = 500;
    testResults(i) = runexperiment(trainImages, trainLabels, test, testLabels, K, T, N);
    
    baseResults(i) = runbaseline(trainImages, trainLabels, test, testLabels, K, N);
end

plot(N_Tests, testResults, N_Tests, baseResults);
legend('Base', 'PCA');
xlabel('N : Number of Training Examples');
ylabel('Accuracy');


% -------------------------------
% Experiment #2
% -------------------------------
% Here I compare KNN in the PCA and original data space for various T.

disp('About to begin Experiment 2. Press ENTER to continue.');
disp('Warning: This takes a LONG time!');
pause;

K = 8;

T_Tests = [ 2 5 10 20 50 100 200 700 ];
testResults = zeros(length(T_Tests), 1);
baseResults = zeros(length(T_Tests), 1);
b = zeros(length(T_Tests), 1);
for i = 1:length(T_Tests)
    N = 60000;
    T = T_Tests(i);
    tic
    testResults(i) = runexperiment(trainImages(:,1:N), trainLabels(:,1:N), test, testLabels, K, T, N);
    toc
    
    tic
    baseResults(i) = runbaseline(trainImages(:,1:N), trainLabels(:,1:N), test, testLabels, K, N);
    toc
end

% These are measured results from a particular run measured with respect
% to the baseline for comparison.
testComputeTime = [ 0.08 0.088 0.09 0.09 0.127 0.1737 0.280 0.776 ];
testMemory      = [ 0.002 0.006 0.0127 0.0255 0.0637 0.1275 0.255 0.893 ];
plot(T_Tests, baseResults, T_Tests, testResults, T_Tests, testComputeTime, T_Tests, testMemory);
legend('Base', 'PCA', 'CPU', 'Memory');
xlabel('T : Number of Eigenvectors');
ylabel('Accuracy');

% -------------------------------
% Experiment #3
% -------------------------------
% Here I apply the Eigenfun algorithm for classification and compare it
% to KNN in the standard PCA and original data spaces for various T.

disp('About to begin Experiment 3. Press ENTER to continue.');
pause;

specialResults = zeros(length(T_Tests), 1);
for i = 1:length(T_Tests)
   N = 60000;
   T = T_Tests(i);
   
   specialResults(i) = eigenfun(trainImages(:,1:N), trainLabels(:,1:N), test, testLabels, 4, T);
end

plot(T_Tests, baseResults, T_Tests, testResults, T_Tests, specialResults)
legend('Base', 'PCA', 'Eigenfun');
xlabel('T : Number of Eigenvectors');
ylabel('Accuracy');





