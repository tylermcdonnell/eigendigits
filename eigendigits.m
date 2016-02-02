% This script is an introduction to Principal Components Analysis
% through analysis of handwritten digits.

% Load test and training data.
disp('Loading test and training data...');
load digits.mat;
disp('Done.');

% -------------------------------
% Data Preprocessing
% -------------------------------

% Reshape data sets to be 2D X by K matrices, where X is the length
% of the features, in this case 28x28=784 pixels, and K is the number
% of examples in the data set.
trainImages = reshape(trainImages, 784, 60000);
testImages  = reshape(testImages, 784, 10000);

% For consistent computations throughout.
trainImages = double(trainImages);
testImages  = double(testImages);

% -------------------------------
% Example PCA Analysis.
% -------------------------------
% For this analysis, we will train using the traditional method,
% where the covariance matrix = AA' (X by X) and all training data.
disp('About to begin example PCA Analysis. Press Enter to continue.');
pause;

% Apply standard PCA algorithm.
[m, V] = pcasvd(trainImages);
m

% Prepare test images for projection.
images = [];
numberOfTestImages = 4;
for i = 1:numberOfTestImages
   images = cat(3, images, toimage(testImages(:,i)));
end
displayimages(images);

disp('Will next project digits into the eigenspace. Press Enter.');
pause;

% Display projected images.
projectedImages = [];
for i = 1:numberOfTestImages
    projection = tovector(images(:,:,i));
    projection = V' * (projection - m);
    projectedImages = cat(3, projectedImages, toimage(projection));
end
displayimages(projectedImages);




