%% Feature Extraction using a custom model
% We extract learned image features from a custom model imported from Python into Matlab using
%importkerasNetwork command. The same process is followed for the other individual models
% used in this study. Feature extraction is the easiest and fastest way use the representational 
% power of deep networks. Because feature extraction only requires a single pass through the data, 
% it is a good starting point if you do not have a GPU to accelerate network training with.
%% Load Data
train_folder = 'malaria100\train\'; %load training data
test_folder = 'malaria100\test\'; %load test data
categories = {'abnormal', 'normal'};

% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.

trainImages = imageDatastore(fullfile(train_folder, categories), 'LabelSource', 'foldernames'); 
testImages = imageDatastore(fullfile(test_folder, categories), 'LabelSource', 'foldernames'); 

% Extract the class labels from the training and test data.
YTrain = trainImages.Labels;
YTest = testImages.Labels;

%% Display some sample images

numTrainImages = numel(YTrain);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainImages,idx(i));
    imshow(I)
end

%% Load custom model using Import Keras Network

custom_modelfile='custom_cnn.25-0.9941.h5';
classNames = {'abnormal', 'normal'};
net = importKerasNetwork(custom_modelfile,'Classes',classNames);

%list network layers
net.Layers

%% plot a layer graph of the network layers

lgraph = layerGraph(net);
figure
plot(lgraph)

% observe the input image requirements for the model
inputSize = net.Layers(1).InputSize;

%% Extract Image Features

% The network constructs a hierarchical representation of input images. 
% Deeper layers contain higher-level features, constructed using the 
% lower-level features of earlier layers. To get the feature representations
% of the training and test images, use activations on the 
% penultimate layer of the model. To get a lower-level representation of the images, 
% use an earlier layer in the network. The network requires input images of 
% size 100-by-100-by-3. To automatically resize the training and test images before 
% they are input to the network, we can create augmented image datastores, 
% specify the desired image size, and use these datastores as 
% input arguments to activations.

augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages); 
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);

%features extracted from the penultimate layer of custom model

layer = 'dense_dropout1'; % varies with the model
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

%% Visualize the test features using t-sne
% Obtain two-dimensional analogs of the data clusters using t-SNE. 
% Use the Barnes-Hut algorithm for better performance on this large data set.
% Use PCA to reduce the initial dimensions to 50.
% For the current study, tsne with the following settings does a good job of embedding the
% high-dimensional initial data into two-dimensional points that have well defined clusters. 
% Perplexity: 70
% Exaggeration: 4
% Algorithm: Barneshut
% Distance: Eucleidean
% Learning Rate: 500
% NumDimensions: 3
% Verbose: 1

rng default % for reproducibility
Y = tsne(featuresTest,'Algorithm','barneshut','NumPCAComponents',50, 'Perplexity',70, 'NumDimensions',3);
figure
scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,YTest,'filled');
view(-93,14)
title('Custom t-SNE with 3-D embedding')

% t-SNE creates a figure with well separated clusters and relatively 
% few data points that seem misplaced.
%% 