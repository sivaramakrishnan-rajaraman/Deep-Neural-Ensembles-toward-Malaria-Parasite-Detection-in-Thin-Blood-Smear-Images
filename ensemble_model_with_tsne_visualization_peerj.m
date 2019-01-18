%% Feature Extraction using an ensemble of models
% This code shows how to extract learned image features from indiviudal 
% models and concatenate the features to visualize the ensembled features.
% All the models are imported from Python into Matlab using
% importkerasNetwork command. The features are extracted from the penultimate layer
% and concatenated to form the feature ensemble.

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

%% Load the models that demonstrate the best performance on the test data

% Load the VGG19 model using Import Keras Network

modelfile1='weights/vgg19_custom.09-0.9959.h5';
classNames = {'abnormal', 'normal'};
net1 = importKerasNetwork(modelfile1,'Classes',classNames);
inputSize = net1.Layers(1).InputSize

%% Analyze the network. 
% AnalyzeNetwork displays an interactive plot of the network architecture and 
% a table containing information about the network layers.
% Investigate the network architecture using the plot to the left. Select a layer in the plot. 
% The selected layer is highlighted in the plot and in the layer table.
% In the table, view layer information such as layer properties, layer type, 
% and sizes of the layer activations and learnable parameters. 
analyzeNetwork(net1)

%% Load the squeezenet model

modelfile2='weights/squeeze_custom.13-0.9912.h5';
classNames = {'abnormal', 'normal'};
net2 = importKerasNetwork(modelfile2,'Classes',classNames);
inputSize = net2.Layers(1).InputSize
analyzeNetwork(net2)

%% Extract Image Features
% The network constructs a hierarchical representation of input images. 
% Deeper layers contain higher-level features, constructed using the 
% lower-level features of earlier layers. To get the feature representations
% of the training and test images, extract features from the penultimate layers
% of these individual models. 
% To get a lower-level representation of the images, 
% use an earlier layer in the network. The network requires input images of 
% size 100-by-100-by-3. To automatically resize the training and test images before 
% they are input to the network, create augmented image datastores, 
% specify the desired image size, and use these datastores as 
% input arguments to activations.

% for the VGG19 model
net1.Layers
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);

layer = 'global_average_pooling2d_5'; 
featuresTrain1 = activations(net1,augimdsTrain,layer,'OutputAs','rows');
featuresTest1 = activations(net1,augimdsTest,layer,'OutputAs','rows');

%% For Squeezenet

net2.Layers
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);

layer = 'global_average_pooling2d_6';
% varies for your data, extract from the penultimate layer
featuresTrain2 = activations(net2,augimdsTrain,layer,'OutputAs','rows');
featuresTest2 = activations(net2,augimdsTest,layer,'OutputAs','rows');

%% concatenate the features to compute the ensemble
% concatenate the features along the second dimension.

concatFeatureTrain = cat(2, featuresTrain1, featuresTrain2); % for training data
concatFeatureTest = cat(2, featuresTest1, featuresTest2); % for testing data

%% Visualize the concatenated test features using t-sne

% Obtain two-dimensional analogs of the data clusters using t-SNE. 
% Use the Barnes-Hut algorithm for better performance on this large data set.
% Use PCA to reduce the initial dimensions to 50.

rng default % for reproducibility
Y = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50);
figure
gscatter(Y(:,1),Y(:,2),YTest)
title('Default Figure')

% t-SNE creates a figure with well separated clusters and relatively 
% few data points that seem misplaced.

%% Perplexity
% Try altering the perplexity setting to see the effect on the figure.

rng default % for fair comparison
Y100 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'Perplexity',100);
figure
gscatter(Y100(:,1),Y100(:,2),YTest)
title('Perplexity 100')

rng default % for fair comparison
Y4 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'Perplexity',4);
figure
gscatter(Y4(:,1),Y4(:,2),YTest)
title('Perplexity 4')

% Setting the perplexity to 4 yields a figure that is largely similar to 
% the default figure. The clusters are tighter than with the default setting. 
% However, setting the perplexity to 100 gives a figure with better
% separated clusters. The clusters are tighter than with the default setting.

%% Vary Exaggeration
% Try altering the exaggeration setting to see the effect on the figure.

rng default % for fair comparison
YEX0 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'Exaggeration',20);
figure
gscatter(YEX0(:,1),YEX0(:,2),YTest)
title('Exaggeration 20')

rng default % for fair comparison
YEx15 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'Exaggeration',1.5);
figure
gscatter(YEx15(:,1),YEx15(:,2),YTest)
title('Exaggeration 1.5')

% While the exaggeration setting has an effect on the figure, 
% it is not clear whether any nondefault setting gives a better picture 
% than the default setting. The figure with an exaggeration of 20 
% is similar to the figure with less perplexity but the clusters are closely packed. 
% In general, a larger exaggeration creates more empty space between embedded clusters. 
% An exaggeration of 1.5 causes the groups labeled to split into two groups
% each, an undesirable outcome. Exaggerating the values in the 
% joint distribution of X makes the values in the joint distribution of Y smaller. 
% This makes it much easier for the embedded points to move relative to one another. 
% The splitting of cluster abnormal reflects this effect.

%% Vary Learning rate
% Try altering the learning rate setting to see the effect on the figure.

rng default % for fair comparison
YL5 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'LearnRate',5);
figure
gscatter(YL5(:,1),YL5(:,2),YTest)
title('Learning Rate 5')

rng default % for fair comparison
YL2000 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'LearnRate',2000);
figure
gscatter(YL2000(:,1),YL2000(:,2),YTest)
title('Learning Rate 2000')

% The figure with a learning rate of 5 has several clusters 
% that split into two or more pieces. This shows that if the learning rate 
% is too small, the minimization process can get stuck in a bad local minimum. 
% A learning rate of 2000 gives a figure similar to the perplexity 4 figure.

%% Initial Behavior with Various Settings
% Large learning rates or large exaggeration values can lead to 
% undesirable initial behavior. To see this, set large values of these 
% parameters and set NumPrint and Verbose to 1 to show all the iterations. 
% Stop the iterations after 10, as the goal of this experiment is simply to look at the initial behavior.
% Begin by setting the exaggeration to 200.

rng default % for fair comparison
opts = statset('MaxIter',10);
YEX200 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'Exaggeration',200,...
    'NumPrint',1,'Verbose',1,'Options',opts);

% The Kullback-Leibler divergence increases during the first few iterations, 
% and the norm of the gradient increases as well.
% To see the final result of the embedding, allow the algorithm to run to 
% completion using the default stopping criteria.

rng default % for fair comparison
YEX200 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'Exaggeration',200);
figure
gscatter(YEX200(:,1),YEX200(:,2),YTest)
title('Exaggeration 200')

% This exaggeration value does not give a clean separation into clusters.

%% Show the initial behavior when the learning rate is 100,000.

rng default % for fair comparison
YL100k = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'LearnRate',1e5,...
    'NumPrint',1,'Verbose',1,'Options',opts);

% Again, the Kullback-Leibler divergence increases during the first few 
% iterations, and the norm of the gradient increases as well.
% To see the final result of the embedding, allow the algorithm to run to 
% completion using the default stopping criteria.

rng default % for fair comparison
YL100k = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'LearnRate',1e5);
figure
gscatter(YL100k(:,1),YL100k(:,2),YTest)
title('Learning Rate 100,000')

% The learning rate is far too large, and gives no useful embedding.

%% Compare Distance Metrics
% Use various distance metrics to try to obtain a better separation between
% classes.

rng('default') % for reproducibility
Y = tsne(concatFeatureTest,'Algorithm','exact','Distance','cosine');
subplot(2,2,1)
gscatter(Y(:,1),Y(:,2),YTest)
title('Cosine')

rng('default') % for reproducibility
Y = tsne(concatFeatureTest,'Algorithm','exact','Distance','chebychev');
subplot(2,2,2)
gscatter(Y(:,1),Y(:,2),YTest)
title('Chebychev')

rng('default') % for reproducibility
Y = tsne(concatFeatureTest,'Algorithm','exact','Distance','euclidean');
subplot(2,2,3)
gscatter(Y(:,1),Y(:,2),YTest)
title('Euclidean')

% In this case, Eucleidean gave better separation in comparison to other.

%% Compare t-SNE loss

% Find both 2-D and 3-D embeddings of the data, and compare the loss 
% for each embedding. It is likely that the loss is lower for a 3-D embedding, 
% because this embedding has more freedom to match the original data.

rng default % for reproducibility
[Y,loss] = tsne(concatFeatureTest,'Algorithm','exact');

rng default % for reproducibility
[Y2,loss2] = tsne(concatFeatureTest,'Algorithm','exact','NumDimensions',3);
fprintf('2-D embedding has loss %g, and 3-D embedding has loss %g.\n',loss,loss2)

% 2-D embedding has loss 1.70449, and 3-D embedding has loss 1.5303.
% As expected, the 3-D embedding has lower loss.

%% View the embeddings. Use RGB colors [1 0 0], [0 1 0], and [0 0 1].
% For the 3-D plot, convert the YTest to numeric values using the 
% categorical command, then convert the numeric values to RGB colors 
% using the sparse function as follows. 
% If v is a vector of positive integers 1, 2, or 3, 
% corresponding to the species data, then the command
% sparse(1:numel(v),v,ones(size(v))) is a sparse matrix whose rows are 
% the RGB colors of the species.

gscatter(Y(:,1),Y(:,2),YTest,eye(3))
title('2-D Embedding')

% 3D Embedding
figure
v = double(categorical(YTest));
c = full(sparse(1:numel(v),v,ones(size(v)),numel(v),3));
scatter3(Y2(:,1),Y2(:,2),Y2(:,3),15,c,'filled')
title('3-D Embedding')
view(-50,8)

%% Conclusion
% tsne with the following settings does a good job of embedding the
% high-dimensional initial data into two-dimensional points that 
% have well defined clusters for the current study. 

% Perplexity: 70
% Exaggeration: 4
% Algorithm: Barneshut
% Distance: Eucleidean
% Learning Rate: 500
% NumPrint: 20
% InitialY: 1e-4
% NumDimensions: 3
% NumPCADimensions: 50

%% Reduce Dimension of Data to Three
% t-SNE can also reduce the data to three dimensions. 
% Set the tsne 'NumDimensions' name-value pair to 3.

rng default % for fair comparison
Y3 = tsne(concatFeatureTest,'Algorithm','barneshut','NumPCAComponents',50,'NumDimensions',3);
figure
scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,YTest,'filled');
view(-93,14)
%%