# Deep-Neural-Ensembles-toward-Malaria-Parasite-Detection-in-Thin-Blood-Smear-Images
Malaria is a disease caused by the Plasmodium parasites that infects the red blood cells and results in life-threatening symptoms. Microscopic thick and thin film blood examination remains the common known method for disease diagnosis. However, manual identification and counting is burdensome. State-of-the-art computer aided diagnostic tools based on data-driven deep learning algorithms like convolutional neural networks (CNN) have become the architecture of choice for medical image recognition tasks. However, CNN suffers from high variance and may overfit due to their sensitivity to training data fluctuations. Model ensembles reduce variance by training and combining multiple models to learn a heterogeneous collection of mapping functions with reduced correlation in their predictions. In this study, we evaluate the performance of custom and pretrained CNNs and construct an optimal model ensemble toward the challenge of classifying parasitized and normal cells in thin blood smear images. The results obtained are encouraging and superior to the state-of-the-art. 

# Prerequisites:

Keras >= 2.4.0
Tensorflow-GPU >= 1.9.0
OpenCV >= 3.3
Jupyter
Matlab >= R2018b

Optimizing the parameters of custom model:
We optimized the parameters and hyperparameters of the custom CNN model using the Talos optimization tool (https://github.com/autonomio/talos). The following parameters are optimized: a) dropout in the convolutional layer; b) dropout in the dense layer; c) optimizer; d) activation function; and e) number of neurons in the dense layer. The process is repeated until an acceptable model is found. The script is made available as a Jupyter notebook file (custom_cnn_optimization.ipynb)
Fine-tuning the pretrained CNN models:
We instantiated the pretrained CNNs including VGG-19, SqueezeNet, and InceptionResNet-V2 with their convolutional layer weights and truncated these models at their deepest convolutional layer. A GAP and dense layer are added to learn and predict on the cell image data.  We fine-tuned the models entirely using a very low learning rate (0.0001) with the Adam optimizer to minimize the categorical cross-entropic loss as not to rapidly modify the pretrained weights. 
Constructing the model ensemble
The predictions of the custom and pretrained CNN models are averaged to construct the model ensemble. The following figure shows the process flow diagram for combining the predictions of the predictive models and selecting the optimal ensemble from a collection of model combinations for further deployment. The script is made available as a Jupyter notebook file (model_ensemble.ipynb)
 
 
Process flow diagram for constructing a model averaging ensemble. 
Dimensionality Reduction and Feature space visualization
Images are often high-dimensional and it would be interesting to explore and analyze the hidden patterns in the data. This could be achieved with a non-linear dimensionality reduction technique like t-distributed stochastic neighbor embedding (t-SNE) (Van Der Maaten & Hinton, 2008). We implemented t-SNE to visualize features extracted from the custom, pretrained and the optimal ensemble models in the two-dimensional space and analyzed the embedded clusters. The keras model is imported into Matlab R2018b. The features from the penultimate layer of the individual models and the concatenated features of the models in the optimal ensemble are visualized in the low-dimensional space. The script is made available as a Jupyter notebook file (model_ensemble.ipynb) custom_model_with_tsne_prediction_peerJ.m and ensemble_model_with_tsne_visualization_peerj.m

