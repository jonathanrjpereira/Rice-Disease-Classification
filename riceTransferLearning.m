imds = imageDatastore('C:\Users\Jonathan\Documents\GitHub\RiceDisease\Rice_All_Resize','IncludeSubfolders',true,'LabelSource','foldernames'); 
% [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7); %Divide the data into training and validation data sets.70%:Train. 30%Val
% [imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.8,0.1,'randomized')

net = mobilenetv2; %Select the type of pretrained model
analyzeNetwork(net) %Visualize the model

net.Layers(1) %Input Layer Properties

inputSize = net.Layers(1).InputSize; %Set Image Resize to Input Size of the model

% Extract the layer graph from the trained network.
% If the network is a SeriesNetwork object, such as AlexNet, VGG-16, or VGG-19, 
% then convert the list of layers in net.Layers to a layer graph.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph); %Find layer to be replaced
[learnableLayer,classLayer] 

% In most networks, the last layer with learnable weights is a fully connected layer. 
% Replace this fully connected layer with a new fully connected layer with the 
% number of outputs equal to the number of classes in the new data set

numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph)
% ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

% Train the Network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
validation_accuracy = mean(YPred == imdsValidation.Labels)
cmValidation=confusionmat(imdsValidation.Labels,YPred)
cmPlotValidation = confusionchart(cmValidation,categories(imdsTrain.Labels))


[YPred,probs] = classify(net,augimdsTest);
test_accuracy = mean(YPred == imdsTest.Labels)
cmTest=confusionmat(imdsTest.Labels,YPred)
cmPlotTest = confusionchart(cmTest,categories(imdsTest.Labels))

% %Display confusion matrix
% figure;
% imagesc(cm)
% colorbar
% figure;
% plotconfusion(imdsValidation.Labels,YPred)
% 
% idx = randperm(numel(imdsValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdsValidation,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
% end