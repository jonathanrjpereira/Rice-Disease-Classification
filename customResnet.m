
imds = imageDatastore('C:\Users\Jonathan\Documents\GitHub\RiceDisease\Rice_All_Resize',...
       'IncludeSubfolders',true,'LabelSource','foldernames');

% imds = imageDatastore('dataset1',...
%        'IncludeSubfolders',true,'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
% Load pretrained Network
net = resnet50;



%Extract the layer graph from the trained network and plot the layer graph.
lgraph = layerGraph(net);


% figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
%plot(lgraph)

% Check first layer input images dimensions

net.Layers(1)
inputSize = net.Layers(1).InputSize;

% Replacing last three layers for transfer learning / retraining

lgraph = removeLayers(lgraph, {'ClassificationLayer_fc1000','fc1000_softmax','fc1000'});

numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

% Connect last transfer layer to new layers and check
lgraph = connectLayers(lgraph,'avg_pool','fc');

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% %plot(lgraph)
% ylim([0,10])

% Set layers to 0 for speed and prevent over fitting

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);

%% Train the network
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing','gray2rgb');

options = trainingOptions('sgdm', ...
    'MiniBatchSize',2, ...
    'MaxEpochs',1, ... % was 6
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

[trainedNet, traininfo] = trainNetwork(augimdsTrain,lgraph,options);
save 'transfer' trainedNet
%load 'transfer' trainedNet
%% Classify Validation Images
[YPred,probs] = classify(trainedNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
%Calculate Confusion Matrix
cm=confusionmat(imdsValidation.Labels,YPred)

%Display confusion matrix
figure;
imagesc(cm)
colorbar
figure;
plotconfusion(imdsValidation.Labels,YPred)

function layers = freezeWeights(layers)
for i = 1:numel(layers)
        if isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
        end
        if isprop(layers(i),'WeightL2Factor')
            layers(i).WeightL2Factor = 0;
        end
        if isprop(layers(i),'BiasLearnRateFactor')
            layers(i).BiasLearnRateFactor = 0;
        end
        if isprop(layers(i),'BiasL2Factor')
            layers(i).BiasL2Factor = 0;
        end
end
end


function lgraph = createLgraphUsingConnections(layers,connections)
% lgraph = createLgraphUsingConnections(layers,connections) creates a layer
% graph with the layers in the layer array |layers| connected by the
% connections in |connections|.

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end
