function net = buildNet(options)
%=========================================================================%
% Build the neural network.
%=========================================================================%

switch options.type
    case 'MLP1'
        input = imageInputLayer(options.inputSize, 'Name', 'input');
        fc1 = fullyConnectedLayer(1024, 'Name', 'fc1');
        relu1 = reluLayer('Name','relu1');
        drop1 = dropoutLayer(0.4, 'Name', 'drop1');
        fc2 = fullyConnectedLayer(1024, 'Name', 'fc2');
        relu2 = reluLayer('Name','relu2');
        drop2 = dropoutLayer(0.4, 'Name', 'drop2');
        fc3 = fullyConnectedLayer(1024, 'Name', 'fc3');
        relu3 = reluLayer('Name','relu3');
        drop3 = dropoutLayer(0.4, 'Name', 'drop3');
        fc4 = fullyConnectedLayer(1024, 'Name', 'fc4');
        relu4 = reluLayer('Name','relu4');
        drop4 = dropoutLayer(0.4, 'Name', 'drop4');
        fc5 = fullyConnectedLayer(options.numAnt(1), 'Name', 'fc5');
        sfm = softmaxLayer('Name','sfm');
        classifier = classificationLayer('Name','classifier');

        layers = [
                  input
                  fc1
                  relu1
                  drop1
                  fc2
                  relu2
                  drop2
                  fc3
                  relu3
                  drop3
                  fc4
                  relu4
                  drop4
                  fc5
                  sfm
                  classifier
                 ];
        net = layerGraph(layers);

end
