function result = baseline_dt_mlp(Xtrain, ytrain, Xtest, params)
% Capacity-matched classical stand-in for the 4-qubit circuit.
% A 4-by-4 linear map replaces the quantum layer. Its output is concatenated
% with the four original features and classified by an 8-16-3 head.

rng(params.seed);
M = 0.1 * randn(4, 4);
head = nn_train(augment(Xtrain, M), ytrain, [8, params.hiddenNodes, params.nClasses], params);
for ep = 1:params.baselineEpochs
    H = augment(Xtrain, M);
    acts = head_activations(head, H);
    Y = class_targets(ytrain, params.nClasses);
    delta = softmax_matrix(acts{end}) - Y;
    n = size(Xtrain, 1);
    dW2 = (delta.' * acts{1}) / n;
    dB2 = sum(delta, 1).' / n;
    deltaH = (delta * head.W{2}) .* (1 - acts{1} .^ 2);
    dW1 = (deltaH.' * H) / n;
    dB1 = sum(deltaH, 1).' / n;
    dH = deltaH * head.W{1};
    dM = (Xtrain.' * dH(:, 5:8)) / n;
    head.W{2} = head.W{2} - params.learningRate * dW2;
    head.B{2} = head.B{2} - params.learningRate * dB2;
    head.W{1} = head.W{1} - params.learningRate * dW1;
    head.B{1} = head.B{1} - params.learningRate * dB1;
    M = M - params.learningRate * dM;
end

result.name = 'DT+MLP';
result.model.head = head;
result.model.map = M;
result.labels = nn_forward(head, augment(Xtest, M));
result.history = [];
end

function H = augment(X, M)
H = [X, X * M];
end

function acts = head_activations(model, X)
hidden = tanh(X * model.W{1}.' + model.B{1}.');
acts = {hidden, hidden * model.W{2}.' + model.B{2}.'};
end

function Y = class_targets(y, classes)
Y = zeros(numel(y), classes);
for i = 1:numel(y)
    Y(i, y(i) + 1) = 1;
end
end
