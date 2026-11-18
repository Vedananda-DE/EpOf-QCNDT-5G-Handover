function model = nn_train(X, y, layerSizes, params)
% Fully connected network with tanh hidden units and a linear output.
% Training minimises mean cross-entropy. layerSizes includes input and output.

rng(params.seed);
n = size(X, 1);
classes = layerSizes(end);
Y = zeros(n, classes);
for i = 1:n
    Y(i, y(i) + 1) = 1;
end

W = cell(numel(layerSizes) - 1, 1);
B = cell(numel(layerSizes) - 1, 1);
for k = 1:numel(W)
    W{k} = sqrt(2 / layerSizes(k)) * randn(layerSizes(k + 1), layerSizes(k));
    B{k} = zeros(layerSizes(k + 1), 1);
end

for ep = 1:params.baselineEpochs
    acts = nn_activations(W, B, X);
    delta = softmax_matrix(acts{end}) - Y;
    for k = numel(W):-1:1
        if k == 1
            incoming = X;
        else
            incoming = acts{k - 1};
        end
        dW = (delta.' * incoming) / n;
        dB = sum(delta, 1).' / n;
        if k > 1
            delta = (delta * W{k}) .* (1 - incoming .^ 2);
        end
        W{k} = W{k} - params.learningRate * dW;
        B{k} = B{k} - params.learningRate * dB;
    end
end
model.W = W;
model.B = B;
end

function acts = nn_activations(W, B, X)
acts = cell(numel(W), 1);
h = X;
for k = 1:numel(W)
    s = h * W{k}.' + B{k}.';
    if k < numel(W)
        h = tanh(s);
    else
        h = s;
    end
    acts{k} = h;
end
end
