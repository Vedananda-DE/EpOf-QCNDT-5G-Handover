function result = baseline_qcndt_gd(Xtrain, ytrain, Xtest, params)
% QCNDT trained by minibatch gradient descent on cross-entropy.
% The four quantum angles and the fully connected weights are updated together.
% EpOf is not used. The hidden unit is tanh, as in the proposed forward pass.

rng(params.seed);
theta = zeros(params.parameterCount, 1);
theta(1:4) = params.quantumLower + rand(4, 1) * (params.quantumUpper - params.quantumLower);
theta(5:end) = params.weightLower + rand(params.parameterCount - 4, 1) * ...
    (params.weightUpper - params.weightLower);
batch = 1024;
history = zeros(params.baselineEpochs, 1);
n = size(Xtrain, 1);

for ep = 1:params.baselineEpochs
    order = randperm(n);
    total = 0;
    seen = 0;
    for start = 1:batch:n
        stop = min(start + batch - 1, n);
        rows = order(start:stop);
        [grad, loss] = qcndt_gradient(Xtrain(rows, :), ytrain(rows), theta);
        theta = theta - params.learningRate * grad;
        theta(1:4) = min(max(theta(1:4), params.quantumLower), params.quantumUpper);
        theta(5:end) = min(max(theta(5:end), params.weightLower), params.weightUpper);
        total = total + loss * numel(rows);
        seen = seen + numel(rows);
    end
    history(ep) = total / seen;
end

result.name = 'QCNDT-GD';
result.model = theta;
result.labels = prediction(theta, Xtest);
result.history = history;
end

function [grad, loss] = qcndt_gradient(X, y, theta)
n = size(X, 1);
phi = theta(1:4).';
angle = pi * X + phi;
de = 0.5 * sin(angle);
e = sin(angle / 2) .^ 2;
p1 = e(:, 1);
p2 = p1 + e(:, 2) - 2 * p1 .* e(:, 2);
p3 = p2 + e(:, 3) - 2 * p2 .* e(:, 3);
p4 = p3 + e(:, 4) - 2 * p3 .* e(:, 4);
Z = 1 - 2 * [p1, p2, p3, p4];
hybrid = [X, Z];

W1 = reshape(theta(5:132), 16, 8);
b1 = theta(133:148).';
W2 = reshape(theta(149:196), 3, 16);
b2 = theta(197:199).';
hidden = tanh(hybrid * W1.' + b1);
scores = hidden * W2.' + b2;
P = softmax_matrix(scores);
Y = zeros(n, 3);
for i = 1:n
    Y(i, y(i) + 1) = 1;
end
loss = -sum(sum(Y .* log(P + 1e-12))) / n;
delta = (P - Y) / n;

dW2 = delta.' * hidden;
db2 = sum(delta, 1).';
deltaH = (delta * W2) .* (1 - hidden .^ 2);
dW1 = deltaH.' * hybrid;
db1 = sum(deltaH, 1).';
dZ = deltaH * W1;
dZ = dZ(:, 5:8);

dp2_1 = de(:, 1) .* (1 - 2 * e(:, 2));
dp2_2 = de(:, 2) .* (1 - 2 * e(:, 1));
dp3_1 = dp2_1 .* (1 - 2 * e(:, 3));
dp3_2 = dp2_2 .* (1 - 2 * e(:, 3));
dp3_3 = de(:, 3) .* (1 - 2 * p2);
dp4_1 = dp3_1 .* (1 - 2 * e(:, 4));
dp4_2 = dp3_2 .* (1 - 2 * e(:, 4));
dp4_3 = dp3_3 .* (1 - 2 * e(:, 4));
dp4_4 = de(:, 4) .* (1 - 2 * p3);

dphi = zeros(1, 4);
dphi(1) = sum(dZ(:, 1) .* (-2 * de(:, 1)) + dZ(:, 2) .* (-2 * dp2_1) + ...
    dZ(:, 3) .* (-2 * dp3_1) + dZ(:, 4) .* (-2 * dp4_1));
dphi(2) = sum(dZ(:, 2) .* (-2 * dp2_2) + dZ(:, 3) .* (-2 * dp3_2) + dZ(:, 4) .* (-2 * dp4_2));
dphi(3) = sum(dZ(:, 3) .* (-2 * dp3_3) + dZ(:, 4) .* (-2 * dp4_3));
dphi(4) = sum(dZ(:, 4) .* (-2 * dp4_4));

grad = [dphi(:); dW1(:); db1(:); dW2(:); db2(:)];
end
