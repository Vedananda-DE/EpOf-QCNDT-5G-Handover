function result = actor_critic(Xtrain, ytrain, Xtest, params, mode)
% Discrete actor-critic used for the DDPG and AHO comparisons.
% The actor emits three class scores. The critic scores a state-action pair.
% Reward is 1 when the action matches the label. AHO also penalises a class
% change on the next training row, which is the ping-pong term.

rng(params.seed);
actor = init_net([size(Xtrain, 2), 16, params.nClasses]);
critic = init_net([size(Xtrain, 2) + params.nClasses, 16, 1]);
n = size(Xtrain, 1);
gamma = 0.9;

for ep = 1:params.baselineEpochs
    scores = net_forward(actor, Xtrain);
    P = softmax_matrix(scores);
    action = zeros(n, 1);
    onehot = zeros(n, params.nClasses);
    for i = 1:n
        draw = find(rand <= cumsum(P(i, :)), 1, 'first');
        action(i) = draw;
        onehot(i, draw) = 1;
    end
    reward = double(action - 1 == ytrain);
    if strcmp(mode, 'aho')
        flips = [0; action(2:end) ~= action(1:end-1)];
        reward = reward - 0.2 * flips;
    end
    nextScores = net_forward(actor, Xtrain([2:n, n], :));
    [~, nextAction] = max(nextScores, [], 2);
    nextHot = zeros(n, params.nClasses);
    for i = 1:n
        nextHot(i, nextAction(i)) = 1;
    end
    target = reward + gamma * net_forward(critic, [Xtrain([2:n, n], :), nextHot]);
    critic = fit_regressor(critic, [Xtrain, onehot], target, params.learningRate);
    qTaken = net_forward(critic, [Xtrain, onehot]);
    actor = policy_step(actor, Xtrain, P, qTaken, params.learningRate);
end

testScores = net_forward(actor, Xtest);
[~, testIdx] = max(testScores, [], 2);
if strcmp(mode, 'aho')
    result.name = 'AHO-DRL';
else
    result.name = 'DDPG';
end
result.model.actor = actor;
result.model.critic = critic;
result.labels = testIdx - 1;
result.history = [];
end

function net = init_net(sizes)
net.W = cell(numel(sizes) - 1, 1);
net.B = cell(numel(sizes) - 1, 1);
for k = 1:numel(net.W)
    net.W{k} = 0.1 * randn(sizes(k + 1), sizes(k));
    net.B{k} = zeros(sizes(k + 1), 1);
end
end

function h = net_forward(net, X)
h = X;
for k = 1:numel(net.W)
    s = h * net.W{k}.' + net.B{k}.';
    if k < numel(net.W)
        h = tanh(s);
    else
        h = s;
    end
end
end

function net = fit_regressor(net, X, target, lr)
hidden = tanh(X * net.W{1}.' + net.B{1}.');
pred = hidden * net.W{2}.' + net.B{2}.';
delta = pred - target;
n = size(X, 1);
dW2 = (delta.' * hidden) / n;
dB2 = sum(delta, 1).' / n;
deltaH = (delta * net.W{2}) .* (1 - hidden .^ 2);
dW1 = (deltaH.' * X) / n;
dB1 = sum(deltaH, 1).' / n;
net.W{2} = net.W{2} - lr * dW2;
net.B{2} = net.B{2} - lr * dB2;
net.W{1} = net.W{1} - lr * dW1;
net.B{1} = net.B{1} - lr * dB1;
end

function net = policy_step(net, X, P, qTaken, lr)
% Increase the probability of actions that received a higher critic value.
hidden = tanh(X * net.W{1}.' + net.B{1}.');
direction = (qTaken - mean(qTaken)) .* (P - 1 / size(P, 2));
delta = direction;
n = size(X, 1);
dW2 = (delta.' * hidden) / n;
dB2 = sum(delta, 1).' / n;
deltaH = (delta * net.W{2}) .* (1 - hidden .^ 2);
dW1 = (deltaH.' * X) / n;
dB1 = sum(deltaH, 1).' / n;
net.W{2} = net.W{2} + lr * dW2;
net.B{2} = net.B{2} + lr * dB2;
net.W{1} = net.W{1} + lr * dW1;
net.B{1} = net.B{1} + lr * dB1;
end
