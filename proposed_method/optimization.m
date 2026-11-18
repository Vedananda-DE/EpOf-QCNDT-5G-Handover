function [best, history] = optimization(X, y, Xval, yval, params)
% Escape Offensive Optimization.
% Exploration mixes the zebra foraging step with a gazelle Levy perturbation.
% Exploitation escapes when a rival is fitter and otherwise moves offensively
% toward the leader and away from the worst member. A candidate replaces the
% current one only when training accuracy does not fall.

rng(params.seed);
dim = params.parameterCount;
popSize = params.populationSize;
maxIter = params.maxIterations;
pop = initialize_population(popSize, dim, params);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = train_accuracy(pop(i, :), X, y);
end

[fitness, order] = sort(fitness, 'descend');
pop = pop(order, :);
best = pop(1, :);
bestTrain = fitness(1);
bestVal = train_accuracy(best, Xval, yval);
valTheta = best;

history = zeros(maxIter, 3);
for t = 1:maxIter
    worst = pop(end, :);
    elite = pop(1, :);
    for i = 1:popSize
        candidate = propose(pop(i, :), i, t, maxIter, pop, fitness, elite, worst);
        candidate = apply_bounds(candidate, params);
        score = train_accuracy(candidate, X, y);
        if score >= fitness(i)
            pop(i, :) = candidate;
            fitness(i) = score;
        end
    end
    [fitness, order] = sort(fitness, 'descend');
    pop = pop(order, :);
    if fitness(1) > bestTrain
        bestTrain = fitness(1);
        best = pop(1, :);
    end
    valScore = train_accuracy(pop(1, :), Xval, yval);
    if valScore >= bestVal
        bestVal = valScore;
        valTheta = pop(1, :);
    end
    history(t, :) = [bestTrain, mean(fitness), valScore];
end
best = valTheta;
history = array2table(history, 'VariableNames', ...
    {'best_train_accuracy', 'mean_train_accuracy', 'leader_val_accuracy'});
end

function pop = initialize_population(popSize, dim, params)
pop = zeros(popSize, dim);
spanQ = params.quantumUpper - params.quantumLower;
spanW = params.weightUpper - params.weightLower;
pop(:, 1:4) = params.quantumLower + rand(popSize, 4) * spanQ;
pop(:, 5:end) = params.weightLower + rand(popSize, dim - 4) * spanW;
end

function theta = apply_bounds(theta, params)
theta(1:4) = min(max(theta(1:4), params.quantumLower), params.quantumUpper);
theta(5:end) = min(max(theta(5:end), params.weightLower), params.weightUpper);
end

function acc = train_accuracy(theta, X, y)
labels = prediction(theta, X);
acc = mean(labels == y);
end

function candidate = propose(position, index, t, maxIter, pop, fitness, elite, worst)
dim = numel(position);
r1 = rand;
r2 = rand;
Y = round(1 + rand);
q = 1 - 2 * (rand > 0.5);
        rivalIndex = randi(size(pop, 1));
rival = pop(rivalIndex, :);
if r1 < (1 - t / maxIter)
    if index <= size(pop, 1) / 2
        candidate = position + r2 * (elite - Y * position) + q * 0.01 * levy_step(dim);
    else
        brown = randn(1, dim);
        candidate = elite + exp(-t / maxIter) * r2 * brown .* (elite - position);
    end
elseif fitness(rivalIndex) > fitness(index)
    candidate = position + r2 * (1 - t / maxIter) * (position - rival) + 0.01 * levy_step(dim);
else
    candidate = position + r2 * (elite - Y * rival) + rand * (position - worst);
end
end

function step = levy_step(dim)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
    (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
u = randn(1, dim) * sigma;
v = randn(1, dim);
step = u ./ (abs(v).^(1 / beta) + 1e-12);
end
