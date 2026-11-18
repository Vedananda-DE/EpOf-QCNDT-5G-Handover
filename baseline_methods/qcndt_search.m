function [theta, history] = qcndt_search(X, y, Xval, yval, params, method)
% Population search for the same 199 QCNDT parameters used by EpOf.
% method is 'zoa' or 'goa'. Fitness is training accuracy. The returned
% vector is the one with the best validation accuracy.

rng(params.seed);
dim = params.parameterCount;
popSize = params.populationSize;
maxIter = params.maxIterations;
pop = zeros(popSize, dim);
pop(:, 1:4) = params.quantumLower + rand(popSize, 4) * (params.quantumUpper - params.quantumLower);
pop(:, 5:end) = params.weightLower + rand(popSize, dim - 4) * (params.weightUpper - params.weightLower);
fitness = zeros(popSize, 1);
for i = 1:popSize
    fitness(i) = mean(prediction(pop(i, :), X) == y);
end
[fitness, order] = sort(fitness, 'descend');
pop = pop(order, :);
bestVal = mean(prediction(pop(1, :), Xval) == yval);
bestTheta = pop(1, :);
history = zeros(maxIter, 2);

for t = 1:maxIter
    elite = pop(1, :);
    for i = 1:popSize
        rival = pop(randi(popSize), :);
        if strcmp(method, 'zoa')
            candidate = zebra_step(pop(i, :), elite, rival, t, maxIter);
        else
            candidate = gazelle_step(pop(i, :), elite, rival, t, maxIter);
        end
        candidate = bound_theta(candidate, params);
        score = mean(prediction(candidate, X) == y);
        if score >= fitness(i)
            pop(i, :) = candidate;
            fitness(i) = score;
        end
    end
    [fitness, order] = sort(fitness, 'descend');
    pop = pop(order, :);
    valScore = mean(prediction(pop(1, :), Xval) == yval);
    if valScore >= bestVal
        bestVal = valScore;
        bestTheta = pop(1, :);
    end
    history(t, :) = [fitness(1), valScore];
end
theta = bestTheta;
end

function candidate = zebra_step(position, elite, rival, t, maxIter)
I = round(1 + rand);
if rand < 0.5
    candidate = position + rand * (elite - I * position);
else
    if rand < 0.5
        candidate = position + 0.01 * (2 * rand - 1) * (1 - t / maxIter) * position;
    else
        candidate = position + rand * (rival - I * position);
    end
end
end

function candidate = gazelle_step(position, elite, rival, t, maxIter)
step = levy_step(numel(position));
if rand < 0.5
    candidate = position + 0.01 * step .* (elite - position);
else
    candidate = elite - rand * abs(elite - position);
end
if rand < 0.3
    candidate = position + 0.01 * (1 - t / maxIter) * step .* (position - rival);
end
end

function step = levy_step(dim)
beta = 1.5;
sigma = (gamma(1 + beta) * sin(pi * beta / 2) / ...
    (gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2))) ^ (1 / beta);
u = randn(1, dim) * sigma;
v = randn(1, dim);
step = u ./ (abs(v) .^ (1 / beta) + 1e-12);
end

function theta = bound_theta(theta, params)
theta(1:4) = min(max(theta(1:4), params.quantumLower), params.quantumUpper);
theta(5:end) = min(max(theta(5:end), params.weightLower), params.weightUpper);
end
