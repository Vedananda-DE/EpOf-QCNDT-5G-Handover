function result = baseline_xgboost(Xtrain, ytrain, Xtest, params)
% Multiclass gradient boosting with regression stumps.
% One stump per class is fit to the negative softmax gradient each round.

rounds = params.boostingRounds;
rate = params.learningRate;
n = size(Xtrain, 1);
score = zeros(n, params.nClasses);
stumps = cell(rounds, params.nClasses);
for t = 1:rounds
    P = softmax_matrix(score);
    for k = 1:params.nClasses
        gradient = P(:, k) - double(ytrain == (k - 1));
        stump = fit_stump(Xtrain, -gradient);
        score(:, k) = score(:, k) + rate * apply_stump(stump, Xtrain);
        stumps{t, k} = stump;
    end
end

result.name = 'XGBoost';
result.model.stumps = stumps;
result.model.rate = rate;
result.labels = predict_boosted(result.model, Xtest);
result.history = [];
end

function labels = predict_boosted(model, X)
score = zeros(size(X, 1), size(model.stumps, 2));
for t = 1:size(model.stumps, 1)
    for k = 1:size(model.stumps, 2)
        score(:, k) = score(:, k) + model.rate * apply_stump(model.stumps{t, k}, X);
    end
end
[~, idx] = max(score, [], 2);
labels = idx - 1;
end
