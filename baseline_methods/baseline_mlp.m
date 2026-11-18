function result = baseline_mlp(Xtrain, ytrain, Xtest, params)
% Data-driven MLP comparison. Width 4-32-3.

model = nn_train(Xtrain, ytrain, [size(Xtrain, 2), 32, params.nClasses], params);
result.name = 'Data-driven MLP';
result.model = model;
result.labels = nn_forward(model, Xtest);
result.history = [];
end
