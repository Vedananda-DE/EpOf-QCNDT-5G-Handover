function result = baseline_mfnn(Xtrain, ytrain, Xtest, params)
% Multi-layer feed-forward network used as the MFNN comparison.
% The cited paper does not give this width. The depth used here is 4-16-16-3.

model = nn_train(Xtrain, ytrain, [size(Xtrain, 2), 16, 16, params.nClasses], params);
result.name = 'MFNN';
result.model = model;
result.labels = nn_forward(model, Xtest);
result.history = [];
end
