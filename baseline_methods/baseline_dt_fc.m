function result = baseline_dt_fc(Xtrain, ytrain, Xtest, params)
% Decision-tree features passed straight to a fully connected head.
% There is no quantum encoding. Width 4-16-3.

model = nn_train(Xtrain, ytrain, [size(Xtrain, 2), params.hiddenNodes, params.nClasses], params);
result.name = 'DT+FC';
result.model = model;
result.labels = nn_forward(model, Xtest);
result.history = [];
end
