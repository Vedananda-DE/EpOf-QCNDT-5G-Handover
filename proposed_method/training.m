function [model, history] = training(Xtrain, ytrain, Xval, yval, params)
% Tune the 199 QCNDT parameters with EpOf.
% The search uses training accuracy. The returned model is the leader with
% the highest validation accuracy. The test split is not read here.

[theta, history] = optimization(Xtrain, ytrain, Xval, yval, params);
model.theta = theta;
model.selectedFeatures = params.selectedFeatures;
model.hiddenNodes = params.hiddenNodes;
end
