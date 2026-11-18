function main_phase4()
% Train EpOf-QCNDT on the labeled production traces.
% Test sessions are loaded only so they stay identifiable. They are not
% used to choose the model.

root = fileparts(mfilename('fullpath'));
addpath(genpath(root));
params = simulation_parameters();

dataFile = fullfile(params.labeledDir, 'labeled_traces.mat');
S = load(dataFile);
X = double(S.features_scaled);
y = double(S.HO_label(:));
split = double(S.split_code(:));

train = split == 0;
val = split == 1;
[model, history] = training(X(train, :), y(train), X(val, :), y(val), params);
valPred = prediction(model.theta, X(val, :));
valAccuracy = mean(valPred == y(val));

outDir = fullfile(params.resultsDir, 'metrics');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
save(fullfile(outDir, 'epof_qcndt_model.mat'), 'model', 'history', 'valAccuracy');
fprintf('Validation accuracy: %.4f\n', valAccuracy);
fprintf('Test sessions were not used for training or model selection.\n');
end
