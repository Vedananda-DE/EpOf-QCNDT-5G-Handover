function main()
% Full simulation entry point.
% Train EpOf-QCNDT and the comparison models, then write metrics and figures.
% The test split is used only after every model has been selected.

root = fileparts(mfilename('fullpath'));
addpath(genpath(root));
params = simulation_parameters();
data = load_prepared_split(params);

Xtr = data.X(data.train, :);
ytr = data.y(data.train);
Xva = data.X(data.val, :);
yva = data.y(data.val);
Xte = data.X(data.test, :);
yte = data.y(data.test);
mob = data.mobility(data.test);

timer = tic;
[model, epofHistory] = training(Xtr, ytr, Xva, yva, params);
proposedLabels = prediction(model.theta, Xte);
proposedSeconds = toc(timer);

results = {};
results{end + 1} = struct('name', 'EpOf-QCNDT', 'labels', proposedLabels, 'history', epofHistory);
results{end + 1} = baseline_xgboost(Xtr, ytr, Xte, params);
results{end + 1} = baseline_ddpg(Xtr, ytr, Xte, params);
results{end + 1} = baseline_mfnn(Xtr, ytr, Xte, params);
results{end + 1} = baseline_fldhdt(data.raw(data.test, :), params);
results{end + 1} = baseline_dt_mlp(Xtr, ytr, Xte, params);
results{end + 1} = baseline_dt_fc(Xtr, ytr, Xte, params);
results{end + 1} = baseline_qcndt_gd(Xtr, ytr, Xte, params);
results{end + 1} = baseline_qcndt_zoa(Xtr, ytr, Xva, yva, Xte, params);
results{end + 1} = baseline_qcndt_goa(Xtr, ytr, Xva, yva, Xte, params);
results{end + 1} = baseline_mlp(Xtr, ytr, Xte, params);
results{end + 1} = baseline_aho_drl(Xtr, ytr, Xte, params);

summary = compare_methods(results, yte, mob);
study = node_round_study(yte, proposedLabels, params);
latency = detection_time(proposedSeconds, numel(yte));

outTables = fullfile(params.resultsDir, 'tables');
outFigures = fullfile(params.resultsDir, 'figures');
outMetrics = fullfile(params.resultsDir, 'metrics');
if ~exist(outTables, 'dir'); mkdir(outTables); end
if ~exist(outMetrics, 'dir'); mkdir(outMetrics); end
writetable(summary, fullfile(outTables, 'method_comparison.csv'));
writetable(study, fullfile(outTables, 'node_round_study.csv'));
fid = fopen(fullfile(outMetrics, 'detection_time_ms.txt'), 'w');
fprintf(fid, 'Mean decision time %.6f ms\n', latency);
fclose(fid);

plot_comparison(summary, outFigures);
plot_performance(summary, outFigures);
if ~isempty(epofHistory)
    curve = [epofHistory.best_train_accuracy, epofHistory.leader_val_accuracy];
    plot_convergence(curve, outFigures);
end
save(fullfile(outMetrics, 'final_results.mat'), 'summary', 'study', 'latency', 'model');
fprintf('Comparison written to %s\n', outTables);
fprintf('Test samples: %d. Mean decision time: %.4f ms\n', numel(yte), latency);
end
