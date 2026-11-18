function main_preprocess()
% Phase 2-3 entry point: load, clean, label, select, scale, save.
% The quantum model is intentionally not called from this file.

root = fileparts(mfilename('fullpath'));
addpath(genpath(root));
params = simulation_parameters();
raw = load_dataset(params);
cleaned = clean_data(raw, params);
sites = deploy_cells(cleaned, params);
labeled = generate_labels(cleaned, params);
[selected, ranking] = feature_selection(labeled, params);
[scaled, scale] = normalize_data(labeled, params);

if ~exist(params.labeledDir, 'dir')
    mkdir(params.labeledDir);
end
if ~exist(params.normDir, 'dir')
    mkdir(params.normDir);
end
if ~exist(params.analysisDir, 'dir')
    mkdir(params.analysisDir);
end

save(fullfile(params.labeledDir, 'labeled_traces_matlab.mat'), ...
    'labeled', 'sites', 'selected', 'params', '-v7.3');
writetable(sites, fullfile(params.labeledDir, 'cell_layout_matlab.csv'));
writetable(ranking, fullfile(params.analysisDir, 'feature_ranking_matlab.csv'));
writetable(scale, fullfile(params.normDir, 'feature_scaling_matlab.csv'));
writetable(scaled, fullfile(params.normDir, 'selected_features_scaled_matlab.csv'));

counts = accumarray(labeled.HO_label + 1, 1, [3 1]);
fprintf('Labeled rows: %d\n', height(labeled));
for c = 1:3
    fprintf('  class %d %-28s %d\n', c - 1, params.classNames{c}, counts(c));
end
fprintf('Selected features: %s\n', strjoin(params.selectedFeatures, ', '));
end
