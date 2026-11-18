function data = load_prepared_split(params)
% Load the labeled MAT file and the session split.
% split codes are 0 train, 1 validation, 2 test.

S = load(fullfile(params.labeledDir, 'labeled_traces.mat'));
data.X = double(S.features_scaled);
data.y = double(S.HO_label(:));
data.raw = [double(S.RSRP(:)), double(S.delta_RSRP(:)), ...
            double(S.SNR(:)), double(S.Speed(:))];
data.split = double(S.split_code(:));
data.mobility = double(S.mobility_code(:));
data.train = data.split == 0;
data.val = data.split == 1;
data.test = data.split == 2;
end
