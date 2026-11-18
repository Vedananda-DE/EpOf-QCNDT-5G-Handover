function [scaled, scale] = normalize_data(labeled, params)
% Min-max scale fitted on the training sessions only, then applied to all rows.
% Values outside the training range are clipped to [0, 1].

names = params.selectedFeatures;
n = numel(names);
train = labeled.split == "train";
lo = zeros(1, n);
hi = zeros(1, n);
Z = zeros(height(labeled), n);
for k = 1:n
    x = double(labeled.(names{k}));
    lo(k) = min(x(train));
    hi(k) = max(x(train));
    span = hi(k) - lo(k);
    if span == 0
        span = 1;
    end
    Z(:, k) = min(max((x - lo(k)) / span, 0), 1);
end

scaled = array2table(Z, 'VariableNames', names);
scaled.HO_label = labeled.HO_label;
scaled.split = labeled.split;
scaled.session_id = labeled.session_id;
scale = table(names.', lo.', hi.', 'VariableNames', {'feature', 'min', 'max'});
end
