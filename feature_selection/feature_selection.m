function [selected, ranking] = feature_selection(labeled, params)
% The handover model uses four features:
%   RSRP, delta_RSRP, SNR, Speed.
% Gini ranking is retained as a diagnostic. It does not replace this set.

selected = labeled(:, [params.selectedFeatures, {'HO_label', 'split', 'session_id', 'mobility'}]);

candidates = {'RSRP','delta_RSRP','SNR','Speed','RSRQ','CQI','Longitude','Latitude','CellID'};
score = zeros(numel(candidates), 1);
train = selected.split == "train";
y = labeled.HO_label(train);
for k = 1:numel(candidates)
    x = double(labeled.(candidates{k})(train));
    score(k) = gini_decrease(x, y);
end
[score, order] = sort(score, 'descend');
ranking = table(candidates(order).', score, 'VariableNames', {'feature', 'gini_decrease'});
ranking.selected_for_model = ismember(ranking.feature, params.selectedFeatures);
end

function gain = gini_decrease(x, y)
% Best one-split Gini decrease. Missing values are left out of the split.
ok = ~isnan(x) & ~isnan(y);
x = x(ok);
y = y(ok);
parent = gini(y);
if numel(y) < 20
    gain = 0;
    return
end
edges = quantile(x, 0.05:0.05:0.95);
edges = unique(edges);
gain = 0;
for e = edges
    left = y(x <= e);
    right = y(x > e);
    if numel(left) < 10 || numel(right) < 10
        continue
    end
    child = (numel(left) * gini(left) + numel(right) * gini(right)) / numel(y);
    gain = max(gain, parent - child);
end
end

function g = gini(y)
if isempty(y)
    g = 0;
    return
end
p = histcounts(y, -0.5:1:2.5);
p = p(p > 0) / numel(y);
g = 1 - sum(p .^ 2);
end
