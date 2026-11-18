function raw = load_dataset(params)
% Read every production trace. The original files are left unchanged.

files = dir(fullfile(params.dataRoot, '**', '*.csv'));
files = files(~startsWith({files.name}, '._'));

n = numel(files);
sessions = cell(n, 1);
for k = 1:n
    path = fullfile(files(k).folder, files(k).name);
    rel = erase(path, [params.dataRoot filesep]);
    rel = strrep(rel, '\', '/');
    parts = strsplit(rel, '/');
    T = readtable(path, 'TextType', 'string');
    T.session_id = repmat(string(erase(rel, '.csv')), height(T), 1);
    T.application = repmat(string(parts{1}), height(T), 1);
    if any(strcmp(parts, 'Driving'))
        mobility = "Driving";
    else
        mobility = "Static";
    end
    T.mobility = repmat(mobility, height(T), 1);
    sessions{k} = T;
end

[~, order] = sort(cellfun(@(t) t.session_id(1), sessions));
raw = vertcat(sessions{order});
raw = assign_session_split(raw, params.seed);
end

function raw = assign_session_split(raw, seed)
ids = unique(raw.session_id, 'stable');
mobility = strings(numel(ids), 1);
application = strings(numel(ids), 1);
for i = 1:numel(ids)
    row = find(raw.session_id == ids(i), 1, 'first');
    mobility(i) = raw.mobility(row);
    application(i) = raw.application(row);
end

keys = mobility + "|" + application;
groups = unique(keys, 'stable');
split = strings(numel(ids), 1);
rng(seed);
for g = 1:numel(groups)
    idx = find(keys == groups(g));
    idx = idx(randperm(numel(idx)));
    n = numel(idx);
    nVal = 0;
    nTest = 0;
    if n >= 3
        nVal = max(1, floor(n * 0.15));
        nTest = max(1, floor(n * 0.15));
    elseif n == 2
        nTest = 1;
    end
    if nVal + nTest >= n
        nVal = double(n >= 4);
        nTest = double(n >= 3);
    end
    nTrain = n - nVal - nTest;
    split(idx(1:nTrain)) = "train";
    split(idx(nTrain + 1:nTrain + nVal)) = "val";
    split(idx(nTrain + nVal + 1:end)) = "test";
end

raw.split = strings(height(raw), 1);
for i = 1:numel(ids)
    raw.split(raw.session_id == ids(i)) = split(i);
end
end
