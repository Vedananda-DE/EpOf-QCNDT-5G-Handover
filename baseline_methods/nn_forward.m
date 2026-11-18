function labels = nn_forward(model, X)
h = X;
for k = 1:numel(model.W)
    s = h * model.W{k}.' + model.B{k}.';
    if k < numel(model.W)
        h = tanh(s);
    else
        h = s;
    end
end
[~, idx] = max(h, [], 2);
labels = idx - 1;
end
