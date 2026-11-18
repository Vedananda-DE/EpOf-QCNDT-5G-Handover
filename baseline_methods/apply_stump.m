function pred = apply_stump(stump, X)
pred = repmat(stump.right, size(X, 1), 1);
pred(X(:, stump.feature) <= stump.threshold) = stump.left;
end
