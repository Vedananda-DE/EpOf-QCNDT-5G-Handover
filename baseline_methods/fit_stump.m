function stump = fit_stump(X, target)
% One-split regression stump. Used by the boosting baseline.

best = inf;
stump.feature = 1;
stump.threshold = 0;
stump.left = 0;
stump.right = 0;
for f = 1:size(X, 2)
    cuts = unique(quantile(X(:, f), 0.1:0.1:0.9));
    for c = cuts(:).'
        left = X(:, f) <= c;
        if ~any(left) || all(left)
            continue
        end
        leftValue = mean(target(left));
        rightValue = mean(target(~left));
        pred = repmat(rightValue, size(target));
        pred(left) = leftValue;
        loss = mean((target - pred) .^ 2);
        if loss < best
            best = loss;
            stump.feature = f;
            stump.threshold = c;
            stump.left = leftValue;
            stump.right = rightValue;
        end
    end
end
end
