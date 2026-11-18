function metrics = calculate_metrics(yTrue, yPred)
% Accuracy and per-class precision, recall, F1 and specificity.
% Classes are 0, 1 and 2.

yTrue = yTrue(:);
yPred = yPred(:);
n = numel(yTrue);
metrics.accuracy = mean(yTrue == yPred);
metrics.error = 1 - metrics.accuracy;
confusion = zeros(3);
for i = 1:n
    confusion(yTrue(i) + 1, yPred(i) + 1) = confusion(yTrue(i) + 1, yPred(i) + 1) + 1;
end
metrics.confusion = confusion;
metrics.precision = zeros(1, 3);
metrics.recall = zeros(1, 3);
metrics.f1 = zeros(1, 3);
metrics.specificity = zeros(1, 3);
for c = 1:3
    tp = confusion(c, c);
    fp = sum(confusion(:, c)) - tp;
    fn = sum(confusion(c, :)) - tp;
    tn = n - tp - fp - fn;
    metrics.precision(c) = safe_div(tp, tp + fp);
    metrics.recall(c) = safe_div(tp, tp + fn);
    metrics.specificity(c) = safe_div(tn, tn + fp);
    metrics.f1(c) = safe_div(2 * metrics.precision(c) * metrics.recall(c), ...
        metrics.precision(c) + metrics.recall(c));
end
metrics.macroPrecision = mean(metrics.precision);
metrics.macroRecall = mean(metrics.recall);
metrics.macroF1 = mean(metrics.f1);
end

function value = safe_div(num, den)
if den == 0
    value = 0;
else
    value = num / den;
end
end
