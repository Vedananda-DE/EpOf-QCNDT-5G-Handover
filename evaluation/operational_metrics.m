function stats = operational_metrics(yTrue, yPred)
% Handover success, failure and frequency from the predicted class sequence.
% A predicted handover is successful when it names the same class as the label.
% Frequency is the number of class changes per 100 samples.

yTrue = yTrue(:);
yPred = yPred(:);
handover = yPred > 0;
matched = handover & (yPred == yTrue);
attempts = sum(handover);
stats.successRate = 100 * safe_div(sum(matched), attempts);
stats.failureRate = 100 - stats.successRate;
if numel(yPred) < 2
    changes = 0;
else
    changes = sum(yPred(2:end) ~= yPred(1:end-1));
end
stats.frequencyRate = 100 * safe_div(changes, numel(yPred));
stats.accuracy = mean(yTrue == yPred);
end

function value = safe_div(num, den)
if den == 0
    value = 0;
else
    value = num / den;
end
end
