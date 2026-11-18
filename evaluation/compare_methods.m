function summary = compare_methods(results, yTrue, mobility)
% Build the comparison table for every trained method.
% mobility uses 0 for static and 1 for driving, on the same rows as yTrue.

n = numel(results);
name = strings(n, 1);
accuracy = zeros(n, 1);
macroF1 = zeros(n, 1);
staticAccuracy = zeros(n, 1);
drivingAccuracy = zeros(n, 1);
successRate = zeros(n, 1);
failureRate = zeros(n, 1);
frequencyRate = zeros(n, 1);
for i = 1:n
    labels = results{i}.labels(:);
    metrics = calculate_metrics(yTrue, labels);
    operational = operational_metrics(yTrue, labels);
    name(i) = string(results{i}.name);
    accuracy(i) = 100 * metrics.accuracy;
    macroF1(i) = 100 * metrics.macroF1;
    successRate(i) = operational.successRate;
    failureRate(i) = operational.failureRate;
    frequencyRate(i) = operational.frequencyRate;
    staticRows = mobility == 0;
    drivingRows = mobility == 1;
    staticAccuracy(i) = 100 * mean(labels(staticRows) == yTrue(staticRows));
    drivingAccuracy(i) = 100 * mean(labels(drivingRows) == yTrue(drivingRows));
end
summary = table(name, accuracy, macroF1, drivingAccuracy, staticAccuracy, ...
    successRate, failureRate, frequencyRate);
end
