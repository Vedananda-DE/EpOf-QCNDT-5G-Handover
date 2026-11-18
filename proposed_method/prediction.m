function labels = prediction(theta, X)
% Predicted handover class is the score with the largest value.

scores = proposed_algorithm(X, theta);
[~, idx] = max(scores, [], 2);
labels = idx - 1;
end
