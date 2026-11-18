function P = softmax_matrix(S)
% Row-wise softmax, stabilized by the row maximum.

shift = S - max(S, [], 2);
E = exp(shift);
P = E ./ sum(E, 2);
end
