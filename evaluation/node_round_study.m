function study = node_round_study(yTrue, yPred, params)
% Aggregate one predicted trace over the paper's node counts and round counts.
% Each node is one contiguous slice of the test sequence. A round count keeps
% that many decisions from the start of every slice. This uses the recorded
% predictions. It does not invent a second radio network.

yTrue = yTrue(:);
yPred = yPred(:);
rows = [];
for nodes = params.nodeCounts
    cuts = round(linspace(1, numel(yPred) + 1, nodes + 1));
    for rounds = params.roundCounts
        trueParts = [];
        predParts = [];
        for k = 1:nodes
            sliceTrue = yTrue(cuts(k):cuts(k + 1) - 1);
            slicePred = yPred(cuts(k):cuts(k + 1) - 1);
            take = min(rounds, numel(slicePred));
            trueParts = [trueParts; sliceTrue(1:take)]; %#ok<AGROW>
            predParts = [predParts; slicePred(1:take)]; %#ok<AGROW>
        end
        operational = operational_metrics(trueParts, predParts);
        rows = [rows; nodes, rounds, 100 * operational.accuracy, ...
            operational.successRate, operational.failureRate, operational.frequencyRate]; %#ok<AGROW>
    end
end
study = array2table(rows, 'VariableNames', ...
    {'nodes', 'rounds', 'accuracy', 'successRate', 'failureRate', 'frequencyRate'});
end
