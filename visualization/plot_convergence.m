function plot_convergence(history, outDir)
% history is maxIter by 2: training accuracy and validation accuracy.

if ~exist(outDir, 'dir')
    mkdir(outDir);
end
fig = figure('Visible', 'off');
plot(history(:, 1), 'LineWidth', 1.2);
hold on;
plot(history(:, 2), 'LineWidth', 1.2);
xlabel('Iteration');
ylabel('Accuracy');
title('QCNDT search');
legend({'Training', 'Validation'}, 'Location', 'best');
grid on;
saveas(fig, fullfile(outDir, 'qcndt_search_convergence.png'));
savefig(fig, fullfile(outDir, 'qcndt_search_convergence.fig'));
close(fig);
end
