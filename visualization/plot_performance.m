function plot_performance(summary, outDir)
% Success, failure and frequency for every method.

if ~exist(outDir, 'dir')
    mkdir(outDir);
end
fig = figure('Visible', 'off');
bar([summary.successRate, summary.failureRate, summary.frequencyRate]);
set(gca, 'XTick', 1:height(summary), 'XTickLabel', summary.name, 'XTickLabelRotation', 30);
ylabel('Percent');
xlabel('Method');
title('Handover operational metrics');
legend({'Success rate', 'Failure rate', 'Frequency rate'}, 'Location', 'best');
grid on;
saveas(fig, fullfile(outDir, 'operational_metrics.png'));
savefig(fig, fullfile(outDir, 'operational_metrics.fig'));
close(fig);
end
