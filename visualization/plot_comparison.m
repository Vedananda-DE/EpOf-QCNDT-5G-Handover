function plot_comparison(summary, outDir)
% Bar chart of test accuracy for every method.

if ~exist(outDir, 'dir')
    mkdir(outDir);
end
fig = figure('Visible', 'off');
bar(summary.accuracy);
set(gca, 'XTick', 1:height(summary), 'XTickLabel', summary.name, 'XTickLabelRotation', 30);
ylabel('Accuracy (%)');
xlabel('Method');
title('Handover classification accuracy');
ylim([0 100]);
grid on;
saveas(fig, fullfile(outDir, 'accuracy_comparison.png'));
savefig(fig, fullfile(outDir, 'accuracy_comparison.fig'));
close(fig);
end
