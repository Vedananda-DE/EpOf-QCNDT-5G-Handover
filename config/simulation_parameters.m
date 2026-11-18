function params = simulation_parameters()
% Simulation parameters taken from the EpOf-QCNDT handover paper.
% Site coordinates are not given in the paper; they are fitted to the
% logged routes by k-means with the seed below.

root = fileparts(fileparts(mfilename('fullpath')));
params.root = root;
params.dataRoot = fullfile(root, 'dataset', 'input', '5G-production-dataset');
params.labeledDir = fullfile(root, 'dataset', 'processed', 'labeled_data');
params.normDir = fullfile(root, 'dataset', 'processed', 'normalized_data');
params.analysisDir = fullfile(root, 'dataset', 'dataset_analysis');

params.seed = 42;
params.homDb = 2.0;
params.rsrpKeepDbm = -100;
params.sinrMinDb = -5.0;
params.speedMacroKmh = 30;
params.pingPongSec = 2.0;
params.gapResetSec = 5.0;
params.noiseDbm = -94.0;
params.d0 = 1.0;
params.dMin = 10.0;
params.earthRadius = 6371000;

params.tiers(1) = tier('macro', 6, 1500, 900e6, 2.1, 36, 0);
params.tiers(2) = tier('bs',    7,  500, 2.4e9, 3.0, 22, 0);
params.tiers(3) = tier('femto', 5,  100,  18e9, 4.0, 20, 0);

params.selectedFeatures = {'RSRP', 'delta_RSRP', 'SNR', 'Speed'};
params.treeMaxDepth = 5;
params.trainFraction = 0.70;
params.valFraction = 0.15;
params.testFraction = 0.15;

params.classNames = {'Maintain Current Link', ...
                     'Handover to Macro BS', ...
                     'Handover to Small Cell'};

params.populationSize = 50;
params.maxIterations = 200;
params.hiddenNodes = 16;
params.nClasses = 3;
params.nQubits = 4;
params.parameterCount = 199;
params.quantumLower = 0;
params.quantumUpper = 2 * pi;
params.weightLower = -1;
params.weightUpper = 1;
params.resultsDir = fullfile(root, 'results');
params.baselineEpochs = 30;
params.learningRate = 0.05;
params.boostingRounds = 100;
params.nodeCounts = [50 100 150 200];
params.roundCounts = [100 1000 2000];
end

function s = tier(name, count, radius, frequency, alpha, attenuation, txPower)
s.name = name;
s.count = count;
s.radius = radius;
s.frequency = frequency;
s.alpha = alpha;
s.attenuation = attenuation;
s.txPower = txPower;
end
