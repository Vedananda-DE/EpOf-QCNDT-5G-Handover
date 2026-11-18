function result = baseline_fldhdt(rawTest, params)
% Fuzzy handover rule on the raw serving-cell measurements.
% Columns of rawTest are RSRP (dBm), delta RSRP (dB), SNR (dB) and speed (km/h).
% Strong RSRP with a stable link votes to maintain. A falling or weak link
% votes for a macro cell when speed is high and for a small cell otherwise.
% A new decision is held for two samples, the ping-pong window at 1 Hz.

rsrp = rawTest(:, 1);
delta = rawTest(:, 2);
snr = rawTest(:, 3);
speed = rawTest(:, 4);

strong = trapmf(rsrp, [-110, -90, -20, -20]);
weak = trapmf(rsrp, [-160, -160, -110, -90]);
falling = trapmf(-delta, [0, params.homDb, 30, 30]);
poor = trapmf(-snr, [0, -params.sinrMinDb, 40, 40]);
fast = trapmf(speed, [params.speedMacroKmh - 10, params.speedMacroKmh + 10, 140, 140]);

maintain = min(strong, 1 - max(falling, poor));
need = max(weak, max(falling, poor));
macroScore = min(need, fast);
smallScore = min(need, 1 - fast);
[~, idx] = max([maintain, macroScore, smallScore], [], 2);
labels = idx - 1;

held = labels;
for i = 2:numel(labels)
    if labels(i) ~= held(i - 1) && i > 2 && labels(i - 1) ~= held(i - 2)
        held(i) = held(i - 1);
    end
end

result.name = 'FLDHDT';
result.model = [];
result.labels = held;
result.history = [];
end

function m = trapmf(x, abcd)
a = abcd(1); b = abcd(2); c = abcd(3); d = abcd(4);
m = zeros(size(x));
rise = (b > a) & (x > a) & (x < b);
flat = (x >= b) & (x <= c);
fall = (d > c) & (x > c) & (x < d);
m(flat) = 1;
m(rise) = (x(rise) - a) ./ (b - a);
m(fall) = (d - x(fall)) ./ (d - c);
end
