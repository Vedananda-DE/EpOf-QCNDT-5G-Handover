function labeled = generate_labels(data, params)
% Serving-cell handover labels.
%   0 maintain the current link
%   1 hand over to a macro cell
%   2 hand over to a small cell
%
% The trace reports the serving cell only. A link is kept when RSRP is at
% least -100 dBm, SNR is at least -5 dB and RSRP has not dropped by more
% than the 2 dB hysteresis. Otherwise the target tier follows speed:
% 30 km/h and above goes to a macro cell, and a slower user goes to a
% small cell. Each new decision is held for the 2 s ping-pong window.

work = data(data.valid, :);
n = height(work);
raw = zeros(n, 1);
keep = work.RSRP >= params.rsrpKeepDbm & work.SNR >= params.sinrMinDb ...
    & work.delta_RSRP > -params.homDb;
raw(~keep & work.Speed >= params.speedMacroKmh) = 1;
raw(~keep & work.Speed < params.speedMacroKmh) = 2;

labels = zeros(n, 1);
sessions = unique(work.session_id, 'stable');
for s = 1:numel(sessions)
    rows = find(work.session_id == sessions(s));
    stamp = datetime(string(work.Timestamp(rows)), 'InputFormat', 'yyyy.MM.dd_HH.mm.ss');
    committed = raw(rows(1));
    since = stamp(1);
    for i = 1:numel(rows)
        proposal = raw(rows(i));
        if i == 1 || isnat(stamp(i)) || isnat(since)
            committed = proposal;
            since = stamp(i);
        elseif seconds(stamp(i) - stamp(max(i - 1, 1))) > params.gapResetSec
            committed = proposal;
            since = stamp(i);
        elseif proposal ~= committed && seconds(stamp(i) - since) >= params.pingPongSec
            committed = proposal;
            since = stamp(i);
        end
        labels(rows(i)) = committed;
    end
end

labeled = work;
labeled.HO_label = labels;
labeled.HO_class = params.classNames(labels + 1).';
end
