function data = clean_data(raw, params)
% Drop empty ping fields, reject impossible RSRP, and build delta RSRP.
% delta_RSRP is the change in the logged serving RSRP over one time step.
% A step longer than the ping-pong window starts again from zero.

dropCols = {'PINGAVG','PINGMIN','PINGMAX','PINGSTDEV','PINGLOSS', ...
            'CELLHEX','NODEHEX','LACHEX','RAWCELLID','NRxRSRP','NRxRSRQ'};
keep = ~ismember(raw.Properties.VariableNames, dropCols);
data = raw(:, keep);

rsrp = double(data.RSRP);
bad = isnan(rsrp) | rsrp < -160 | rsrp > -20;
rsrp(bad) = nan;
data.RSRP = rsrp;

snr = double(data.SNR);
snr = fill_short_gaps(snr, data.session_id, 3);
data.SNR = snr;

data.delta_RSRP = session_delta(data.RSRP, string(data.Timestamp), data.session_id, params.pingPongSec);
data.valid = ~isnan(data.RSRP) & ~isnan(data.SNR) & ~isnan(data.Speed) ...
             & ~isnan(data.Latitude) & ~isnan(data.Longitude);
end

function x = fill_short_gaps(x, session, limit)
% Linear fill of gaps up to `limit` samples, applied inside each session.
ids = unique(session, 'stable');
for i = 1:numel(ids)
    rows = find(session == ids(i));
    x(rows) = fill_vector(x(rows), limit);
end
end

function x = fill_vector(x, limit)
nanIdx = isnan(x);
if ~any(nanIdx) || all(nanIdx)
    return
end
good = find(~nanIdx);
for k = find(nanIdx).'
    left = good(find(good < k, 1, 'last'));
    right = good(find(good > k, 1, 'first'));
    if isempty(left) || isempty(right)
        continue
    end
    if (right - left - 1) <= limit
        w = (k - left) / (right - left);
        x(k) = (1 - w) * x(left) + w * x(right);
    end
end
end

function delta = session_delta(rsrp, timestamp, session, maxStep)
delta = zeros(numel(rsrp), 1);
ids = unique(session, 'stable');
for i = 1:numel(ids)
    rows = find(session == ids(i));
    t = datetime(timestamp(rows), 'InputFormat', 'yyyy.MM.dd_HH.mm.ss');
    for k = 2:numel(rows)
        step = seconds(t(k) - t(k - 1));
        if ~isnan(step) && step <= maxStep && ~isnan(rsrp(rows(k))) && ~isnan(rsrp(rows(k - 1)))
            delta(rows(k)) = rsrp(rows(k)) - rsrp(rows(k - 1));
        end
    end
end
end
