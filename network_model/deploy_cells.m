function sites = deploy_cells(data, params)
% Place the Table 1 sites on the measured routes.
% Macro and BS centres follow the full trace. Femto centres follow static traces.

lat = data.Latitude(data.valid);
lon = data.Longitude(data.valid);
lat0 = median(lat);
lon0 = median(lon);
xy = local_xy(lat, lon, lat0, lon0, params.earthRadius);

rng(params.seed);
take = randperm(size(xy, 1), min(40000, size(xy, 1)));
sample = xy(take, :);
static = data.mobility(data.valid) == "Static";
staticXy = xy(static, :);
if isempty(staticXy)
    staticXy = sample;
elseif size(staticXy, 1) > 20000
    pick = randperm(size(staticXy, 1), 20000);
    staticXy = staticXy(pick, :);
end

occupied = zeros(0, 2);
names = {};
tiers = {};
radius = [];
frequency = [];
alpha = [];
attenuation = [];
txPower = [];
latitude = [];
longitude = [];

for t = 1:numel(params.tiers)
    tier = params.tiers(t);
    if strcmp(tier.name, 'femto')
        pool = staticXy;
    else
        pool = sample;
    end
    centres = simple_kmeans(pool, tier.count, params.seed);
    for j = 1:size(centres, 1)
        point = centres(j, :);
        for u = 1:size(occupied, 1)
            shift = point - occupied(u, :);
            dist = hypot(shift(1), shift(2));
            if dist < 80
                if dist < 1
                    direction = [0 1];
                else
                    direction = shift / dist;
                end
                point = occupied(u, :) + 80 * direction;
            end
        end
        occupied = [occupied; point]; %#ok<AGROW>
        prefix = 'M';
        if strcmp(tier.name, 'bs')
            prefix = 'B';
        elseif strcmp(tier.name, 'femto')
            prefix = 'F';
        end
        names{end + 1, 1} = sprintf('%s%d', prefix, j); %#ok<AGROW>
        tiers{end + 1, 1} = tier.name; %#ok<AGROW>
        radius(end + 1, 1) = tier.radius; %#ok<AGROW>
        frequency(end + 1, 1) = tier.frequency; %#ok<AGROW>
        alpha(end + 1, 1) = tier.alpha; %#ok<AGROW>
        attenuation(end + 1, 1) = tier.attenuation; %#ok<AGROW>
        txPower(end + 1, 1) = tier.txPower; %#ok<AGROW>
        [siteLat, siteLon] = xy_to_ll(point, lat0, lon0, params.earthRadius);
        latitude(end + 1, 1) = siteLat; %#ok<AGROW>
        longitude(end + 1, 1) = siteLon; %#ok<AGROW>
    end
end

sites = table(names, tiers, radius, frequency, alpha, attenuation, txPower, latitude, longitude, ...
    'VariableNames', {'cell_name','tier','radius_m','frequency_hz','alpha', ...
    'reference_attenuation_db','tx_power_dbm','Latitude','Longitude'});
end

function xy = local_xy(lat, lon, lat0, lon0, radius)
xy = [deg2rad(lon - lon0) * radius * cos(deg2rad(lat0)), ...
      deg2rad(lat - lat0) * radius];
end

function [lat, lon] = xy_to_ll(point, lat0, lon0, radius)
lat = lat0 + (point(2) / radius) * 180 / pi;
lon = lon0 + (point(1) / (radius * cos(deg2rad(lat0)))) * 180 / pi;
end

function centres = simple_kmeans(X, k, seed)
rng(seed);
pick = randperm(size(X, 1), k);
centres = X(pick, :);
for iter = 1:30
    dist = zeros(size(X, 1), k);
    for j = 1:k
        dist(:, j) = sum((X - centres(j, :)).^2, 2);
    end
    [~, owner] = min(dist, [], 2);
    updated = centres;
    for j = 1:k
        members = X(owner == j, :);
        if ~isempty(members)
            updated(j, :) = mean(members, 1);
        end
    end
    if max(abs(updated(:) - centres(:))) < 1
        centres = updated;
        break
    end
    centres = updated;
end
end
