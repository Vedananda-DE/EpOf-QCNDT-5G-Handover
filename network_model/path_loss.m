function [pl, rsrp] = path_loss(distance, site, params)
% Log-distance path loss using the reference attenuation in Table 1.
% PL = A_ref + 10 * alpha * log10(d / d0), RSRP = P_tx - PL.
% Equation (6) in the paper does not use A_ref and collapses the 18 GHz
% femto tier, so the table values are applied in this standard form.

d = max(distance, params.dMin);
pl = site.attenuation + 10 * site.alpha * log10(d / params.d0);
rsrp = site.txPower - pl;
end
