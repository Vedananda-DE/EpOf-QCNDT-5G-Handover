# Handover labels and selected features

The production traces do not contain a handover class. Each kept sample is labeled from the serving-cell measurements.

| Label | Decision | Rule |
|---|---|---|
| 0 | Maintain current link | RSRP ≥ −100 dBm, SNR ≥ −5 dB, and delta RSRP > −2 dB |
| 1 | Handover to a macro cell | The maintain test fails and speed ≥ 30 km/h |
| 2 | Handover to a small cell | The maintain test fails and speed < 30 km/h |

`delta_RSRP(t) = RSRP(t) − RSRP(t−1)` inside the same session. The difference is set to zero when the time step is longer than 2 s or either RSRP sample is missing.

A new decision is held for 2 s. That is the paper’s ping-pong window, and it stops the label from flipping when speed sits on the 30 km/h boundary. A gap longer than 5 s starts the decision again.

The thresholds are the paper’s own operating points: the weak/moderate RSRP boundary is −100 dBm, the hysteresis is 2 dB, and the outage floor is −5 dB. The 30 km/h macro/small split is the usual two-tier mobility rule when the log has no neighbour-cell measurements.

Rows with RSRP below −160 dBm or above −20 dBm are removed. RSRP of −200 dBm is the logger’s empty-signal marker. SNR gaps of up to three samples are interpolated inside the session. The original CSV files are not edited.

The four features passed to the decision tree and to the later model are fixed:

`RSRP`, `delta_RSRP`, `SNR`, `Speed`

Gini ranking is computed on the training sessions and saved, but it does not replace this set.

Sessions are split 70% / 15% / 15% inside each mobility and application group, with seed 42. Scaling of the four features uses the training minimum and maximum only.

The six macro, seven BS and five femto sites from Table 1 are placed on the routes and saved in `cell_layout.csv` for the later network simulation. They are not used to assign these labels. A geometric handover fired only on the crossing sample, which left 99.86% of the rows in class 0 and made longitude and latitude the top tree features. The serving-measurement rule above is the one used for the labeled dataset.
