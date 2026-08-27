# Vibration Monitoring & Bearing Wear (Condition-Based Maintenance)

## Why trend vibration

Vibration is the earliest and most reliable indicator of rotating-machinery health. Bearing wear,
imbalance, misalignment, and looseness all raise vibration long before they produce audible noise or
a hard failure. The right practice is condition-based: trend the overall vibration velocity (mm/s
RMS) per machine over time and act when the trend rises, rather than waiting for a fixed threshold or
for the machine to fail. A steadily climbing trend is a stronger signal than any single reading.

## Bearing-wear signature

A degrading bearing produces a characteristic signature: the overall vibration level drifts upward
over days or weeks, with increasingly frequent short spikes, and the temperature of the housing
creeps up. This "slow climb with growing spikes" is the classic bearing-wear signature. On a machine
under condition monitoring, a rolling-window baseline (for example a 60-minute rolling mean and
standard deviation) lets you flag readings that deviate from a machine's own recent normal — catching
drift while the absolute value is still within the spec-sheet control limit, which a fixed threshold
would miss.

## Alarm thresholds vs. data-driven anomalies

Two complementary checks are used together. Fixed control limits (from the machine's spec sheet, e.g.
7 mm/s for a class of machine) alarm on any hard breach. A data-driven rolling z-score — how many
standard deviations a reading sits from the machine's own recent rolling mean — catches early drift
and spikes that are still inside the fixed limit. A reading flagged by the z-score but not yet past
the control limit is exactly the early warning that lets maintenance be scheduled before a breakdown.

## Responding to a rising trend

When a machine's vibration trend crosses its alarm level: first rule out the cheap causes —
imbalance (rebalance the wheel or tool), looseness (check mounting bolts), and misalignment. If the
trend persists with those ruled out, schedule a bearing inspection and replacement at the next
planned downtime rather than running to failure. Replacing a bearing on a planned stop costs a
fraction of the unplanned breakdown, the collateral damage, and the lost production of running it to
destruction.

## What to record

For each vibration alarm, record the machine ID, timestamp, the overall velocity level, the rolling
baseline at the time, and the action taken. This history is what turns reactive firefighting into a
predictive-maintenance program and lets you tune the alarm thresholds per machine class.
