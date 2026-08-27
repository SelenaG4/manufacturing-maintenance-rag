# Fault Code Reference

## LUB-01 — Low lubrication

The central way-lube reservoir is below the minimum level or the lube pump has not built pressure.
Motion is halted for protection. Top up with the specified way oil (ISO VG 68) and confirm the pump
builds pressure; if it does not, check for a pump fault or a broken lube line. Do not bypass this
alarm — running the ways dry causes rapid, expensive wear.

## ATC-04 — Tool changer arm timeout

The automatic tool-changer arm did not complete its cycle in the allowed time. Usual causes: shop air
below 6 bar, a tool not seated in the pocket, chips in a pocket, or a mechanical obstruction. Apply
lockout/tagout before clearing any obstruction in the tool changer.

## TUR-02 — Turret index fault

The lathe turret failed to index or clamp. Check hydraulic pressure, clean chips from the coupling
teeth, and verify the clamp/unclamp solenoid. A turret that indexes but does not clamp firmly must
not be run in production, as it causes tool-position errors.

## HYD-07 — Low hydraulic pressure

Measured hydraulic pressure stayed below the commanded setpoint for more than three cycles. Check
fluid level and external leaks, verify the relief-valve setting with a calibrated gauge, and look for
internal leakage past a cylinder seal or a worn pump. Common on presses that have lost pressing
force.

## VIB-03 — Vibration alarm

A machine's vibration exceeded its alarm level — either a hard breach of the fixed control limit or a
rolling z-score anomaly indicating early drift. Rule out imbalance, looseness, and misalignment
first; if the rising trend persists, schedule a bearing inspection at the next planned downtime.

## SPN-05 — Spindle overtemperature

Spindle temperature exceeded its limit (typically 70 °C). Stop the cycle and idle to cool. Check
spindle coolant flow and the coolant filter; persistent overtemperature after coolant checks points
to bearing wear. Always run the shift-start warm-up program to avoid thermal shock.
