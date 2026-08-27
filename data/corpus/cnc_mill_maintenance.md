# CNC Milling Machine — Maintenance & Troubleshooting

## Spindle overheating

If the spindle temperature exceeds 70 °C during normal operation, stop the cycle and let the
spindle idle at low RPM to cool. The most common causes are insufficient spindle coolant flow, a
clogged coolant filter, worn spindle bearings, or running at high RPM for extended periods without
adequate warm-up. Check the coolant reservoir level and the filter first; a warm-up program of 10
minutes at gradually increasing RPM should be run at the start of every shift. Persistent overheating
after coolant checks usually points to bearing wear and should be escalated to a bearing inspection.

## Coolant system

The flood-coolant system requires the reservoir to be kept above the minimum line and the
concentration checked weekly with a refractometer (target 6–8% for general machining). Low
concentration causes rust and poor tool life; high concentration causes residue and foaming. Clean
the coolant filter weekly and replace it monthly. Symptoms of a coolant problem include poor surface
finish, smoking at the cut, rapid tool wear, and rust spots on the table.

## Way lubrication

The linear ways and ballscrews are lubricated automatically by the central lube pump. Check the lube
reservoir daily and top up with the specified way oil (ISO VG 68). A low-lube alarm (see fault code
LUB-01) halts motion; do not bypass it. Symptoms of inadequate way lubrication include stick-slip
motion, position error, and audible squealing during rapid moves. Never substitute hydraulic oil for
way oil — it lacks the tackiness additives that keep the film on vertical ways.

## Tool changer faults

An automatic tool changer (ATC) that fails to complete a change usually reports fault code ATC-04
(arm timeout). Common causes are low air pressure (the ATC needs 6 bar minimum), a tool not seated
correctly in the pocket, or a mechanical obstruction. Verify shop air pressure at the machine
regulator, check that all pockets are clean and empty of chips, and confirm the tool holders are the
correct pull-stud type. Never reach into the ATC without applying lockout/tagout first.

## Axis position error / servo alarm

A position (following) error alarm means an axis could not keep up with its commanded position.
Causes range from mechanical (way lube failure, chip buildup, a crashed axis) to electrical (servo
drive fault, encoder problem). First check for obvious obstructions and confirm way lubrication.
Repeated following errors on one axis under light load suggest an encoder or drive issue and require
a service technician. Do not increase the following-error tolerance to silence the alarm.
