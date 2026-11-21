Operator note (sim worker's diagnosis of attempt 112, your first sim-policy run): no sign/frame mismatch: your
sim_insert matches the sim's conventions (t = (-N1, N0), yaw CCW about +z, force in the base frame, f0 zeroing all OK).
The drift was the policy's jam-recovery search against a FIXED-DEPTH jam the sim does not contain: steps 0-6 went
straight in 10 mm/step, the first 40 N stop came at tool n-OFF -48 mm (drawer back ~6-7 cm into the slot), then it
cycled back/sideways/push and jammed again at -50..-67 mm until it hit its t/yaw bounds. Your episodes have hit this same
depth since about attempt 44. Most likely cause: a VERTICAL wedge: the tool was 9.4 cm above the table with a
6.3-6.6 cm drawer and the opening is ~1.5-9.5 cm above the table, so the drawer top was ~1 mm under the rail (and the
floor lip is ~1.5 cm). Please:
1. Carry/insert height: drawer top >= 5 mm under the rail, bottom >= 3 mm above the slot floor (check against
   slot_h and the measured drawer height); log the drawer's top/bottom heights at the insertion.
2. With the sim arm: right after building Insertion set ins.T_BOUND = 0.008 and ins.YAW_BOUND = 3.0 for the
   table-aligned start; after two force stops within 5 mm of the same depth, stop calling the policy and hand over;
   log ins.obs and the step every decision (policy.py now keeps ins.obs; keep all events, not just the last 10).
3. If the jam stays at the same depth with a good height, look at the wrist/side images at that moment: something at
   ~6.5 cm depth (the shelf rail, a stop, the floor) may be catching; tell me what you see.
