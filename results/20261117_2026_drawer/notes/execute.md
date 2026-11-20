
RESUMED 2026-11-20 ~15:50 AFTER A LONG PAUSE (lab reboot, the setup was used in between). What changed:
- The side camera was moved and is RE-CALIBRATED (automated: the arm visited 11-12 views of a board; spread 5.4 mm,
  two independent runs agree to 3 mm; table check +1.3 mm).
- The robot base moved relative to the table: the cabinet (on its tape marks) now sits (+8.5, -46.2) mm in the base
  frame. The task yaml fixtures are SHIFTED accordingly: front_offset 0.5133 (was 0.5226), slot_t [-0.166, -0.091]
  (was [-0.120, -0.045]), all keep-out boxes moved by the same offset, reset targets y [-0.210, -0.126]. Read every
  cabinet/slot number from the task yaml (config.load()["task"]["fixtures"]), never from constants in your code
  (e.g. a hard-coded slot centre t = -0.0825 is now wrong: the window centre is -0.1285).
- The drawer is in the bottom slot, pull out, but pushed in ~2 cm past flush (a person put it back).
- Arm at Home; agents run at effort high.
