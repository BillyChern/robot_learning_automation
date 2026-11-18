
Reset 15 (attempt 54) was right: the drawer was upright at the target with the pull facing the robot. The orchestrator
rejected it with "drawer: not found" because its side-depth measurement merged the drawer and the cabinet into one blob
at that target (0.401, -0.03): one blob at (0.578, 0.117), 0.28 x 0.25 m, 0.27 m tall. Not your program's fault. The
layout's y range is now [-0.20, -0.08] (was up to -0.02), so targets stay further from the cabinet.
