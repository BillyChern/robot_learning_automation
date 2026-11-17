Operator (15:43): robot library change (rlpj/robot/session.py). The release safeguard now also refuses to
open the gripper outside the task's release_zone (or above 0.20 m) when the wrist image shows the bee on the fingers
(task gripper_check colour at the bottom of the image). Before, it only applied when the reading was in the held
range, and not at all before a program's first gripper command. At 10:10 a DAgger correction opened at 0.27 m with
the bee stuck on the closed fingers (reading 100); the bee fell off the table and the run waited 5.5 h for a person.
If a bee is stuck on the fingers, lower it into the zone, near the table, before opening. "release refused: the
object is on the fingers" means exactly that.
