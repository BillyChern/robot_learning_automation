
THE USER'S RULE (00:30), URGENT, top priority: "the robot has to make sure that the drawer is STRAIGHT out, then it can
push it in." So:
1. Before the drawer enters the slot, make it STRAIGHT: its axis exactly along the slot normal (front_normal, yaw 0
   relative to the slot, not the +9 deg ramp of prior v5) and its centre line on the slot centre (t of the slot window
   centre), at the right height. Measure the held drawer's actual yaw and offset against the cabinet front first
   (wrist camera looking at the drawer's rim and the slot edges, or the side image), correct it, and re-measure until
   it is within ~1-2 deg and ~2-3 mm.
2. Only then push it in STRAIGHT along the normal, small steps, no yaw changes during the insertion; the learned part
   corrects only small residual t/z (and stops on force).
3. A jam means it was not straight: back out 1-2 cm, re-align, try again (do not twist it in).
Time is very short: change the prior to this now and test on the robot immediately.
