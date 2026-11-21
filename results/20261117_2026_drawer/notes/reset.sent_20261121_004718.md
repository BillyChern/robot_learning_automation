Operator note 00:38 (attempt 163): all 4 picks missed again at the side-camera point. Each wrist footprint was partial
(L 0.06-0.09, at y ~0.18 while the side camera said ~0.10-0.11) and was thrown away as "implausible"; no re-centring
happened. The side camera's table positions are biased: wrist - side = (+0.010, +0.057) (+0.013, +0.070) (0.000, +0.046)
(-0.019, +0.056) (+0.010, +0.052) (+0.017, +0.070) in attempts 150-158, and 163's partial footprints agree (+y ~6-8 cm).
Do this:
1. Start every table look-down at the side-camera centre + (0.010, 0.058).
2. A partial footprint (touching the image edge, L < 0.10) is not implausible: move the look-down to its centre and
   re-image (up to 3 times) until the whole outline is inside the view; then grasp from the wrist outline only.
3. Fall back to the side-camera point only with the bias added, never raw.
Operator note 00:50 (library): the force stop fired 4 times tonight at 65-67 N on approach moves (pinch approach x2,
down_slide, and a 61.9 N escape). You can now lower the limit for a single move: set `s.executor.force_limit = 20`
(or 25) before an approach or press into an unknown contact and restore it after (`s.executor.force_limit =
s.executor.force_cap`); a contact then stops at ~20 N instead of ~65 N. You cannot raise it above the task's 65 N cap.
Use the low limit for every final approach toward the drawer, knob or table; keep 65 N only for the deliberate
insertion push and the pull-out of a stuck drawer.
Operator note 00:48 (attempts 165, 166: identical force stops 66.2 / 65.3 N on down_slide): "wrist look: drawer at
[0.524, -0.18] (y -0.235..-0.140)", then "low push at [0.524, -0.225]": the push point is INSIDE the drawer's y span,
1 cm in from its far edge, so the fingers come down on top of the wall. Put the pusher outside the edge it pushes:
y < -0.235 - (finger half-width + 1.5 cm), i.e. about -0.27, check that point is reachable, come down with
`s.executor.force_limit = 20`, and repeat the wrist look after the slide.
