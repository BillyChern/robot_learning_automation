
EPISODE 26 vs 25 (17:48): ep 25 (insertion reached its end) drifted dt +0.002..+0.003; ep 26 jammed at n-OFF -0.045..-0.056
(3 force stops, no push) after the learned t-residual drove dt to +0.0075 (policy mean dt +0.48..+0.77, at the bound
every step). The alignment you do right before the insertion is good to ~1-2 mm; the learned t correction then pulls
the drawer 7 mm off it. Consider holding t (no t residual, or a much smaller bound) during the insertion once aligned,
like yaw, and letting the policy learn only advance and small z.
