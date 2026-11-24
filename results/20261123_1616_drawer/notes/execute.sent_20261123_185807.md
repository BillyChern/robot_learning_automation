Operator note 18:56: episode 5 inserted the drawer (bottom 0.012, flush) but was scored "unsure": the orchestrator's
check and YOUR program's own check both said "the cabinet moved 3.3 cm". That was a measurement bug: with the drawer
in the compartment the cabinet blob's centroid shifts 3-5 cm. The orchestrator now uses the cabinet's outline centre
(rlpj.success.checks.outline_xy(blob, yaw_deg): 2nd-98th percentile midpoints along/across the cabinet axis); use the
same in your program's own verify, so a real insertion is claimed as success.
