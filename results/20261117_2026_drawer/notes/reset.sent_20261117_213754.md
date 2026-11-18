Attempt 16's pull was refused by the keep-out box although the grasp itself was right (gripper reading 86.9: the tab
was held). Cause: with the gripper closed, the model puts the fingertips about 1 cm deeper than the real ones, so the
held tab pose was modelled 10.1 mm behind the cabinet front (the limit was 10 mm). The box is now changed (task yaml
`keep_out`): the bottom-slot opening (1.5-9.5 cm above the table, t -0.120..-0.045 along the front) may be entered up
to 3 cm behind the front plane; the rest of the cabinet front keeps the 1 cm limit and the body starts 3 cm behind.
Your attempt-16 pull plan passes the check unchanged now, so re-run the same grasp and pull.
