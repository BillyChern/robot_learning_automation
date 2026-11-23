Operator note 16:41 (user: "be careful not to put the drawer too close to the edge of the table, it might drop off"):
the table target moved: targets.json now asks for about [0.29, -0.02] (between the robot base and the cabinet's near
end). The old one (y -0.22) was at the table's camera-side edge, which runs at y -0.18 (x 0.3) to -0.23 (x 0.6); the
far edge is at y +0.28..+0.43, the robot end at x ~0.05. Rule for every placement and every recovery move: keep the
drawer >= 10 cm inside every table edge (check its outline against these edges before letting go), never slide or
push it toward an edge, and never release it above or near an edge.
