The contact-force stop is removed (user's decision, 22:30): task yaml `rl.force_limit_n` is now null, so no motion segment
stops for "contact force ... > 25 N" any more. Drawer pulls and pushes needed 22-28 N and kept being stopped. The peak
force change of every segment is still logged (`max_force_n` in the exec lines), and `s.feedback()["force"]` still
reads the estimate. The other checks stay: model check, keep-in and keep-out boxes, speed limits, tracking deviation,
and the arm's own Kinova protections. Push and pull slowly; get the drawer in and out.
