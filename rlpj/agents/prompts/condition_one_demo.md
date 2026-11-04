## Task specification: one human demonstration
One RGB-D recording of a person doing the task once is at `{WORKSPACE}/inputs/demo` (-> `{DEMO}`), read-only:
- `meta.json`: cameras (name, serial, intrinsics, `depth_scale`), fps, frame counts.
- `<cam>_color.mp4` (1280x720), `<cam>_depth.mkv` (FFV1, uint16; metres = value x `depth_scale`, aligned to colour),
  `<cam>_timestamps.csv`, for the cameras `side_left` and `front`.
- `side_left` is the same camera that the robot program sees through `s.side` (serial {SIDE_SERIAL}), at about the same
  pose: `inputs/calibration/side_pose_{SIDE_SERIAL}.json` maps its frame into the robot base. `front` is a static
  camera on the far side of the table facing the demonstrator.
Derive the task from what `side_left` saw, by geometry (which object moved, from where to where, relative to the other
object; the final arrangement), never by colour, with `cd {ROOT} && uv run python <your script>` on this machine
(cv2 decodes both videos; sample frames, do not read everything). Verify your rule numerically on the recording and
write it down in `notes/`. The goal sentence above is the rule's name, not its substitute: the numbers (insertion
depth, clearances, where the object ends up) come from the demonstration.
