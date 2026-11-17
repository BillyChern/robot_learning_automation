Operator (19:17): robot library change. RobotSession now waits until its watchdog reports "watchdog up"
before anything moves (one restart, else Abort "watchdog did not start" with no motion), and the first arm connection
retries for about 10 s. If you see "watchdog did not start", it is a connection problem, not your program.
