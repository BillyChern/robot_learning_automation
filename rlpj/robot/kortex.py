"""Kortex access: TCP for Base commands, UDP for BaseCyclic feedback.
Commands are refused unless RLPJ_MOTION=1, which only the orchestrator sets, and only for real runs."""
import os
import threading
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2

from . import calib


def motion_allowed():
    return os.environ.get("RLPJ_MOTION") == "1"


class _Conn:
    def __init__(self, ip, port, udp=False):
        self.transport = UDPTransport() if udp else TCPTransport()
        self.router = RouterClient(self.transport, lambda e: print("kortex router error:", e))
        self.transport.connect(ip, port)
        si = Session_pb2.CreateSessionInfo()
        si.username = "admin"
        si.password = "admin"
        si.session_inactivity_timeout = 60000
        si.connection_inactivity_timeout = 2000
        self.sm = SessionManager(self.router)
        self.sm.CreateSession(si)

    def close(self):
        for f in (self.sm.CloseSession, self.transport.disconnect):
            try:
                f()
            except Exception:
                pass


class Robot:
    def __init__(self, ip=None, dry=False):
        self.ip = ip or calib.ip()
        self.dry = dry or not motion_allowed()
        self.tcp = _Conn(self.ip, 10000)
        self.udp = _Conn(self.ip, 10001, udp=True)
        self.base = BaseClient(self.tcp.router)
        self.cyc = BaseCyclicClient(self.udp.router)
        # a fresh cyclic (UDP) session can refuse its first requests (INVALID_USER_SESSION_ACCESS) for seconds: 26 robot
        # runs failed to start within 2 s (2026-11-09). Retry for ~10 s and open a new UDP session halfway.
        for k in range(25):
            try:
                self.cyc.RefreshFeedback()
                break
            except Exception:
                if k == 24:
                    raise
                if k == 12:
                    self.udp.close()
                    self.udp = _Conn(self.ip, 10001, udp=True)
                    self.cyc = BaseCyclicClient(self.udp.router)
                time.sleep(0.4)
        self.lock = threading.Lock()      # serialises TCP calls between threads
        self.fb_lock = threading.Lock()
        self.gripper_cmd = float("nan")   # last commanded gripper value (0 open .. 1 closed)

    def close(self):
        self.udp.close()
        self.tcp.close()

    def keepalive(self):
        try:
            self.arm_state()
        except Exception:
            pass

    # ---------- reading ----------
    def feedback(self):
        with self.fb_lock:
            fb = self.cyc.RefreshFeedback()
        b = fb.base
        try:
            g = fb.interconnect.gripper_feedback.motor[0].position
        except Exception:
            g = float("nan")
        return dict(q=np.array([a.position for a in fb.actuators]), dq=np.array([a.velocity for a in fb.actuators]),
                    pose=np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z,
                                   b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z]),
                    gripper=float(g), t=time.time(),
                    state=Base_pb2.ArmState.Name(b.active_state) if b.active_state else "")

    def arm_state(self):
        with self.lock:
            s = self.base.GetArmState()
        return Base_pb2.ArmState.Name(s.active_state)

    def fk_oracle(self, q_deg):
        """Read-only: the controller's own FK. Rejects some joint sets with ERROR_INTERNAL."""
        ja = Base_pb2.JointAngles()
        for i, v in enumerate(q_deg):
            j = ja.joint_angles.add()
            j.joint_identifier = i
            j.value = float(v)
        with self.lock:
            p = self.base.ComputeForwardKinematics(ja)
        return np.array([p.x, p.y, p.z, p.theta_x, p.theta_y, p.theta_z])

    # ---------- commanding ----------
    def _cmd_ok(self):
        if self.dry:
            raise RuntimeError("motion is not allowed in this run (dry run, or RLPJ_MOTION is not 1)")

    def send_joint_speeds(self, v_deg):
        self._cmd_ok()
        js = Base_pb2.JointSpeeds()
        for i, v in enumerate(v_deg):
            s = js.joint_speeds.add()
            s.joint_identifier = i
            s.value = float(v)
            s.duration = 0
        with self.lock:
            self.base.SendJointSpeedsCommand(js)

    def stop(self):
        self._cmd_ok()
        with self.lock:
            self.base.Stop()

    def gripper_position(self, value):
        """value 0 (open) .. 1 (closed). Any later arm command cancels a gripper motion still running."""
        self._cmd_ok()
        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION
        f = cmd.gripper.finger.add()
        f.finger_identifier = 1
        f.value = float(np.clip(value, 0, 1))
        with self.lock:
            self.base.SendGripperCommand(cmd)
        self.gripper_cmd = float(f.value)
