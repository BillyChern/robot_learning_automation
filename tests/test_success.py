"""Offline success tests: geometric checks on rendered before/after scenes, verdict truth table, judge dry run."""
import json

import numpy as np
import pytest

import test_perception as tp
from rlpj.perception import tabletop
from rlpj.success import checks, verdict, vlm_judge

RING, BEE, CUP, TZ = tp.RING, tp.BEE, tp.CUP, tp.TZ
MOVE = (CUP["xy"][0] - 0.15, CUP["xy"][1] + 0.01)


def blobs(objects):
    return tabletop.detect(tp.scene_cloud(objects), table_z=TZ)


def ring(xy=RING["xy"]):
    return tp.ring_pts(xy, RING["r_out"], RING["r_in"], RING["h"])


def bee(xy, sx=0.04, sy=0.03, h=0.02):
    return tp.box_pts(xy, sx, sy, h)


def cup(xy, h=CUP["h"]):
    return tp.cup_pts(xy, CUP["r_top"], CUP["r_rim"], h)


@pytest.fixture(scope="module")
def bee_cfg():
    return tp.task("bee_in_tape_roll")


@pytest.fixture(scope="module")
def cup_cfg():
    return tp.task("cup_down_stacking")


@pytest.fixture(scope="module")
def bee_before():
    return blobs([ring(), bee(BEE["xy"], 0.05, 0.03)])


@pytest.fixture(scope="module")
def cups_before():
    return blobs([cup(CUP["xy"]), cup(MOVE)])


def test_bee_success_cases(bee_cfg, bee_before):
    ok, rep = checks.run(bee_cfg, bee_before, blobs([ring(), bee(RING["xy"])]))
    assert ok is True, rep
    assert rep["ring_moved_m"] < 0.01 and rep["outside_bees"] == []
    ok, rep = checks.run(bee_cfg, bee_before, blobs([ring(), bee(RING["xy"], h=0.045)]))       # taller than the ring
    assert ok is True, rep
    off = (RING["xy"][0] - 0.012, RING["xy"][1] + 0.005)
    ok, rep = checks.run(bee_cfg, bee_before, blobs([ring(), bee(off, 0.025, 0.02, 0.015)]))  # small, off-centre
    assert ok is True, rep


def test_bee_fail_cases(bee_cfg, bee_before):
    ok, rep = checks.run(bee_cfg, bee_before, blobs([ring(), bee((0.36, 0.10), 0.05, 0.03)]))
    assert ok is False and "outside" in rep["reason"]
    moved = (RING["xy"][0] - 0.05, RING["xy"][1] + 0.02)
    ok, rep = checks.run(bee_cfg, bee_before, blobs([ring(moved), bee(moved)]))
    assert ok is False and "moved" in rep["reason"]
    tipped = tp.box_pts(RING["xy"], 0.10, 0.025, 0.10)                                         # roll on its edge
    ok, rep = checks.run(bee_cfg, bee_before, blobs([tipped]))
    assert ok is False and "disturbed" in rep["reason"]


def test_bee_unknown_cases(bee_cfg, bee_before):
    ok, rep = checks.run(bee_cfg, bee_before, [])
    assert ok is None and rep["reason"]
    ok, rep = checks.run(bee_cfg, bee_before, blobs([ring()]))                                 # bee not visible
    assert ok is None and "hidden" in rep["reason"]
    ok, rep = checks.run(bee_cfg, bee_before, blobs([bee((0.36, 0.10))]))                      # ring gone
    assert ok is None and "not found" in rep["reason"]
    ok, rep = checks.run({"name": "nothing"}, [], [])
    assert ok is None


def test_cups_cases(cup_cfg, cups_before):
    ok, rep = checks.run(cup_cfg, cups_before, blobs([cup(CUP["xy"], CUP["h"] + 0.02)]))
    assert ok is True, rep
    assert 14 < rep["stack_minus_base_mm"] < 26
    ok, rep = checks.run(cup_cfg, cups_before, blobs([cup(CUP["xy"]), cup(MOVE)]))
    assert ok is False and "moved cup" in rep["reason"]
    ok, rep = checks.run(cup_cfg, cups_before, blobs([cup(CUP["xy"], CUP["h"] + 0.05)]))
    assert ok is False and "taller" in rep["reason"]
    ok, rep = checks.run(cup_cfg, cups_before, blobs([cup(CUP["xy"])]))
    assert ok is False and "taller" in rep["reason"]
    ok, rep = checks.run(cup_cfg, cups_before, blobs([cup(MOVE)]))
    assert ok is False and "base cup" in rep["reason"]
    ok, rep = checks.run(cup_cfg, cups_before, blobs([cup(CUP["xy"], CUP["h"] + 0.02), cup((0.40, 0.10))]))
    assert ok is False and "another" in rep["reason"]
    ok, rep = checks.run(cup_cfg, blobs([cup(CUP["xy"])]), blobs([cup(CUP["xy"], CUP["h"] + 0.02)]))
    assert ok is None
    ok, rep = checks.run(cup_cfg, cups_before, [])
    assert ok is None
    assert json.dumps(rep)


def test_verdict_truth_table():
    yes, no, low = dict(success=True, confidence=0.9), dict(success=False, confidence=0.9), dict(success=True, confidence=0.2)
    cases = [
        ("success", True, yes, "success"), ("success", True, None, "success"), ("success", None, yes, "success"),
        ("success", None, low, "unsure"), ("success", None, None, "unsure"), ("success", True, no, "unsure"),
        ("success", False, yes, "unsure"), ("success", False, None, "fail"), ("success", None, no, "fail"),
        ("success", False, no, "fail"),
        ("fail", True, yes, "unsure"), ("fail", None, yes, "unsure"), ("fail", True, no, "fail"),
        ("fail", None, None, "fail"), ("fail", False, None, "fail"), ("fail", False, no, "fail"),
        (None, True, yes, "unsure"), (None, None, None, "fail"),
    ]
    for claim, p, v, want in cases:
        out = verdict.decide(claim, p, v)
        assert out["final"] == want, (claim, p, v, out)
        assert out["reason"] and out["votes"]["claim"] == claim
        assert verdict.needs_operator(out) == (want == "unsure")


def test_vlm_judge_dry_and_parse(bee_cfg):
    res = vlm_judge.judge({"before_side": "/x/before_side.jpg", "after_side": "/x/after_side.png"}, bee_cfg, dry=True,
                          host="h", model="m", effort="low", extra="Note: the bee is yellow.")
    assert res["success"] is None and res["reason"] == "dry run"
    p = res["prompt"]
    assert "before_side.jpg" in p and "after_side.png" in p and bee_cfg["success"].strip() in p and "yellow" in p
    c = res["command"]
    assert c.startswith("export PATH=") and "claude -p --model m --effort low --output-format json" in c
    assert "--json-schema" in c and "--tools Read" in c and res["remote"] in c and "rm -rf" in c
    good = json.dumps(dict(structured_output=dict(success=True, confidence=0.85, reason="bee in the roll"),
                           total_cost_usd=0.0123, usage={"input_tokens": 10}, is_error=False))
    out = vlm_judge.parse("noise\n" + good)
    assert out["success"] is True and out["confidence"] == 0.85 and out["cost_usd"] == 0.0123
    bad = vlm_judge.parse(json.dumps(dict(is_error=True, result="quota", total_cost_usd=0)))
    assert bad["success"] is None and "quota" in bad["reason"]
    assert vlm_judge.parse("not json")["success"] is None
    assert vlm_judge.main(["/x/before_side.jpg", "/x/after_side.jpg", "--dry", "--task", "bee_in_tape_roll"]) == 0
