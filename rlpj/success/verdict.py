"""Final outcome from the program's claim, the geometric check and the VLM judge (DESIGN.md section 8)."""

VLM_MIN_CONFIDENCE = 0.5   # a VLM vote below this confidence counts as no vote


def decide(claim, perception, vlm):
    """{'final': 'success'|'fail'|'unsure', 'reason', 'votes'}: success needs the program's claim plus at least one
    independent yes (perception True or a confident VLM yes) and no independent no; conflicting or missing
    independent evidence on a claimed success is unsure, as is independent evidence against a claimed fail. claim None
    (a policy rollout, which has no verification of its own): success needs both independent checks to say yes; one
    yes without the other is unsure; otherwise fail."""
    v = None
    if vlm and vlm.get("success") is not None and float(vlm.get("confidence") or 0) >= VLM_MIN_CONFIDENCE:
        v = bool(vlm["success"])
    p = None if perception is None else bool(perception)
    yes, no = (p is True) or (v is True), (p is False) or (v is False)
    claimed = claim == "success"
    votes = dict(claim=claim, perception=p, vlm=v)
    if claim is None:
        final = "success" if (p is True and v is True) else ("unsure" if yes else "fail")
    elif claimed and yes and not no:
        final = "success"
    elif claimed and no and not yes:
        final = "fail"
    elif claimed:
        final = "unsure"
    elif yes and not no:
        final = "unsure"
    else:
        final = "fail"
    txt = ", ".join(f"{k}={'none' if x is None else x}" for k, x in votes.items())
    return dict(final=final, reason=f"{_explain(final, None if claim is None else claimed, yes, no)} ({txt})", votes=votes)


def needs_operator(verdict):
    """True when the verdict must be confirmed by the operator."""
    return verdict.get("final") == "unsure"


def _explain(final, claimed, yes, no):
    if claimed is None:
        return {"success": "both independent checks say success", "unsure": "the independent checks disagree or one is missing",
                "fail": "no independent check says success"}[final]
    if final == "success":
        return "claim confirmed by an independent check"
    if not claimed:
        return "program claimed fail but an independent check says success" if yes else "program claimed fail"
    if yes and no:
        return "independent checks disagree"
    if no:
        return "claim contradicted by an independent check"
    return "no independent evidence for the claim"
