"""Paired legacy-operator regression: updated native OA vs pre-fix native OA.

Reuse the already validated full-output/canary harness. Its historical `ptxas`
transport slot carries the baseline OA cubin here; every saved result key is
renamed and provenance identifies both binaries before the result is written.
There is no ptxas performance comparator in this regression run.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from exp.whole_stack import native_ops

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/"exp/weaknesses"


def rename(value):
    if isinstance(value,list):return [rename(item) for item in value]
    if isinstance(value,dict):
        def key(k):
            return (k.replace("oa_ptxas", "candidate_baseline")
                     .replace("oa/", "candidate_oa/").replace("ptxas/", "baseline_oa/")
                     if k not in ("oa","ptxas") else {"oa":"candidate_oa","ptxas":"baseline_oa"}[k])
        return {key(k):rename(v) for k,v in value.items()}
    if isinstance(value,str):
        return {"oa":"candidate_oa","ptxas":"baseline_oa"}.get(value,value)
    return value


def validated_baseline(payload):
    baseline=base64.b64decode(payload["legacy"]["legacy_oa"],validate=True)
    identity=json.loads((HERE/"reductions_production_identity.json").read_text())
    expected=identity["legacy_baseline_provenance"]["legacy_oa_sha256"]
    if hashlib.sha256(baseline).hexdigest()!=expected:
        raise RuntimeError("pre-fix scalar control changed; restore the recorded baseline commit/payload "
                           "instead of labeling a current build as pre-fix")
    return baseline


def main():
    payload=json.loads((HERE/"reductions_build_production/payload.json").read_text())
    baseline=validated_baseline(payload)
    out=ROOT/"exp/whole_stack/native_ops_build"
    entry=json.loads((out/"manifest.json").read_text())["entries"][0]
    candidate=(out/entry["cubin"]).read_bytes()
    with native_ops.APP.run():
        raw=native_ops.execute.remote({"oa":base64.b64encode(candidate).decode(),
                                       "ptxas":base64.b64encode(baseline).decode()})
    result=rename(raw)
    result["comparison"]="updated native OA versus pre-fix native OA; no ptxas comparator"
    result["provenance"]={"candidate_oa_sha256":hashlib.sha256(candidate).hexdigest(),
                          "baseline_oa_sha256":hashlib.sha256(baseline).hexdigest(),
                          "source_sha256":hashlib.sha256((ROOT/"examples/nn_ops.py").read_bytes()).hexdigest(),
                          "candidate_manifest":json.loads((out/"manifest.json").read_text()),
                          "transport_harness_sha256":hashlib.sha256(Path(native_ops.__file__).read_bytes()).hexdigest()}
    result["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
    result["harness_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=HERE/"reductions_runs"/f"scalar-regression-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print("evidence:",path)


if __name__=="__main__":main()
