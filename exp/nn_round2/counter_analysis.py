"""Summarize diagnostic counter observations without replay speed claims."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/"exp/nn_round2"
BASE=HERE/"counter_recovered_20261107T143821.759059Z.json"
FMA=HERE/"runs/counters-20261107T144534.159647Z.json"
FIELDS={
    "profile_duration_ns":"gpu__time_duration.sum",
    "observed_sm_clock_hz":"sm__cycles_elapsed.avg.per_second",
    "tensor_active_pct":"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "active_warps_pct":"sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_cycle":"smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct":"smsp__issue_active.avg.pct_of_peak_sustained_active",
    "dynamic_instructions":"smsp__inst_executed.sum",
    "dynamic_fma_pipe_instructions":"smsp__inst_executed_pipe_fma.sum",
    "dynamic_tensor_pipe_instructions":"smsp__inst_executed_pipe_tensor.sum",
    "dynamic_xu_pipe_instructions":"smsp__inst_executed_pipe_xu.sum",
    "short_scoreboard_pct":"smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "long_scoreboard_pct":"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "wait_pct":"smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "barrier_pct":"smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "mio_throttle_pct":"smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "math_pipe_throttle_pct":"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "shared_load_bank_conflicts":"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "shared_load_wavefronts":"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "dram_bytes":"dram__bytes.sum",
    "l2_bytes":"lts__t_bytes.sum",
    "l2_sector_hit_pct":"lts__t_sector_hit_rate.pct",
    "registers_per_thread":"launch__registers_per_thread",
}


def row_metrics(row):
    raw=row.get("requested_metrics",row.get("metrics",{}))
    return {label:raw[name][0]["value"] if raw.get(name) else None for label,name in FIELDS.items()}


def summarize():
    old=json.loads(BASE.read_text())
    fresh=json.loads(FMA.read_text())["result"]
    rows=[]
    for profile in old["profiles"]:
        if not profile["recovered_status"].startswith("collected"):
            raise ValueError("base counter proof must be recovered successfully")
        rows.append({"id":profile["id"],"scope":"base","metrics":row_metrics(profile),"proof_passed":profile["proof"]["passed"]})
    for profile in fresh["profiles"]:
        if profile["status"]!="collected" or not profile["proof"]["passed"]:
            raise ValueError("FMA counter proof failed")
        rows.append({"id":profile["id"],"scope":"fma_followup","metrics":row_metrics(profile),"proof_passed":True})
    by_id={row["id"]:row["metrics"] for row in rows}
    pairs=[]
    for stem in ("op_rmsnorm_backward_w513_c4","op_softmax_backward_w513_c8"):
        baseline=by_id[stem+"_baseline/oa"]
        candidate=by_id[stem+"_fma/oa"]
        pairs.append({"kernel_family":stem,"baseline":baseline,"explicit_fma":candidate,
                      "register_reduction":baseline["registers_per_thread"]-candidate["registers_per_thread"],
                      "dynamic_instruction_reduction_pct":100*(1-candidate["dynamic_instructions"]/baseline["dynamic_instructions"]),
                      "active_warps_percentage_point_change":candidate["active_warps_pct"]-baseline["active_warps_pct"],
                      "dram_bytes_change_pct":100*(candidate["dram_bytes"]/baseline["dram_bytes"]-1)})
    return {"schema":1,"generated_utc":datetime.now(timezone.utc).isoformat(),
            "sources":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (BASE,FMA,Path(__file__))},
            "profiles":rows,"fma_pairs":pairs,
            "scope":"16 successful single-launch diagnostic profiles, not independent performance replications. Raw clocks/caches were uncontrolled; observed clock differences preclude treating NCU durations as benchmark gains.",
            "findings":[
                "Tall CTA-order controls change L2/DRAM traffic and long-scoreboard stalls strongly while tensor instruction work stays equal. This supports a locality/memory-wait explanation for the measured order gap, not a claim that the baseline ptxas scheduler globally loses to OA.",
                "Wide OA legacy/direct executes more instructions and has more short-scoreboard stalls than its same-source ptxas controls; pointer-copy removal does not close the tensor scheduling gap.",
                "Explicit local FMA removes arithmetic and live temporaries, lowers reported registers, and increases measured active warps. DRAM bytes/L2 traffic and zero shared-bank conflicts are essentially unchanged in the contrasted scalar kernels.",
                "RMSNorm eligible-warp and issue metrics do not improve despite increased active warps; long-scoreboard stalls remain dominant. Do not equate occupancy with useful issue throughput.",
                "The SFU-only metric spelling is unavailable. The supported XU instruction counter is a combined named execution-pipe observation, not an isolated special-function performance cause."]}


if __name__=="__main__":
    result=summarize()
    output=HERE/"counter_summary_20261107.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(output)
