"""Run the draft workflow over a rows file (workflow 2). Default ``--placement simple``: ① LLM plan, ② FastFill size, ③ one
LLM call places the room, ④ one program check (hard violations repaired once), ⑤ one LLM review (``workflow.simple``).
``--placement staged``: ① LLM plan, ② FastFill size, then the
floor furniture and then the small items, each stage placed group by group with program checks and reviewed once by
the LLM (``workflow.staged``; ``--no-semantic`` skips the reviews); ``--placement group``: one functional group at a
time placed by the LLM, checked and repaired, and a final whole-room critic loop with up to ``--gpt-repairs`` GPT
repairs and the critic-judged polish; ``--placement all``: ③ one LLM draft of the whole room and the critic loop;
``--no-critic``: ③ draft, ④ local repair, polish off, as v4.

LLM calls go to the official OpenAI API (``--api official``, settings in ``.env`` beside ``workflow/``) or the relay
(``--api relay``, ``~/.fastfill_api.env``); see ``workflow/credentials.py``.
Step ② either runs a FastFill checkpoint in this process (``--checkpoint``, GPU recommended) or reads sizes from a
predictions file aligned with the rows (``--sizes-file``, e.g. ``scripts/fastfill_server_run.py``'s output); ``--draft-sizes`` (the
workflow without FastFill) leaves step ② out: step ③'s placement call gives every size with the poses, the later
steps are unchanged (``simple.run_simple(draft_sizes=True)``). ``--task-roles`` (rows with a ``task`` field, e.g.
``examples/task_rooms.jsonl``): the task mode, with each row's ``task.required_ids`` / ``articulation_ids`` (``simple.run_simple``:
required objects never removed or shrunk, legal and reachable; expansion objects give way).

Before anything is loaded a free preflight (GET /v1/models/<model>) checks network, key and model, so a node without
internet access fails in seconds, not after the checkpoint load. Step ② runs first for every room (the checkpoint on
the GPU in this process); then ``--workers`` rooms run in parallel processes (LLM calls overlap, the CPU-bound
repairs use separate cores); resumable (rooms with a saved ``room_XXX.json``
are skipped). Budget: every room's cost (official prices) goes to ``ledger.json``, and the run stops
before a room when spent + ``--reserve-usd`` per room in flight would pass ``--budget-usd``; a rejected key or
exhausted quota starts no new room (those in flight finish). At the end: ``rows.jsonl`` / ``predictions.jsonl`` (the rooms run, aligned, ``layout`` null for a failure)
for ``fastfill.v2.evaluate``, and ``summary.json``; ``--evaluate`` also runs the evaluator into ``eval/``.

    python scripts/run_draft.py --rows examples/task_rooms.jsonl --indices 0 --task-roles \\
        --checkpoint ../runs/main4-cell05-v4detail-e5-1.7b-all-full/model-step-4500 --device cuda \\
        --api relay --out my_out --budget-usd 5
"""
import argparse, json, multiprocessing as mp, os, random, shutil, subprocess, sys, time, traceback, urllib.error, urllib.request
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # the workflow folder (holds workflow/, size_catalog.json, .env)
sys.path.insert(0, str(HERE))
from workflow.contract import _main as FASTFILL_ROOT  # noqa: E402  (also puts the fastfill package on sys.path)
from workflow.credentials import FatalAPIError, api_settings  # noqa: E402
from workflow.pipeline import run_draft_workflow  # noqa: E402
from workflow.robot import Robot  # noqa: E402


def preflight(settings: dict, attempts: int = 6) -> None:
    # the model list (a relay does not serve /models/<id>); the key, the host and the model are checked. A dropped
    # connection is retried with backoff (the relay resets a connection now and then: one reset must not stop a run);
    # an HTTP answer (key, model access) is final.
    request = urllib.request.Request(f"{settings['base_url']}/models",
                                     headers={"Authorization": f"Bearer {settings['api_key']}",
                                              "User-Agent": "fastfill-workflow/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                listed = [m.get("id") for m in json.load(response).get("data", [])]
            if listed and settings["model"] not in listed:
                raise SystemExit(f"preflight: model {settings['model']} is not served at {settings['base_url']}")
            return
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"preflight: the API answered HTTP {exc.code} for model {settings['model']} "
                             f"(key or model access): {exc.read()[:200]!r}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts - 1:
                raise SystemExit(f"preflight: cannot reach {settings['base_url']} from this node after {attempts} "
                                 f"tries ({exc}); run where there is internet access")
            print(f"preflight: {exc}; retrying in {5 * 2 ** attempt} s", flush=True)
            time.sleep(5 * 2 ** attempt)


def run_room(condition: dict, sizes: dict | None, yaws: dict | None, size_error: str | None, catalog: str,
             settings: dict, yaw_mode: str, critic: bool = True, gpt_repairs: int = 2,
             placement: str = "simple", semantic: bool = True, robot=None, drafts: int = 1,
             resize: bool = False, robot_margin: float = 0.0, hard_zones: bool = False,
             tag: str | None = None, relocate: bool = False, repair_rounds: int = 1,
             repair_seconds: float | None = None, repair: bool = True,
             draft_sizes: bool = False, roles: dict | None = None) -> tuple[dict, float, dict, int]:
    """One room in a worker process: steps ①, ③ and the critic loop with polish (``critic=False``: ④ local repair
    without polish, the v4 run); step ②'s sizes and yaws are given (``draft_sizes``: none, step ③ gives them)."""
    start = time.perf_counter()
    if tag:  # llm.ask logs this room's calls to <out>/llm_calls/<tag>.jsonl (WORKFLOW_LLM_LOG_DIR set by main)
        os.environ["WORKFLOW_LLM_TAG"] = tag
    try:
        result = run_draft_workflow(condition, sizes, catalog_path=catalog, polish=critic, model=settings["model"],
                                    base_url=settings["base_url"], api_key=settings["api_key"], yaws=yaws,
                                    yaw_mode=yaw_mode, critic=critic, gpt_repairs=gpt_repairs, placement=placement,
                                    semantic=semantic, robot=robot, drafts=drafts, resize=resize,
                                    robot_margin=robot_margin, hard_zones=hard_zones, relocate=relocate,
                                    repair_rounds=repair_rounds, repair_seconds=repair_seconds, repair=repair,
                                    draft_sizes=draft_sizes, **(roles or {}))
    except FatalAPIError:
        raise
    except Exception as exc:
        result = {"status": "error", "layout": None, "error": f"{type(exc).__name__}: {exc}"[:500],
                  "traceback": traceback.format_exc()[-2000:]}
    d = result.get("diagnostics") or {}
    if size_error and d:
        d["size_error"] = size_error
    loop = (d.get("critic") or {}).get("loop") or {}
    calls = d.get("plan_llm", {}).get("calls", []) + d.get("draft", {}).get("calls", []) + \
        [c for c in loop.get("gpt_calls", []) if "usage" in c] + \
        [c for stage in ("stage_one", "stage_two") for c in
         (((d.get("stages") or {}).get(stage) or {}).get("semantic") or {}).get("llm_calls", []) if "usage" in c] + \
        [c for c in (d.get("semantic") or {}).get("llm_calls", []) if "usage" in c]  # workflow 2's one review
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    usage = [c.get("usage") or {} for c in calls]
    tokens = {"prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
              "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
              "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                                      for u in usage)}
    result.update(wall_s=round(time.perf_counter() - start, 2), cost_usd=round(cost, 4), tokens=tokens)
    return result, cost, tokens, len(calls)


def roles(row: dict) -> dict:
    """The task mode's arguments from a row's ``task`` field (make_task_rooms.py)."""
    task = row.get("task") or {}
    if "required_ids" not in task:
        raise SystemExit("--task-roles needs rows with task.required_ids (e.g. examples/task_rooms.jsonl)")
    return {"required": set(task["required_ids"]), "articulation": set(task.get("articulation_ids") or ())}


def pick(rows: list[dict], a) -> list[int]:
    if a.indices:
        return a.indices
    if a.per_bin:
        rng, picked = random.Random(0), []
        for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 6)]:
            pool = [i for i, r in enumerate(rows) if lo <= len(r["condition"]["objects"]) <= hi]
            picked += sorted(rng.sample(pool, min(a.per_bin, len(pool))))
        return picked
    return list(range(len(rows)))[:a.limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="FastFill model dir (RUN/model-step-N, tokenizer beside it)")
    source.add_argument("--sizes-file", type=Path, help="predictions jsonl aligned with --rows (sizes are read)")
    source.add_argument("--draft-sizes", action="store_true", help="the workflow without FastFill: no step ②; "
                        "the placement call gives every size with the poses, the later steps are unchanged")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--reserve-usd", type=float, default=0.6, help="kept free for the next room (a large one ~0.4)")
    ap.add_argument("--indices", type=int, nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-bin", type=int, help="this many rooms per object-count bin (seed 0) instead of all")
    ap.add_argument("--catalog", type=Path, default=HERE / "size_catalog.json")
    ap.add_argument("--yaw-mode", choices=("prompt", "llm", "fastfill", "hybrid"), default="llm",
                    help="llm (default): FastFill gives sizes only and the LLM decides every facing; prompt: FastFill's "
                         "yaw goes to the LLM as a suggestion (pipeline.run_draft_workflow)")
    ap.add_argument("--workers", type=int, default=1, help="rooms in parallel processes (8 suits ~500 RPM)")
    ap.add_argument("--mp-start", default="spawn", help=argparse.SUPPRESS)  # tests use fork to keep their mocks
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--no-critic", action="store_true", help="the v4 workflow: ④ local repair, soft doors, no polish")
    ap.add_argument("--gpt-repairs", type=int, default=2, help="critic loop: GPT repair requests per room (K_repair)")
    ap.add_argument("--placement", choices=("simple", "staged", "group", "all"), default="simple",
                    help="simple (workflow 2): one LLM placement, one program check, one LLM review; "
                         "staged: floor furniture then small items, program checks per group and an LLM review per "
                         "stage; group: one functional group per LLM call, then a whole-room repair; all: one LLM call")
    ap.add_argument("--no-semantic", action="store_true", help="simple / staged: skip the LLM semantic review(s)")
    ap.add_argument("--robot-width", type=float, help="simple: the robot checks (workflow/robot.py) for a robot this "
                    "wide (m) are hard violations")
    ap.add_argument("--robot-arm", type=float, default=.4, help="with --robot-width: how far beyond its half width "
                    "the robot's centre may stay from a side it works at (m)")
    ap.add_argument("--drafts", type=int, default=1, help="simple: placement calls in parallel; the one with the fewest "
                    "hard violations is repaired (each call is paid)")
    ap.add_argument("--resize", action="store_true", help="simple: when repair leaves legality violations, retry "
                    "with objects up to 20 %% smaller (reported in diagnostics.resized)")
    ap.add_argument("--robot-margin", type=float, default=0.0, help="with --robot-width: placement, critic and search "
                    "plan for a robot this much wider (m); the result is reported at --robot-width")
    ap.add_argument("--hard-zones", action="store_true", help="with --robot-width: fix each carrying furniture's "
                    "approach side from the draft and let the search filter on the robot zones")
    ap.add_argument("--relocate", action="store_true", help="simple: step ④ may move a robot check's blocker (or "
                    "the unreachable furniture) anywhere in the room")
    ap.add_argument("--repair-rounds", type=int, default=1, help="simple: passes over the issues in step ④ (a new "
                    "pass only while the last one changed something)")
    ap.add_argument("--repair-seconds", type=float, help="simple: time limit of step ④'s program search (s)")
    ap.add_argument("--api", choices=("official", "relay"), default="official", help="official: the workflow's own "
                    "OpenAI .env; relay: ~/.fastfill_api.env's key at --base-url (adds ~3.5k hidden input tokens a call)")
    ap.add_argument("--base-url", help="with --api relay: e.g. https://api.kbsfdxc.net/codex/v1 (default), "
                    "https://api.kbsfdxc.com/codex/v1, https://api-slb.kbsfdxc.net/codex/v1")
    ap.add_argument("--model", help="model id (default: the settings file's OPENAI_MODEL), e.g. gpt-6.1-sol")
    ap.add_argument("--rpm", type=int, help="requests per minute across all workers (the relay allows 30)")
    ap.add_argument("--rpm-file", type=Path, help="with --rpm: the request log shared by every run that uses it "
                    "(runs started at the same time then stay under --rpm together)")
    ap.add_argument("--no-repair", action="store_true", help="simple: no program check-and-repair (step ④): the "
                    "LLM's draft as placed (with --no-semantic: the plan and draft calls only, a draft-only baseline)")
    ap.add_argument("--task-roles", action="store_true", help="the task mode: each row's task.required_ids and "
                    "task.articulation_ids (required objects kept, legal and reachable; expansion objects may be "
                    "removed or changed)")
    ap.add_argument("--no-llm-log", action="store_true", help="do not write <out>/llm_calls/room_<i>.jsonl (each LLM "
                    "call's instructions, prompt and answer; the API key is never written)")
    a = ap.parse_args()
    if (a.robot_margin or a.hard_zones) and not a.robot_width:
        raise SystemExit("--robot-margin / --hard-zones need --robot-width")
    if (a.drafts > 1 or a.resize) and (a.placement != "simple" or a.no_critic):
        raise SystemExit("--drafts / --resize work with --placement simple only")
    robot = Robot(width_m=a.robot_width, arm_m=a.robot_arm) if a.robot_width else None
    if robot and (a.placement != "simple" or a.no_critic):
        raise SystemExit("--robot-width works with --placement simple only")

    settings = api_settings(a.api, a.base_url, a.model)
    preflight(settings)
    print(f"model {settings['model']} at {settings['base_url']}: reachable", flush=True)
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    picked = pick(rows, a)
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.no_llm_log:  # every LLM call's instructions, prompt and answer (never the key), one file per room
        os.environ["WORKFLOW_LLM_LOG_DIR"] = str((a.out / "llm_calls").resolve())
    (a.out / "rooms.json").write_text(json.dumps(picked))
    if a.rpm:  # every request of every worker, retries included, stays under this per rolling minute (workflow/llm.py)
        os.environ["WORKFLOW_RPM"] = str(a.rpm)  # --rpm-file shared by concurrent runs keeps them under it together
        os.environ["WORKFLOW_RPM_FILE"] = str((a.rpm_file or a.out / ".rpm_state.json").resolve())
    sizer, file_sizes = None, None
    if a.checkpoint:
        from workflow.fastfill_sizes import FastFillSizer
        sizer = FastFillSizer(a.checkpoint, device=a.device)
        print(f"FastFill loaded: {sizer.describe()}", flush=True)
    elif a.sizes_file:
        lines = [json.loads(l) for l in a.sizes_file.read_text().splitlines() if l.strip()]
        file_sizes = [r["layout"] for r in lines]
        file_errors = [r.get("error") for r in lines]
        if len(file_sizes) != len(rows):
            raise SystemExit(f"--sizes-file has {len(file_sizes)} lines, --rows {len(rows)}")
    (a.out / "config.json").write_text(json.dumps({
        "rows": str(a.rows), "model": settings["model"], "base_url": settings["base_url"], "api": a.api, "rpm": a.rpm,
        "sizes": sizer.describe() if sizer else {"draft": settings["model"]} if a.draft_sizes else {"file": str(a.sizes_file)}, "yaw_mode": a.yaw_mode, "placement": a.placement if not a.no_critic else "v4",
        "semantic": a.placement in ("simple", "staged") and not a.no_semantic, "critic": not a.no_critic,
        "robot": {"width_m": a.robot_width, "arm_m": a.robot_arm, "margin_m": a.robot_margin,
                  "hard_zones": a.hard_zones} if a.robot_width else None,
        "repair": {"off": a.no_repair, "relocate": a.relocate, "rounds": a.repair_rounds, "seconds": a.repair_seconds},
        "drafts": a.drafts, "resize": a.resize, "task_roles": a.task_roles,
        "budget_usd": a.budget_usd,
        "rooms": len(picked)}, indent=1))

    ledger_path = a.out / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"rooms": {}, "spent_usd": 0.0}
    pending = [i for i in picked if not (a.out / f"room_{i:03d}.json").exists()]
    # step ② for every pending room first, on the GPU in this process (well under a second a room); the rooms then
    # run in worker processes, so the CPU-bound repair of several rooms runs on several cores
    sizes, yaws, size_errors = {}, {}, {}
    for n, i in enumerate(pending):
        if a.draft_sizes:  # given by the placement call
            sizes[i], yaws[i] = None, None
        elif sizer is not None:
            try:
                sizes[i], yaws[i] = sizer.predict(rows[i]["condition"])
            except ValueError as exc:  # over the model's object or context budget: the catalog's sizes
                sizes[i], yaws[i], size_errors[i] = None, None, f"{type(exc).__name__}: {exc}"[:300]
        else:
            layout = file_sizes[i]
            sizes[i] = {b["id"]: b["target_size_local_m"] for b in layout["objects"]} if layout else None
            yaws[i] = {b["id"]: b["yaw_rad"] for b in layout["objects"]} if layout else None
            if not layout and file_errors[i]:  # the file's own record of why (as the in-process sizer reports it)
                size_errors[i] = str(file_errors[i])[:300]
        if sizer is not None and (n + 1) % 50 == 0:
            print(f"sizes: {n + 1}/{len(pending)}", flush=True)
    if sizer is not None:
        del sizer  # free the GPU before the workers start
    running: dict = {}
    stop = False
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context(a.mp_start)) as pool:
        while pending or running:
            # the budget holds a reserve for every room in flight, so parallel rooms cannot overshoot it
            while pending and not stop and len(running) < a.workers:
                if ledger["spent_usd"] + (len(running) + 1) * a.reserve_usd > a.budget_usd:
                    print(f"budget stop: spent ${ledger['spent_usd']:.2f} of ${a.budget_usd:.2f} "
                          f"({len(running)} rooms still finishing)", flush=True)
                    stop = True
                    break
                i = pending.pop(0)
                running[pool.submit(run_room, rows[i]["condition"], sizes[i], yaws[i], size_errors.get(i),
                                    str(a.catalog), settings, a.yaw_mode, not a.no_critic, a.gpt_repairs,
                                    a.placement, not a.no_semantic, robot, a.drafts, a.resize, a.robot_margin,
                                    a.hard_zones, tag=f"room_{i:03d}", relocate=a.relocate,
                                    repair_rounds=a.repair_rounds, repair_seconds=a.repair_seconds,
                                    repair=not a.no_repair, draft_sizes=a.draft_sizes,
                                    roles=roles(rows[i]) if a.task_roles else None)] = i
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                i = running.pop(future)
                try:
                    result, cost, tokens, ncalls = future.result()
                except FatalAPIError as exc:
                    print(f"stopped: {exc} (no new rooms; {len(running)} still finishing)", flush=True)
                    stop = True
                    continue
                result["room"] = i
                d = result.get("diagnostics") or {}
                (a.out / f"room_{i:03d}.json").write_text(json.dumps(result, ensure_ascii=False))
                ledger["rooms"][str(i)] = {"cost_usd": round(cost, 4), **tokens, "calls": ncalls}
                ledger["spent_usd"] = round(ledger["spent_usd"] + cost, 4)
                ledger_path.write_text(json.dumps(ledger, indent=1))
                print(f"room {i:3d} objects {len(rows[i]['condition']['objects']):3d} {result['status']:12s} "
                      f"{result['wall_s']:6.1f}s ${cost:.3f} (total ${ledger['spent_usd']:.2f}) "
                      f"plan {d.get('plan_source')} sizes {d.get('size_source')} {d.get('size_error') or ''}",
                      flush=True)

    done = [i for i in picked if (a.out / f"room_{i:03d}.json").exists()]
    results = {i: json.loads((a.out / f"room_{i:03d}.json").read_text()) for i in done}
    (a.out / "rows.jsonl").write_text("".join(json.dumps(rows[i]) + "\n" for i in done))
    (a.out / "predictions.jsonl").write_text("".join(json.dumps({"layout": results[i].get("layout")}) + "\n"
                                                     for i in done))
    walls = sorted(r["wall_s"] for r in results.values())
    summary = {"rooms_picked": len(picked), "rooms_done": len(done),
               "status": dict(Counter(r["status"] for r in results.values())),
               "spent_usd": ledger["spent_usd"],
               "tokens": {k: sum(r.get("tokens", {}).get(k, 0) for r in results.values())
                          for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
               "wall_s": {"median": walls[len(walls) // 2] if walls else None, "max": walls[-1] if walls else None,
                          "sum": round(sum(walls), 1)},
               "size_fallback_rooms": sum(1 for r in results.values() if (r.get("diagnostics") or {}).get("size_error")),
               # an unexpected failure after placement started: the committed layout was kept, not dropped
               "error_recovered_rooms": sum(1 for r in results.values() if (r.get("diagnostics") or {}).get("error"))}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary:", json.dumps(summary), flush=True)
    if a.evaluate and done:
        shutil.rmtree(a.out / "eval", ignore_errors=True)  # evaluate refuses an existing output
        log = open(a.out / "eval.log", "w")
        code = subprocess.call([sys.executable, "-m", "fastfill.v2.evaluate", "--data", str((a.out / "rows.jsonl").resolve()),
                                "--predictions", str((a.out / "predictions.jsonl").resolve()),
                                "--output", str((a.out / "eval").resolve())], cwd=FASTFILL_ROOT, stdout=log, stderr=log)
        print(f"evaluate exit {code}; report: {a.out / 'eval' / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
