"""FastFill sizes for the rooms of a rows file, on a GPU node, with per-room timing and a full record of the run.

One forward pass per room (``fastfill.v2.evaluate.predict_layout``, spread decode: the decode the workflow's sizes
come from), timed with CUDA synchronised before and after; the first rooms after loading are warm-up and marked.
Writes into --out:
  predictions_spread.jsonl  aligned with --rows (a layout for the chosen rooms, null elsewhere): the workflow reads
                            it with ``run_draft.py --sizes-file`` and it is the "FastFill alone" baseline
  rooms/room_XXX.json       per room: layout, seconds, object count, error
  timing.jsonl              per room: seconds, objects, warm-up flag
  run.json                  checkpoint, model config, GPU, torch / CUDA versions, host, Slurm job, code commit,
                            rows sha256, load time, totals
  (stdout)                  one line per room; the sbatch script keeps it as run.log

    python scripts/fastfill_server_run.py --rows examples/task_rooms.jsonl --rooms my_rooms.json \
        --checkpoint ../runs/main4-cell05-v4detail-e5-1.7b-all-full/model-step-4500 --out my_sizes --device cuda
    (my_rooms.json: the row numbers to run, from 0, e.g. [0, 1])
"""
import argparse, hashlib, json, os, platform, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from workflow import contract  # noqa: E402,F401  (puts the fastfill package on sys.path)
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--rooms", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2, help="rooms run first and marked as warm-up (still recorded)")
    a = ap.parse_args()
    import torch
    from fastfill.v2.batch import load_tokenizer
    from fastfill.v2.evaluate import predict_layout
    from fastfill.v2.io import load_checkpoint_config
    from fastfill.v2.model import load_model

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rooms").mkdir(exist_ok=True)
    rows = [migrate_legacy_row(json.loads(l)) for l in a.rows.read_text().splitlines() if l.strip()]
    rooms = json.loads(a.rooms.read_text())
    cuda = a.device.startswith("cuda") and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    t0 = time.perf_counter()
    model = load_model(a.checkpoint, device=a.device).eval()
    backbone = model.config.backbone
    tokenizer = load_tokenizer("tiny" if backbone == "tiny" else str(a.checkpoint.parent / "tokenizer"),
                               local_files_only=True)
    max_length = load_checkpoint_config(a.checkpoint)["max_length"] or 4096
    sync()
    load_s = time.perf_counter() - t0
    run = {"checkpoint": str(a.checkpoint.resolve()), "backbone": backbone, "max_length": max_length,
           "max_objects": model.config.max_objects, "decode": "spread", "device": a.device,
           "gpu": torch.cuda.get_device_name(0) if cuda else None, "torch": torch.__version__,
           "cuda": torch.version.cuda, "python": platform.python_version(), "host": socket.gethostname(),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "code_commit": git_commit(),
           "rows": str(a.rows), "rows_sha256": sha256(a.rows), "rooms": len(rooms), "load_s": round(load_s, 2),
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(run), flush=True)

    results = {}
    with open(a.out / "timing.jsonl", "w") as timing:
        for n, k in enumerate(rooms):
            condition = rows[k]["condition"]
            layout, error = None, None
            sync()
            start = time.perf_counter()
            try:
                with torch.no_grad():
                    layout = predict_layout(model, tokenizer, condition, max_length=max_length, device=a.device,
                                            grid_decode="spread")
            except Exception as exc:  # over the model's object or context budget: recorded, the workflow falls back
                error = f"{type(exc).__name__}: {exc}"[:300]
            sync()
            seconds = time.perf_counter() - start
            rec = {"room": k, "objects": len(condition["objects"]), "seconds": round(seconds, 4),
                   "warmup": n < a.warmup, "error": error}
            results[k] = {**rec, "layout": layout}
            (a.out / "rooms" / f"room_{k:03d}.json").write_text(json.dumps(results[k], ensure_ascii=False))
            timing.write(json.dumps(rec) + "\n")
            print(f"{n + 1}/{len(rooms)} room {k:3d} objects {rec['objects']:3d} {seconds * 1000:8.1f} ms"
                  + (" warm-up" if rec["warmup"] else "") + (f" ERROR {error}" if error else ""), flush=True)

    with open(a.out / "predictions_spread.jsonl", "w") as f:  # aligned with --rows, as the earlier runs' files
        for k in range(len(rows)):
            r = results.get(k)
            f.write(json.dumps({"layout": r["layout"] if r else None, "error": (r["error"] if r else "not run"),
                                "latency_s": r["seconds"] if r else None}) + "\n")
    timed = [r["seconds"] for r in results.values() if not r["warmup"] and not r["error"]]
    timed.sort()
    run.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), rooms_ok=sum(not r["error"] for r in results.values()),
               rooms_failed=sum(bool(r["error"]) for r in results.values()),
               seconds_total=round(sum(r["seconds"] for r in results.values()), 2),
               seconds_median=timed[len(timed) // 2] if timed else None, seconds_max=timed[-1] if timed else None)
    (a.out / "run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps({k: run[k] for k in ("rooms_ok", "rooms_failed", "seconds_total", "seconds_median", "seconds_max",
                                           "load_s")}), flush=True)


if __name__ == "__main__":
    main()
