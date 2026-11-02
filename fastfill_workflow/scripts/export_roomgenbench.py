"""A saved run's layouts as RoomGenBench inputs, ready for asset generation and ``bench/assemble.py``.

Writes ``<out>/inputs/scenes/room_<i>.json`` (one per room with a layout, ``workflow.export.roomgenbench_scene``)
and ``<out>/inputs/assets.jsonl`` (``roomgenbench_assets``), in the layout of RoomGenBench's ``bench/inputs/``.
The room's fixed objects are exported too (``--no-fixed`` leaves them out). ``--preview`` also runs RoomGenBench's
own assembler in its ``layout_boxes`` mode (boxes only, no assets) to ``<out>/results/layout_boxes/scenes/``.

    python scripts/export_roomgenbench.py --run my_out --preview --roomgenbench <RoomGenBench checkout>

To build the rooms in RoomGenBench, copy ``<out>/inputs/scenes/*.json`` into its ``bench/inputs/scenes/`` and use
``<out>/inputs/assets.jsonl`` as its ``bench/inputs/assets.jsonl`` (or append it), generate the assets with a
method, then ``python bench/assemble.py --method <m> --scenes room_015,room_022,...``.
"""
import argparse, glob, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from workflow.contract import check_request, fill_room_defaults  # noqa: E402  (puts fastfill on sys.path)
from workflow.export import roomgenbench_assets, roomgenbench_scene  # noqa: E402
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="a run_draft.py output folder (room_<i>.json files)")
    ap.add_argument("--rows", type=Path, help="the run's rows file (default: the one in its config.json)")
    ap.add_argument("--out", type=Path, help="default: <run>/roomgenbench")
    ap.add_argument("--no-fixed", action="store_true", help="leave the rooms' fixed objects out")
    ap.add_argument("--preview", action="store_true", help="assemble box-only room GLBs with RoomGenBench's assembler")
    ap.add_argument("--roomgenbench", type=Path, help="RoomGenBench checkout (needed for --preview)")
    a = ap.parse_args()
    if a.preview and a.roomgenbench is None:
        ap.error("--preview needs --roomgenbench <RoomGenBench checkout>")
    out = a.out or a.run / "roomgenbench"
    rows_path = a.rows or (HERE / json.loads((a.run / "config.json").read_text())["rows"])
    rows = [migrate_legacy_row(json.loads(line)) for line in open(rows_path) if line.strip()]
    scene_dir = out / "inputs" / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scenes, skipped = [], []
    for f in sorted(glob.glob(str(a.run / "room_*.json"))):
        result = json.loads(Path(f).read_text())
        if not result.get("layout"):
            skipped.append(result["room"])
            continue
        i = result["room"]
        condition = check_request(fill_room_defaults(rows[i]["condition"])[0])
        scene = roomgenbench_scene(condition, result["layout"], scene_key=f"room_{i:03d}",
                                   include_fixed=not a.no_fixed)
        (scene_dir / f"{scene['scene_key']}.json").write_text(json.dumps(scene, ensure_ascii=False, indent=1))
        scenes.append(scene)
    assets = roomgenbench_assets(scenes)
    (out / "inputs" / "assets.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in assets))
    print(f"{len(scenes)} scenes, {sum(len(s['objects']) for s in scenes)} instances, {len(assets)} assets -> {out}"
          + (f"; no layout: {skipped}" if skipped else ""))
    if a.preview:
        spec = importlib.util.spec_from_file_location("rgb_assemble", a.roomgenbench / "bench" / "assemble.py")
        assemble = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assemble)
        assemble.INPUTS, assemble.ROOT = out / "inputs", out  # read our scenes, write under <out>/results
        for scene in scenes:
            assemble.assemble("layout_boxes", scene["scene_key"], out / "results" / "layout_boxes" / "scenes")
        print(f"preview: {out / 'results' / 'layout_boxes' / 'scenes'}")
    print("scenes:", ",".join(s["scene_key"] for s in scenes))


if __name__ == "__main__":
    main()
