"""Convert canonical layout to RoomGenBench's SAGE-style scene envelope.

``roomgenbench_scene`` gives one ``bench/inputs/scenes/<scene>.json`` and ``roomgenbench_assets`` the matching
``bench/inputs/assets.jsonl`` rows (one per type + description, as ``bench/prepare_inputs.py`` deduplicates), so
RoomGenBench's methods can generate the assets and ``bench/assemble.py`` can build the room mesh."""

from __future__ import annotations

import hashlib
import math

RAISED_M = .15  # a fixed object whose bottom is higher than this above the floor is treated as wall-mounted


def _asset_key(category: str, description: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in category.lower()).strip("_")[:24]
    return f"{key}_{hashlib.sha1(description.encode('utf-8')).hexdigest()[:8]}"


def _sage_object(ident, category, description, size, position, yaw, place_id) -> dict:
    sx, sy, sz = size
    # FastFill local +X is front; SAGE local +Y is front. Swap local XY and rotate by -90 degrees.
    sage_yaw = math.degrees((yaw - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
    return {"id": ident, "type": category, "description": description,
            "asset_key": _asset_key(category, description),
            "position": {"x": position[0], "y": position[1], "z": position[2]},
            "rotation": {"x": 0, "y": 0, "z": sage_yaw},
            "dimensions": {"width": sy, "length": sx, "height": sz},
            "place_id": place_id}


def roomgenbench_scene(condition: dict, layout: dict, *, scene_key: str = "fastfill_workflow",
                       include_fixed: bool = False) -> dict:
    """``include_fixed`` also exports the room's fixed objects (beams, columns, built-ins) as instances, placed on
    the floor or, when raised, on the wall."""
    room = condition["room"]
    polygon = room["floor_polygon_xy_m"]
    floor = room["floor_z_m"]
    xmin, xmax = min(p[0] for p in polygon), max(p[0] for p in polygon)
    ymin, ymax = min(p[1] for p in polygon), max(p[1] for p in polygon)
    walls = [{"id": f"wall_{i}",
              "start_point": {"x": a[0], "y": a[1], "z": floor},
              "end_point": {"x": b[0], "y": b[1], "z": floor},
              "height": room["height_m"], "thickness": .1}
             for i, (a, b) in enumerate(zip(polygon, polygon[1:] + polygon[:1]))]
    doors, windows = [], []
    for index, opening in enumerate(room.get("openings", [])):
        shared = {"id": f"opening_{index}", "wall_id": f"wall_{opening['wall_index']}",
                  "position_on_wall": opening["position_on_wall"],
                  "width": opening["width_m"], "height": opening["height_m"]}
        if opening["type"] == "door":
            doors.append({**shared, "door_type": opening.get("door_type", "door")})
        elif opening["type"] == "window":
            windows.append({**shared, "sill_height": opening.get("sill_height_m", 0)})
    by_id = {o["id"]: o for o in condition["objects"]}
    objects = []
    for box in layout["objects"]:
        req = by_id[box["id"]]
        objects.append(_sage_object(box["id"], req["category"], req["description"], box["target_size_local_m"],
                                    box["bottom_center_m"], box["yaw_rad"], box["support_parent"]))
    if include_fixed:
        for fixed in room.get("fixed_objects", []):
            category = fixed.get("category", "fixed object")
            raised = fixed["bottom_center_m"][2] - floor > RAISED_M
            objects.append(_sage_object(fixed["id"], category, fixed.get("description") or category,
                                        fixed["size_local_m"], fixed["bottom_center_m"], fixed["yaw_rad"],
                                        "wall" if raised else "floor"))
    return {"scene_key": scene_key, "room_type": room.get("room_type", "room"),
            "building_style": room.get("style", ""),
            "prompt": room.get("description", ""),
            "room": {"dimensions": {"width": xmax - xmin, "length": ymax - ymin,
                                    "height": room["height_m"]},
                     "position": {"x": xmin, "y": ymin, "z": floor},
                     "ceiling_height": room["height_m"], "walls": walls,
                     "doors": doors, "windows": windows},
            "objects": objects}


def roomgenbench_assets(scenes: list[dict]) -> list[dict]:
    """``assets.jsonl`` rows for ``scenes``: one per asset key, with the first instance's box and placement
    (``floor`` / ``wall`` / ``on_object``), the scenes using it and its number of instances."""
    assets = {}
    for scene in scenes:
        for obj in scene["objects"]:
            entry = assets.setdefault(obj["asset_key"], {
                "asset_key": obj["asset_key"], "type": obj["type"], "description": obj["description"],
                "dimensions": dict(obj["dimensions"]),
                "place": obj["place_id"] if obj["place_id"] in ("floor", "wall") else "on_object",
                "scenes": [], "n_instances": 0})
            if scene["scene_key"] not in entry["scenes"]:
                entry["scenes"].append(scene["scene_key"])
            entry["n_instances"] += 1
    return sorted(assets.values(), key=lambda a: a["asset_key"])
