# FastFill workflow: how it works

How to install and run it, the input and output formats and the coordinate conventions are in the
[README](../README.md) (in Chinese). This page describes the steps. Paths are relative to `fastfill_workflow/`.

## Steps (`workflow/simple.py`, entry `pipeline.run_draft_workflow`)

1. **Plan**: one LLM call for supports, groups and relations (`planning.make_plan`, semantic only).
2. **Sizes**: one per object, from FastFill when given (`--sizes-file` or `--checkpoint`), otherwise from the size
   catalog (`size_catalog.json`). With `--draft-sizes` there is no FastFill: the placement call gives every size.
3. **Place**: LLM calls place every object in the room and decide every facing (`drafting.make_draft`); with
   `--drafts N`, N calls run in parallel and the draft with the fewest hard violations is kept (`draft_rank`).
   FastFill gives the sizes only: its yaw is trained modulo π, so it gives a box axis, not which way the front faces.
4. **Check and repair**: the program checks once (`critic.Critic`); each hard violation is taken up by a program
   search that keeps sizes fixed (`loop.Loop`, `local_repair.py`):
   - hard violations: collision, out of bounds, missing support, blocked door, a broken hard input constraint, and,
     with `--robot-width`, the robot checks below;
   - `--relocate` lets the search move an object anywhere in the room; `--resize` lets a second pass shrink an object
     by up to 20 % when legality violations remain (never a required object);
   - soft costs are not chased, so whatever is already legal stays as the LLM placed it.
5. **Review**: the LLM reviews the whole room once (`semantic.SemanticPass`) and only severe issues are fixed. A fix is
   kept only if the program repairs it, it adds no hard violation and leaves reach, access and doors no worse, and a
   blind pairwise comparison prefers it in both orders.
6. **Result**: the layout, plus the layout after each step (`diagnostics.stages`), so a bad result can be traced
   back to the step that caused it.

## Task mode: required and expansion furniture

The upstream task agent sends the room shell and type, the full furniture list (each object with a description and
its support), which objects the task **requires**, which ones it **opens** (articulation, a subset of the required
ones), and hard constraints. Every other object is **expansion** furniture. All are placed, with this priority:
legality, then the robot's reach, then the task.

- **Required objects** are never removed and never shrunk. Their violations are repaired first, and they must end
  legal and reachable (the robot reaches them, items on furniture are within arm's reach, articulation objects can
  open whatever their category).
- **Expansion objects** are checked and repaired the same way. One is removed only to give way to a required object,
  and only after moving it anywhere in the room failed. Its own violations never remove it; they are reported
  (`expansion_issues`).

Steps added by the task mode (`run_simple(required=..., articulation=...)`, `run_draft.py --task-roles`):

| Step | What it does |
|---|---|
| 3 place | the placement request marks required objects (`"required": true`, `drafting.REQUIRED_RULE`) |
| 4 check | the robot checks cover every object plus the task's (`task_robot`); the repair takes the required objects' violations first and never shrinks them |
| 4b give way | an expansion object in a required object's violation is moved anywhere in the room, else removed with what stands on it; when several block together, the nearest are tried (at most `NEAR_BLOCKER_TRIES`), a removal kept only if it helps |
| 4c required first | if a required object still has a violation: the required objects alone are repaired (or searched for from scratch), then the expansion objects come back one at a time, each kept only where the required objects stay as good (objects standing on required furniture move with it) |
| 7 fallback | if a required object is still not legal and reachable, or the workflow fails on an error: only the required objects, laid out by the program without the LLM (`required_only`), legality and reach before the constraints; returned when it leaves the required objects better off, with `status` `fallback_required_only` and the constraints it could not meet (`unmet_constraints`) |

**When the fallback runs.** Only when the steps above leave a required object with a violation, or on an unexpected
error. A busy relay does not trigger it: LLM calls are retried (`WORKFLOW_LLM_ATTEMPTS=0`: until they pass), and a
step whose call fails falls back on its own (rules plan, default draft positions, no review). A rejected key or an
exhausted quota (`FatalAPIError`) stops the run.

## Robot checks (`workflow/robot.py`)

With `--robot-width W` the checker asks whether a mobile robot of that width can use the room (2D footprints, about
2 ms per room):

- **`robot_door`**: every door is linked to the others through floor the robot can drive on (with no door, the
  largest free area stands for the entrance).
- **`robot_reach`**: the robot can reach the furniture it has to work at: furniture carrying another object,
  openable furniture, and the task's objects (`Robot.reach_ids`).
- **`robot_openable`**: the sweep in front of a cabinet, wardrobe, fridge and so on is free, so its doors and drawers
  can open (`Robot.openable_ids` adds the task's articulation objects).
- **`robot_arm`**: an object standing on furniture is within the arm's 0.85 m (in plan) of a spot the robot can drive
  to, as in EmbodiedGen's `check_reachable`.

The navigation check computes the same free space as EmbodiedGen's `RoamTrajectoryGenerator` (floor minus furniture,
keeping a clearance), exactly instead of on a 5 cm grid.

The robot is also part of placement and repair:

- **Placement request** (`drafting.ROBOT_RULE`): the LLM is told the robot's width; openable furniture carries
  `front_clearance_m`, and furniture that holds objects carries `robot_works_here`.
- **Repair search** (`robot.RobotZones`): candidates are ranked with the checks' cheap local form: the zone in front
  of openable furniture, a standing strip on one side of furniture that holds objects, and objects on it within arm's
  reach of the strip. `--robot-margin` widens these zones and `--hard-zones` makes them hard.
- **Commit rule** (`Critic.better`): legality first, then the robot checks, then the soft cost, so a robot check
  never blocks a legality repair.

## Export (`workflow/export.py`, `scripts/export_roomgenbench.py`)

`scripts/export_roomgenbench.py --run <out>` writes a run's layouts as RoomGenBench / SAGE scenes (front +Y, width
along local X, bottom-centre position, yaw in degrees) with the room's fixed objects, and one asset row per type and
description. `--preview --roomgenbench <checkout>` assembles box-only room meshes with RoomGenBench's own assembler.

## Other entry point

`python -m workflow run --input <condition.json> --output <layout.json>` runs the earlier workflow on one condition;
it has no task mode. Use `scripts/run_draft.py` for the pipeline above.
