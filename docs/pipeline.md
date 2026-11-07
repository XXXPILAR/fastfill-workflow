# FastFill workflow: how it works

How to install and run it (`python -m workflow run`), the input and output formats and the coordinate conventions
are in the [README](../README.md) (in Chinese); every rule of the program repair, with its threshold and where it is
in the code, is in [program_repair_rules.md](program_repair_rules.md) (in Chinese). This page describes the steps.
Paths are relative to `fastfill_workflow/`.

## Steps (`workflow/simple.py`, entry `pipeline.run_draft_workflow`)

1. **Plan**: one LLM call for supports, groups and relations (`planning.make_plan`).
2. **Sizes**: one per object, from FastFill when given (or `fastfill_sizes.FastFillSizer`), otherwise from the size
   catalog. No step ever resizes an object.
3. **Place**: one LLM call places every object in the room (`drafting.make_draft`) and decides every facing. FastFill
   gives the sizes only: its yaw is trained modulo π, so it gives a box axis, not which way the front faces (73% of
   its wall-side furniture faces into the room, against 92% for GPT-5.6). `yaw_mode="prompt"` still passes it to the
   LLM as a suggestion.
4. **Check**: the program checks once (`critic.Critic`). If there are hard violations, the repair loop
   (`loop.Loop`) takes each of them up:
   - hard violations: collision, out of bounds, missing support, blocked door, a wall object over a window, a broken
     hard input constraint, and with a robot the robot checks;
   - the program search keeps sizes fixed;
   - soft costs are not chased and there is no polish, so any part of the layout that is already legal stays as the
     LLM placed it.
5. **Review**: the LLM reviews the whole room once (`semantic.SemanticPass`, one round), and only severe issues are
   fixed. A fix is kept only if all of the following hold:
   - the program repairs it;
   - it passes the program gate: no new hard violation, and reach, access and doors are no worse;
   - a blind pairwise comparison prefers it in both orders.
6. **Result**: the layout, plus the layout after each step (`diagnostics.stages`), so a bad result can be traced back
   to the step that caused it.

## Task mode: required and expansion furniture

The upstream task agent sends, per room: the room shell and type, the full furniture list (every object with a
detailed description and its support), which objects the task **requires**, which ones the task **opens**
(articulation), and hard constraints. Every other object is **expansion** furniture. The task mode places all of them,
with this order of priority: legality, then the robot's reach, then the task.

- **Required objects** are never removed and never shrunk. Their violations are repaired first, and they must end
  legal (no collision, inside the room, no blocked door, constraints met) and reachable (the robot reaches them, items
  on furniture are within arm's reach, articulation objects can open).
- **Expansion objects** are checked and repaired the same way. An expansion object is removed only to give way to a
  required object (after moving it anywhere in the room failed), or at the end when its own physical violation cannot
  be fixed (④d).

Steps added to `workflow/simple.py` (on top of steps 1-6 above):

| Step | What it does |
|---|---|
| 3 place | the placement request marks required objects (`"required": true`, `drafting.REQUIRED_RULE`) |
| 4 check | the robot checks cover every object plus the task's (`simple.task_robot`); the repair takes the required objects' violations first (`Loop(first=...)`) |
| 4b give way | an expansion object in a required object's violation is moved anywhere in the room, else removed with what stands on it; when several expansion objects block together, the nearest ones are tried (at most `NEAR_BLOCKER_TRIES`), a removal kept only if it helps; one more repair pass after removals |
| 4c required first | if a required object still has a violation: the required objects alone are repaired (or searched for from scratch), then the expansion objects come back one at a time, each kept only where the required objects stay as good (`carry` moves objects standing on required furniture with it) |
| 4d left out | after the review: an expansion object still in a physical violation (collision, out of bounds, no support, blocking a door or a window) goes to a free spot on its parent's top (`free_slot`, `repack`), else it is removed |
| 4e wall fronts | a wall object whose front floor strip, as SceneEval's accessibility scores it, is under 55% free (it hangs over furniture) slides along the walls to the nearest place at least 65% free, with what stands on it, if that adds no violation and breaks no relation (`clear_wall_fronts`); nothing on the floor moves and nothing is removed |
| 7 fallback | if a required object is still not legal and reachable, or the workflow fails on an error: only the required objects, laid out by the program without the LLM (`simple.required_only`), legality and reach before the constraints; returned when it leaves the required objects better off, with `status` `fallback_required_only` |

**Input.** The request is the usual `fastfill.v2` condition; every object needs a description beyond its category
(`contract.check_request`). The task's ids are passed beside it: `required=ids, articulation=ids`.

**Output.** The layout holds the objects kept. `diagnostics.task` gives `required`, `articulation`, `expansion`,
`expansion_kept`, `moved` (each with the step that moved it), `removed` (each with what stood on it and the
violations it was removed for), `required_first` (whether step 4c ran and helped), `wall_clear` (step 4e: the wall
objects, those under the 55%, what moved where and how far, what found no place), `expansion_issues`, `required_ok`
and `required_issues`. A fallback result also keeps the full layout and its diagnostics in
`diagnostics.fallback_from` and lists the constraints it could not meet (`unmet_constraints`).

**When the fallback runs.** Only when the steps above leave a required object with a violation (1 room of the 150 of
the test150doors run), or on an unexpected error. A busy or failing relay does not trigger it: LLM calls are
retried (`WORKFLOW_LLM_ATTEMPTS=0`: until they pass), and a step whose call fails falls back on its own (rules plan,
default draft positions, no review). A rejected key or an exhausted quota (`FatalAPIError`) still stops the run.

**Known limits.**
- Constraint types: `on`, `between`, `against_wall`, `faces`, `faces_direction`, `near`, `clearance`, `keepout`;
  any other type is refused at the input.
- Constraints that involve a removed expansion object are dropped with it; they are not reported yet.
- Contradictory constraints are the upstream agent's to avoid; the fallback still delivers the required objects
  legal and reachable and lists the constraints it could not meet.
- Large rooms take minutes (with a 180 s limit on the program search per pass); median 200 s per room on
  test150doors.

## Robot checks (`workflow/robot.py`)

The program checker also asks whether a mobile robot of a given width can use the room (2D footprints only, about
2 ms per room):

- **`robot_door`:** every door is linked to the others through floor the robot can drive on.
- **`robot_reach`:** the robot can reach the furniture it has to work at: furniture carrying another requested
  object, openable furniture, and the task's objects (`Robot.reach_ids`).
- **`robot_openable`:** the sweep in front of a cabinet, wardrobe, fridge and so on is free, so its doors and drawers
  can open.
- **`robot_arm`:** an object standing on furniture is within the arm's 0.85 m (in plan) of a spot the robot can drive
  to. This follows EmbodiedGen's `check_reachable`.

The navigation check computes the same free space as EmbodiedGen's `RoamTrajectoryGenerator` (floor minus furniture,
keeping a clearance), but exactly instead of on a 5 cm grid.

With `robot=...`, these checks are hard violations, the result reports them for that robot (`diagnostics.robot`),
and the robot is also part of placement and repair:

- **Placement request** (`drafting.ROBOT_RULE`): the LLM is told the robot's width. Openable furniture carries
  `front_clearance_m`, and furniture that holds objects carries `robot_works_here`.
- **Repair search** (`robot.RobotZones`): the search ranks its candidates with the checks' cheap local form, added to
  `Geometry.soft_score`: the zone in front of openable furniture; a standing strip on one approach side of furniture
  that holds objects; objects on that furniture within arm's reach of the strip. These zones play the role of the
  usage areas in Make it Home and Merrell et al. (2011), and of Infinigen's accessibility cost. Door linking is not
  modelled in the search; only the critic checks it.
- **Objects on furniture**: with a robot, the search also tries spots along each edge of the support's top
  (EmbodiedGen's surface sampling). The arm-reach zone then ranks these spots.
- **Commit rule** (`Critic.better`): decisions run in tiers. Legality comes first, then the robot checks, then the
  soft cost. A robot check therefore never blocks a legality repair. Robot violations are identified by their object,
  not by the list of blockers.

Two more options:

- **`drafts=N`:** makes N placement calls in parallel and keeps the draft with the fewest hard violations of any kind
  (`draft_rank`). Ranking by legality first was wrong: the repair fixes small collisions easily but rarely fixes robot
  violations. This follows the generate-then-verify idea of WeakVerifiers. Every call is paid.
- **`draft_sizes=True`** (no FastFill): the placement call gives every object's size with its pose, and the later
  steps keep the chosen draft's sizes.

Footprints, wall contacts and robot-zone coverage are cached by pose. This made the repair of the largest room about
5 times faster.
