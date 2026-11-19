# Camera names

Episodes name their cameras freely: the same wrist camera appears as `wrist`, `wrist_cam`, `eye_in_hand`, `hand` or `TCP_camera`, and `left` is an exterior camera in one setup and a wrist camera in another. `configs/camera_names.yaml` maps every recorded camera to a unified name, which the index stores in `unified_cameras` and training recipes use for their camera roles ([training data](training-data.md#cameras)).

| Stage | Target | Runs after | Writes |
|---|---|---|---|
| `unified_camera_names` | `preqa` | `duplicate_episodes`, `camera_names_config` | `unified_cameras`, `camera_name_issues` |

## Names

| Name | Camera |
|---|---|
| `wrist` | on the end effector of a single-arm robot |
| `wrist_left`, `wrist_right` | on the end effectors of a bimanual robot |
| `head` | on the robot body or the operator: head, chest, torso and egocentric cameras |
| `exterior_front` | static, facing the robot across the workspace |
| `exterior_side` | static, from the side of the workspace |
| `exterior_top` | static, looking down onto the workspace |
| `exterior_back` | static, behind the robot, looking over it |
| `tactile` | the image of a visuotactile sensor; not a camera view, so recipes only use it when a role asks for it |

Two suffixes extend a name:
- `_2`, `_3`, …: a further camera with the same placement, such as the second of two side cameras (`exterior_side_2`) or a second camera on one wrist (`wrist_2`).
- `_stereo`: the second image of a stereo pair, whose first image has the plain name (`exterior_top`, `exterior_top_stereo`).

## Mapping

A camera's unified name is looked up in this order:
1. `datasets.<dataset>.robots.<robot_name>`: names for one robot setup of a dataset, for datasets whose setups use one camera name for different views.
2. `datasets.<dataset>`: names for one dataset, by repository name without its namespace. A duplicate copy uses the dataset of its original.
3. `biarm_aliases`: names for bimanual episodes. `left` and `right` are wrist cameras on bimanual robots and exterior cameras on single-arm robots.
4. `aliases`: names for every dataset.

In a single-arm episode, a lone `wrist_left` or `wrist_right` becomes `wrist`.

A mapping to `null` leaves the camera out, as for infrared images. Training skips an episode none of whose cameras has a unified name, so setting every camera of a dataset to `null` removes it from training data. A mapping to `unknown` also leaves it out, and reports it as an issue: the camera is known, but which view it shows is not settled.

`camera_name_issues` lists, per episode, the cameras with no mapping, the cameras marked `unknown`, and unified names that two cameras of the episode share. The command prints how often each issue occurs.

## Checking the mapping

`oopsie index camera-sheet` writes a page with one row per dataset and camera combination: the middle frame of every camera of one episode, labelled with its recorded and unified name. Rows with issues come first. When a new dataset arrives, its cameras show up as issues in the `camera-names` output; add them to `configs/camera_names.yaml` after checking their frames on the sheet.

The placements (`front`, `side`, `top`, `back`) are judged from a single frame, relative to the robot base. A camera at about 45° is ambiguous between front and side.

## Running it by hand

```bash
uv run oopsie index camera-names --index episode_index.parquet
uv run oopsie index camera-sheet --data-dir <mirror> --index episode_index.parquet --out reports/camera_names.html
```
