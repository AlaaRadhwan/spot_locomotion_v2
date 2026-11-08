# Third-party material

Original code in this repository is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). The Apache-2.0 licence does **not** replace or override the licences of the third-party material below.

## Bundled assets

### Spot reference model: `src/spot_locomotion_v2/robot/mjcf/reference/`

| | |
|---|---|
| Content | `spot.xml`, `scene.xml` and 54 mesh files in `assets/`. The V3 model (`robot/mjcf/v3/`) loads these meshes via `meshdir="../reference/assets"` |
| Immediate source | `boston_dynamics_spot` in [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The meshes and `LICENSE` have the same checksums as a local Menagerie checkout (commit `accb6df`, 2026-06-15) |
| Upstream lineage (as stated by the Menagerie README) | Derived from the publicly available Spot URDF in the `spot_ros2` repository (originally `bdaiinstitute/spot_ros2`; now `rai-opensource/spot_ros2`) |
| Licence | BSD-3-Clause, Copyright (c) 2021, Clearpath Robotics Inc. Full text, unmodified, in [`src/spot_locomotion_v2/robot/mjcf/reference/LICENSE`](src/spot_locomotion_v2/robot/mjcf/reference/LICENSE) |
| Not verified here | The licence status of the original Boston Dynamics geometry upstream of Menagerie was not independently checked |

The same meshes are redistributed in the companion repository [`spot_ros2_ws`](https://github.com/AlaaRadhwan/spot_ros2_ws) with the same licence notice.

"Spot" is a Boston Dynamics product name. This project is not affiliated with or endorsed by Boston Dynamics, Clearpath Robotics or Google DeepMind.

### Boston Dynamics base URDF archive: `src/spot_locomotion_v2/robot/spot_base_urdf.zip`

| | |
|---|---|
| Content | `model.urdf` and 13 `link_models/*.obj` meshes (Blender 2.79 exports, file dates 2020-05-05); no README, licence or copyright text is embedded in the archive |
| Confirmed source | https://dev.bostondynamics.com/files/spot_base_urdf.zip. The local file is byte-identical (MD5 `c804f579e68d43f5d486e306b4980795`) to a fresh download of that URL |
| Official context | The same file is tracked in the `boston-dynamics/spot-sdk` repository at `files/spot_base_urdf.zip`, and the SDK documentation (`docs/concepts/robot_services.md`) links to it as "a urdf model of the base robot and its geometry" |
| Applicable licence | The repository's [Boston Dynamics SDK License (20191101-BDSDK-SL)](https://github.com/boston-dynamics/spot-sdk/blob/master/LICENSE). Complete copy, unmodified, in [`src/spot_locomotion_v2/robot/spot_base_urdf.LICENSE`](src/spot_locomotion_v2/robot/spot_base_urdf.LICENSE) |
| Basis for coverage | An inference, not an explicit statement: the licence grants rights to "the SDK Software" without listing files, and the archive is distributed inside the SDK repository and linked from its documentation. No separate licence for this archive was found |

Conditions of that licence that apply to this file:
- Any distribution must include the complete licence copy and retain copyright and proprietary notices (done via the file above; the archive itself contains no notices).
- The material must not be combined with other software in a way that subjects it to different terms. It is therefore kept as a separate, unmodified component that remains under the Boston Dynamics licence, not under this repository's Apache-2.0 licence.
- Use is licensed solely for use with Boston Dynamics products; software-based simulation is expressly not prohibited. This project uses it in simulation only.
- No use of the Boston Dynamics name or trademarks to endorse or promote products without written consent. Modified versions must be conspicuously marked (the archive is unmodified).

This repository does not reference the archive from any current code, model or task (a search of tracked source and tools found no use). It is retained as historical reference material.

### Original work

The V3 MJCF construction and audit tooling, actuator characterization scripts and reports, task and PPO configurations (including the robust command generator), and the export and benchmark tools were written for this project. `files.tar.xz` and the `robot/files/` directory contain copies of the project's own scripts.

## Dependencies (not vendored)

MjLab 1.5.3, MuJoCo, RSL-RL and PyTorch are installed as dependencies under their own licences. PPO is an existing published algorithm; the MjLab velocity-task framework is third-party and this project builds task configurations on top of it.
