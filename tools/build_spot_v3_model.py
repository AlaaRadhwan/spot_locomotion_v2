#!/usr/bin/env python3
"""Build the Spot V3 rigid-body MJCF from the existing reference Spot MJCF.

V3 keeps the existing project geometry, joint hierarchy, collisions, sensors,
joint names, and home pose, but replaces the rigid-body inertial properties
with those extracted from the flattened Isaac Sim Spot asset.

This script deliberately does NOT choose an actuator model.  The generated
V3 MJCF is a pure rigid-body model with no actuators so that actuator design
can be evaluated separately.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_XML = (
    REPO_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "reference"
    / "spot.xml"
)

V3_DIR = (
    REPO_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "v3"
)

OUTPUT_XML = V3_DIR / "spot.xml"
OUTPUT_SCENE_XML = V3_DIR / "scene.xml"


# Isaac Sim flattened USD rigid-body inertials.
#
# Important:
# - USD quatf is (real, i, j, k), i.e. wxyz.
# - MuJoCo MJCF quat is also wxyz.
# - The existing project link/body frames closely match the Isaac asset
#   frames, so these principal-axis quaternions can be transferred directly.
#
# We intentionally omit the four ~1e-6 kg Isaac foot helper bodies because
# this project already models the feet as contact geoms attached to the
# lower legs.  The expected articulated mass is therefore 31.5999994 kg.
ISAAC_INERTIALS = {
    "body": {
        "mass": "16.707651",
        "pos": "0 0 -0.00496172",
        "diaginertia": "0.08154379 0.5492312 0.5694544",
        "quat": "1 0 0 0",
    },

    "fl_hip": {
        "mass": "1.1368834",
        "pos": "-0.01586739 0.00855842 0.00000903",
        "diaginertia": "0.001149959 0.0016612653 0.0017290359",
        "quat": "0.9816311 0.0044215247 0.00104843 0.19073467",
    },
    "fr_hip": {
        "mass": "1.1368834",
        "pos": "-0.01586739 -0.00855842 0.00000903",
        "diaginertia": "0.001149959 0.0016612653 0.0017290359",
        "quat": "0.9816311 -0.0044215247 0.00104843 -0.19073467",
    },
    "hl_hip": {
        "mass": "1.1368834",
        "pos": "0.01586739 0.00855842 0.00000903",
        "diaginertia": "0.001149959 0.0016612653 0.0017290359",
        "quat": "0.9816311 0.0044215247 -0.00104843 -0.19073467",
    },
    "hr_hip": {
        "mass": "1.1368834",
        "pos": "0.01586739 -0.00855842 0.00000903",
        "diaginertia": "0.001149959 0.0016612653 0.0017290359",
        "quat": "0.9816311 -0.0044215247 -0.00104843 0.19073467",
    },

    "fl_uleg": {
        "mass": "2.2562037",
        "pos": "0.00214442 -0.01110184 -0.07881204",
        "diaginertia": "0.026939424 0.026025372 0.0029866563",
        "quat": "0.99868387 0.045455117 -0.010625258 -0.02124732",
    },
    "fr_uleg": {
        "mass": "2.2562037",
        "pos": "0.00214442 0.01110184 -0.07881204",
        "diaginertia": "0.026939424 0.026025372 0.0029866563",
        "quat": "0.99868387 -0.045455117 -0.010625258 0.02124732",
    },
    "hl_uleg": {
        "mass": "2.2562037",
        "pos": "0.00214442 -0.01110184 -0.07881204",
        "diaginertia": "0.026939424 0.026025372 0.0029866563",
        "quat": "0.99868387 0.045455117 -0.010625258 -0.02124732",
    },
    "hr_uleg": {
        "mass": "2.2562037",
        "pos": "0.00214442 0.01110184 -0.07881204",
        "diaginertia": "0.026939424 0.026025372 0.0029866563",
        "quat": "0.99868387 -0.045455117 -0.010625258 0.02124732",
    },

    "fl_lleg": {
        "mass": "0.33",
        "pos": "0.0059736 0 -0.17466427",
        "diaginertia": "0.0070141945 0.00709946 0.00014465585",
        "quat": "0.99998844 0 -0.004804094 0",
    },
    "fr_lleg": {
        "mass": "0.33",
        "pos": "0.0059736 0 -0.17466427",
        "diaginertia": "0.0070141945 0.00709946 0.00014465585",
        "quat": "0.99998844 0 -0.004804094 0",
    },
    "hl_lleg": {
        "mass": "0.33",
        "pos": "0.0059736 0 -0.17466427",
        "diaginertia": "0.0070141945 0.00709946 0.00014465585",
        "quat": "0.99998844 0 -0.004804094 0",
    },
    "hr_lleg": {
        "mass": "0.33",
        "pos": "0.0059736 0 -0.17466427",
        "diaginertia": "0.0070141945 0.00709946 0.00014465585",
        "quat": "0.99998844 0 -0.004804094 0",
    },
}


SCENE_XML = """<mujoco model="spot v3 scene">

  <include file="spot.xml"/>

  <visual>
    <headlight
      diffuse="0.6 0.6 0.6"
      ambient="0.3 0.3 0.3"
      specular="0 0 0"
    />
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture
      type="skybox"
      builtin="gradient"
      rgb1="0.3 0.5 0.7"
      rgb2="0 0 0"
      width="512"
      height="3072"
    />
    <texture
      name="groundplane_texture"
      type="2d"
      builtin="checker"
      mark="edge"
      rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8"
      width="300"
      height="300"
    />
    <material
      name="groundplane"
      texture="groundplane_texture"
      texuniform="true"
      texrepeat="5 5"
      reflectance="0.2"
    />
  </asset>

  <worldbody>
    <geom
      name="terrain"
      type="plane"
      size="0 0 0.01"
      material="groundplane"
    />
  </worldbody>

</mujoco>
"""


def find_named_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"Could not find body {name!r} in {SOURCE_XML}")
    return body


def replace_inertials(root: ET.Element) -> None:
    for body_name, params in ISAAC_INERTIALS.items():
        body = find_named_body(root, body_name)
        inertial = body.find("inertial")
        if inertial is None:
            raise RuntimeError(f"Body {body_name!r} has no <inertial> element")

        inertial.set("mass", params["mass"])
        inertial.set("pos", params["pos"])
        inertial.set("diaginertia", params["diaginertia"])
        inertial.set("quat", params["quat"])


def remove_actuator_definition(root: ET.Element) -> None:
    # Remove the reference model's native position actuators.  Actuator
    # selection belongs to the next V3 phase rather than the rigid-body model.
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)

    # Remove actuator-only defaults from class="spot".
    spot_default = root.find("./default/default[@class='spot']")
    if spot_default is not None:
        joint_default = spot_default.find("joint")
        if joint_default is not None:
            joint_default.attrib.pop("actuatorfrcrange", None)

        position_default = spot_default.find("position")
        if position_default is not None:
            spot_default.remove(position_default)

    # The home keyframe remains useful, but ctrl is invalid when nu == 0.
    for key in root.findall("./keyframe/key"):
        key.attrib.pop("ctrl", None)


def configure_v3_paths(root: ET.Element) -> None:
    root.set("model", "spot_v3_isaac_dynamics")

    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("Reference MJCF has no <compiler> element")

    # V3 reuses the existing reference meshes without duplicating them.
    compiler.set("meshdir", "../reference/assets")


def write_tree(tree: ET.ElementTree, path: Path) -> None:
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    if not SOURCE_XML.is_file():
        raise FileNotFoundError(
            "Reference Spot MJCF not found.\n"
            f"Expected: {SOURCE_XML}\n"
            "Run this script from a checkout of spot_locomotion_v2."
        )

    V3_DIR.mkdir(parents=True, exist_ok=True)

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(SOURCE_XML, parser=parser)
    root = tree.getroot()

    configure_v3_paths(root)
    replace_inertials(root)
    remove_actuator_definition(root)

    write_tree(tree, OUTPUT_XML)
    OUTPUT_SCENE_XML.write_text(SCENE_XML, encoding="utf-8")

    total_mass = sum(float(x["mass"]) for x in ISAAC_INERTIALS.values())

    print("Created Spot V3 rigid-body model:")
    print(f"  {OUTPUT_XML}")
    print(f"  {OUTPUT_SCENE_XML}")
    print()
    print(f"Expected articulated mass: {total_mass:.7f} kg")
    print("Actuators intentionally omitted: yes")
    print("Reference meshes reused: yes")
    print()
    print("Next:")
    print("  uv run python tools/audit_spot_v3_model.py")


if __name__ == "__main__":
    main()
