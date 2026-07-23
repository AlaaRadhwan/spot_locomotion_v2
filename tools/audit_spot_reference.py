from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "src" / "spot_locomotion_v2"
    / "robot" / "mjcf" / "reference"
    / "spot.xml"
)

JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}

GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}

INTEGRATOR_NAMES = {
    int(mujoco.mjtIntegrator.mjINT_EULER): "euler",
    int(mujoco.mjtIntegrator.mjINT_RK4): "rk4",
    int(mujoco.mjtIntegrator.mjINT_IMPLICIT): "implicit",
    int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
}

CONE_NAMES = {
    int(mujoco.mjtCone.mjCONE_PYRAMIDAL): "pyramidal",
    int(mujoco.mjtCone.mjCONE_ELLIPTIC): "elliptic",
}

SOLVER_NAMES = {
    int(mujoco.mjtSolver.mjSOL_PGS): "pgs",
    int(mujoco.mjtSolver.mjSOL_CG): "cg",
    int(mujoco.mjtSolver.mjSOL_NEWTON): "newton",
}


def get_joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )

    if name is None:
        return f"<unnamed: {joint_id}>"

    return name

def get_actuator_name(
        model: mujoco.MjModel,
        actuator_id: int,
) -> str:
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id
    )

    if name is None:
        return f"<unnamed:{actuator_id}>"

    return name

def get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
    )

    if name is None:
        return f"<unnamed:{body_id}>"
    
    return name

def get_geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id,
    )

    if name is None:
        return f"<unnamed:{geom_id}>"

    return name

def get_keyframe_name(
    model: mujoco.MjModel,
    keyframe_id: int,
) -> str:
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        keyframe_id,
    )

    if name is None:
        return f"<unnamed:{keyframe_id}>"

    return name


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    print(f"Loaded model: {MODEL_PATH}")

    print(f"model.nbody: {model.nbody}")
    print(f"model.njnt: {model.njnt}")
    print(f"model.nq: {model.nq:}")
    print(f"model.nv: {model.nv}")
    print(f"model.nu: {model.nu}")
    print(f"model.ngeom: {model.ngeom}")
    print(f"model.nkey: {model.nkey}")

    print("\nJoints: ")

    for joint_id in range(model.njnt):
        joint_name = get_joint_name(model, joint_id)

        joint_type = model.jnt_type[joint_id]
        joint_type_name = JOINT_TYPE_NAMES[int(joint_type)]

        qpos_address = model.jnt_qposadr[joint_id]
        qvel_address = model.jnt_dofadr[joint_id]

        print(f"{joint_id:2d} | {joint_name:10s} | type={joint_type_name:7s} | qpos={int(qpos_address):2d} | qvel={int(qvel_address):2d}")

    print("\nActuated joint properties: ")

    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])

        if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue

        joint_name = get_joint_name(model, joint_id)
        dof_id = int(model.jnt_dofadr[joint_id])

        axis = model.jnt_axis[joint_id]
        limits = model.jnt_range[joint_id]
        limited = model.jnt_limited[joint_id]

        damping = model.dof_damping[dof_id]
        frictionloss = model.dof_frictionloss[dof_id]
        armature = model.dof_armature[dof_id]

        print(
            f"{joint_name:7s} | axis=[{axis[0]:5.2f}, {axis[1]:5.2f}, {axis[2]:5.2f}] | range=[{limits[0]:7.3f}, {limits[1]:7.3f}] | limited={limited} | damping={damping:5.2f} | friction={frictionloss:5.2f} | armature={armature:5.2f}"  
        )

    print("\nActuators:")

    for actuator_id in range(model.nu):
        actuator_name = get_actuator_name(model, actuator_id)

        # The first transmission ID is the joint ID
        # for a joint-transmission actuator.

        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = get_joint_name(model, joint_id)

        gear = model.actuator_gear[actuator_id, 0]
        
        control_limited = bool(model.actuator_ctrllimited[actuator_id])
        control_range = model.actuator_ctrlrange[actuator_id]

        force_limited = bool(model.actuator_forcelimited[actuator_id])
        force_range = model.actuator_forcerange[actuator_id]

        gain = model.actuator_gainprm[actuator_id, 0]
        bias = model.actuator_biasprm[actuator_id, 0:3]

        print(
            f"{actuator_id:2d} | {actuator_name:8s} | joint={joint_name:8s} | "
            f"gear={gear} | control_limited={control_limited} | control_range={control_range} | "
            f"force_limited={force_limited} | force_range={force_range} | gain={gain} | bias={bias}"
        )

    print("\nJoint-level actuator-force limits:")

    for joint_id in range(model.njnt):
        
        joint_type = int(model.jnt_type[joint_id])

        if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue

        joint_name = get_joint_name(model, joint_id)

        limited = bool(model.jnt_actfrclimited[joint_id])
        force_range = model.jnt_actfrcrange[joint_id]

        print(
            f"{joint_id:2d} | joint={joint_name:8s} | limited={limited} | force_range={force_range}"
        )


    print("\nConfirmed V1 actuator contract")
    print(30*"-")
    print(
        "Control input: desired joint position"
        "\nControl range: inherited joint position limits"
        "\nKp: 500"
        "\nKd: 40"
        "\nTransmission gear: 1"
        "\nPassive joint damping: 0"
        "\nJoint friction loss: 0"
        "\nArmature: 0"
        "\nJoint-level actuator-force range: [-1000, 1000]"
    )

    print("\nBody inertial properties:")

    total_mass = 0.0

    for body_id in range(model.nbody):
        body_name = get_body_name(model, body_id)

        parent_id = int(model.body_parentid[body_id])

        if body_id == 0:
            parent_name = "<none>"
        else:
            parent_name = get_body_name(model, parent_id)

        mass = model.body_mass[body_id]
        com = model.body_ipos[body_id]
        inertia = model.body_inertia[body_id]
        inertial_quaternion = model.body_iquat[body_id]

        total_mass += mass

        print(
            f"{body_id:2d} | {body_name:8s} | parent={parent_name:8s} | mass={mass:5.2f} | "
            f"COM={com} | inertia={inertia} | inertia_quat={inertial_quaternion}"
        )

    print(f"\nTotal compiled mass: {total_mass:.6f} kg")

    print("\nGeometry properties:")

    for geom_id in range(model.ngeom):
        geom_name = get_geom_name(model, geom_id)

        body_id = int(model.geom_bodyid[geom_id])
        body_name = get_body_name(model, body_id)

        geom_type_id = int(model.geom_type[geom_id])
        geom_type_name = GEOM_TYPE_NAMES[geom_type_id]

        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        friction = model.geom_friction[geom_id]
        size = model.geom_size[geom_id]
        position = model.geom_pos[geom_id]

        collision_enabled = contype != 0 or conaffinity != 0

        print(
            f"{geom_id:2d} | geom={geom_name:20s} | body={body_name:8s} | type={geom_type_name} | "
            f"collision={collision_enabled} | contype={contype} | conaffinity={conaffinity} | "
            f"friction={friction} | size={size} | position={position}"
        )

    print("\nCollision contact parameters:")

    for geom_id in range(model.ngeom):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])

        can_participate = contype != 0 or conaffinity != 0

        if not can_participate:
            continue

        geom_name = get_geom_name(model, geom_id)

        body_id = int(model.geom_bodyid[geom_id])
        body_name = get_body_name(model, body_id)

        condim = int(model.geom_condim[geom_id])
        priority = int(model.geom_priority[geom_id])
        solmix = float(model.geom_solmix[geom_id])
        solref = model.geom_solref[geom_id]
        solimp = model.geom_solimp[geom_id]
        margin = float(model.geom_margin[geom_id])
        gap = float(model.geom_gap[geom_id])

        print(
            f"{geom_id:2d} | "
            f"geom={geom_name:20s} | "
            f"body={body_name:8s} | "
            f"condim={condim} | "
            f"priority={priority} | "
            f"solmix={solmix:.3f} | "
            f"solref={solref} | "
            f"solimp={solimp} | "
            f"margin={margin:.4f} | "
            f"gap={gap:.4f}"
        )

    print("\nGlobal simulation options:")

    options = model.opt

    integrator_id = int(options.integrator)
    cone_id = int(options.cone)
    solver_id = int(options.solver)

    integrator_name = INTEGRATOR_NAMES[integrator_id]
    cone_name = CONE_NAMES[cone_id]
    solver_name = SOLVER_NAMES[solver_id]

    print(
        f"timestep={options.timestep}\n"
        f"gravity={options.gravity}\n"
        f"integrator={integrator_name}\n"
        f"cone={cone_name}\n"
        f"solver={solver_name}\n"
        f"impratio={options.impratio}\n"
        f"iterations={options.iterations}\n"
        f"tolerance={options.tolerance}\n"
        f"noslip_iterations={options.noslip_iterations}\n"
        f"noslip_tolerance={options.noslip_tolerance}"
    )

    print("\nKeyframes:")

    for keyframe_id in range(model.nkey):
        keyframe_name = get_keyframe_name(
            model,
            keyframe_id,
        )

        qpos = model.key_qpos[keyframe_id]
        qvel = model.key_qvel[keyframe_id]
        ctrl = model.key_ctrl[keyframe_id]

        print(
            f"\nKeyframe {keyframe_id}: {keyframe_name}\n"
            f"  {'Base position':<16}: {qpos[0:3]}\n"
            f"  {'Base quaternion':<16}: {qpos[3:7]}\n"
            f"  {'Joint positions':<16}: {qpos[7:19]}\n"
            f"  {'Velocity':<16}: {qvel}\n"
            f"  {'Control':<16}: {ctrl}"
        )


if __name__ == "__main__":
    main()