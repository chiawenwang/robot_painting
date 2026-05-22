#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_base_module():
    script_dir = Path(__file__).resolve().parent
    base_file = script_dir / "0521.py"
    spec = importlib.util.spec_from_file_location("painting_base", base_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base script: {base_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_task(base):
    # Use the existing Piper model + wall patch, and set a 15 cm brush.
    base.EXTERNAL_BRUSH_OFFSET = base.np.array([0.0, 0.0, 0.15], dtype=float)
    base.USE_TOUCH_SENSOR = False

    # Slightly moving rigid wall.
    base.WALL_PROFILE_MODE = "rigid"
    base.WALL_MOTION_ENABLED = True
    base.WALL_PUSH_DISTANCE = 0.008
    base.WALL_RETREAT_DISTANCE = 0.006

    # Contact mechanics assumptions (brush + wall).
    base.CONTACT_STIFFNESS = 3000.0  # N/m
    base.CONTACT_DAMPING = 60.0      # N*s/m
    base.F_DES = 10.0                # N

    # UCLA painting path.
    base.PAINT_TEXT = "UCLA"
    base.LETTER_WIDTH = 0.055
    base.LETTER_HEIGHT = 0.095
    base.LETTER_SPACING = 0.015
    base.LETTER_STROKE_SPEED = 0.055
    base.PEN_UP_TRAVEL_SPEED = 0.11
    base.SIM_DURATION = 20.0
    base.PAINT_START_DELAY = 0.8
    base.PAINT_END_HOLD = 1.0
    # Rebuild global paint plan so updated geometry/speeds take effect.
    base.PAINT_PLAN = base.build_paint_plan()


def run_impedance_case(base):
    model, model_source = base.load_model()
    data = base.mujoco.MjData(model)
    ik_data = base.mujoco.MjData(model)

    site_id = base.mujoco.mj_name2id(model, base.mujoco.mjtObj.mjOBJ_SITE, base.EE_SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Site '{base.EE_SITE_NAME}' not found in model.")

    qpos_idx, dof_idx, joint_lower, joint_upper = base.get_joint_indices(model, base.JOINT_NAMES)
    act_idx = base.get_joint_actuator_indices(model, base.JOINT_NAMES)

    # Cartesian normal-direction impedance params (stabilized for position-actuated robot).
    M_imp = 1.6
    B_imp = 145.0
    K_imp = 560.0
    pen_up_clearance = 0.006
    ex_limit = 0.035
    v_limit = 0.18
    x_cmd_rate_limit = 0.12  # m/s
    force_lpf_alpha = 0.25

    # Initial pose.
    q_seed = base.np.zeros(len(base.JOINT_NAMES))
    p0, _ = base.painting_nominal_pose(0.0)
    q_init = base.ik_damped_least_squares(
        model,
        ik_data,
        site_id,
        p0,
        q_seed,
        qpos_idx,
        dof_idx,
        joint_lower,
        joint_upper,
    )
    data.qpos[qpos_idx] = q_init
    data.qvel[:] = 0.0
    base.mujoco.mj_forward(model, data)
    paint_mark_site_ids = base.get_paint_mark_site_ids(model)
    base.reset_paint_marks(model, paint_mark_site_ids)
    next_paint_mark_index = 0
    last_paint_mark_local_yz = None

    sim_time = max(base.SIM_DURATION, base.effective_sim_duration())
    steps = int(sim_time / base.DT)
    frames = []
    renderer = None
    camera = None
    scene_option = None
    if base.should_attempt_rendering():
        renderer, camera, scene_option = base.make_renderer(model)

    x_imp = float(data.site_xpos[site_id][0])
    x_imp_dot = 0.0
    x_ref_prev = x_imp
    x_cmd_prev = x_imp
    f_meas_filt = 0.0
    log_t, log_f, log_fdes = [], [], []
    log_xcmd, log_xee, log_xref = [], [], []

    for i in range(steps):
        t = i * base.DT

        nominal_pos, pen_down = base.painting_nominal_pose(t)
        ee_pos = base.get_ee_pose(model, data, site_id)
        contact = base.contact_force(model, data, site_id, t)

        wall_surface_x = contact["wall_surface_x"]
        desired_penetration = base.F_DES / base.CONTACT_STIFFNESS
        x_ref_contact = wall_surface_x - base.BRUSH_RADIUS + desired_penetration
        x_ref = x_ref_contact if pen_down else (x_ref_contact - pen_up_clearance)

        f_des = base.F_DES if pen_down else 0.0
        f_meas_raw = float(contact["force_used"])
        if i == 0:
            f_meas_filt = f_meas_raw
        else:
            f_meas_filt = force_lpf_alpha * f_meas_raw + (1.0 - force_lpf_alpha) * f_meas_filt
        f_error = f_des - f_meas_filt

        # True impedance dynamics in wall-normal direction:
        # M*e_ddot + B*e_dot + K*e = F_des - F_meas
        x_ref_dot = 0.0 if i == 0 else float((x_ref - x_ref_prev) / base.DT)
        e = x_imp - x_ref
        e_dot = x_imp_dot - x_ref_dot
        e_ddot = (f_error - B_imp * e_dot - K_imp * e) / M_imp

        x_imp_dot = float(base.np.clip(x_imp_dot + e_ddot * base.DT, -v_limit, v_limit))
        x_imp = float(x_imp + x_imp_dot * base.DT)
        x_imp = float(base.np.clip(x_imp, x_ref - ex_limit, x_ref + ex_limit))

        x_cmd_target = x_imp
        max_step = x_cmd_rate_limit * base.DT
        x_cmd = float(base.np.clip(x_cmd_target, x_cmd_prev - max_step, x_cmd_prev + max_step))
        x_ref_prev = x_ref
        x_cmd_prev = x_cmd
        target = nominal_pos.copy()
        target[0] = x_cmd

        q_curr = data.qpos[qpos_idx].copy()
        q_cmd = base.ik_damped_least_squares(
            model,
            ik_data,
            site_id,
            target,
            q_curr,
            qpos_idx,
            dof_idx,
            joint_lower,
            joint_upper,
        )
        data.ctrl[act_idx] = q_cmd

        if pen_down and f_meas_filt > 0.15:
            ee_pos_mark = base.get_ee_pose(model, data, site_id)
            next_paint_mark_index, last_paint_mark_local_yz = base.maybe_add_paint_mark(
                model,
                paint_mark_site_ids,
                next_paint_mark_index,
                ee_pos_mark,
                last_paint_mark_local_yz,
            )

        log_t.append(t)
        log_f.append(f_meas_filt)
        log_fdes.append(f_des)
        log_xcmd.append(x_cmd)
        log_xee.append(float(ee_pos[0]))
        log_xref.append(x_ref)

        if renderer is not None and (i % base.RENDER_EVERY_N_STEPS == 0):
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(base.np.asarray(pixels).copy())

        base.mujoco.mj_step(model, data)

    out_dir = base.get_output_dir() / "ucla_true_impedance"
    out_dir.mkdir(parents=True, exist_ok=True)

    base.plt.figure(figsize=(10, 7))
    ax1 = base.plt.subplot(2, 1, 1)
    ax1.plot(log_t, log_f, label="measured force")
    ax1.plot(log_t, log_fdes, "--", label="desired force")
    ax1.set_ylabel("Force [N]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = base.plt.subplot(2, 1, 2)
    ax2.plot(log_t, log_xcmd, label="x_cmd")
    ax2.plot(log_t, log_xee, label="ee_x")
    ax2.plot(log_t, log_xref, "--", label="x_ref")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("X [m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    base.plt.tight_layout()
    fig_path = out_dir / "impedance_response.png"
    base.plt.savefig(fig_path, dpi=150)
    base.plt.close()
    mp4_path, gif_path = (None, None)
    if frames:
        mp4_path, gif_path = base.save_render_outputs(frames, out_dir)
    if renderer is not None:
        renderer.close()

    print("=== True Impedance Control Run Complete ===")
    print(f"Model source: {model_source}")
    print("Controller: M*e_ddot + B*e_dot + K*e = F_des - F_meas")
    print(f"Output dir: {out_dir}")
    print(f"Figure: {fig_path}")
    if mp4_path is not None:
        print(f"MP4: {mp4_path}")
    if gif_path is not None:
        print(f"GIF: {gif_path}")


def main():
    base = load_base_module()
    configure_task(base)
    run_impedance_case(base)


if __name__ == "__main__":
    main()
