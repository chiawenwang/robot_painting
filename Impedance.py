#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List


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
    base.LETTER_STROKE_SPEED = 0.020
    base.PEN_UP_TRAVEL_SPEED = 0.11
    base.SIM_DURATION = 20.0
    base.PAINT_START_DELAY = 0.8
    base.PAINT_END_HOLD = 1.0
    # Make text readable from the viewer/user side in your current camera setup.
    base.TEXT_READABLE_FROM_ROBOT_SIDE = True
    # Keep base plan for compatibility; runtime uses a local strict plan in this file.
    base.PAINT_PLAN = base.build_paint_plan()
    # Improve Cartesian tracking fidelity.
    base.IK_MAX_ITERS = 60
    base.IK_DAMPING = 1.5e-3


def build_local_ucla_plan(base):
    w = base.LETTER_WIDTH
    h = base.LETTER_HEIGHT
    s = base.LETTER_SPACING
    total_width = 4 * w + 3 * s
    x0 = -0.5 * total_width
    z_bottom = -0.5 * h
    z_top = 0.5 * h
    z_mid = 0.02 * h

    settle_before_stroke_s = float(getattr(base, "PRE_STROKE_SETTLE_TIME", 0.40))
    corner_hold_c_s = float(getattr(base, "CORNER_HOLD_C_TIME", 0.14))
    corner_hold_a_s = float(getattr(base, "CORNER_HOLD_A_TIME", 0.14))
    c_a_speed_scale = float(getattr(base, "CA_STROKE_SPEED_SCALE", 0.35))
    pen_up_clearance = float(getattr(base, "PEN_UP_Z_OFFSET", 0.03))
    base.PEN_UP_Z_OFFSET = pen_up_clearance

    def letter_origin(letter_index):
        return x0 + letter_index * (w + s)

    # Each stroke spec: points, speed_scale, optional corner_holds(after stroke end)
    letters: List[List[object]] = []

    u0 = letter_origin(0)
    letters.append(
        [
            {
                "points": base.np.array([[u0, z_top], [u0, z_bottom], [u0 + w, z_bottom], [u0 + w, z_top]], dtype=float),
                "speed_scale": 1.0,
                "corner_holds": [],
            }
        ]
    )

    c0 = letter_origin(1)
    letters.append(
        [
            {
                # C one-stroke path: top-right -> top-left -> down-left -> bottom-right
                "points": base.np.array(
                    [
                        [c0 + w, z_top],
                        [c0 + 0.65 * w, z_top],
                        [c0 + 0.30 * w, z_top],
                        [c0, z_top],
                        [c0, 0.50 * z_top],
                        [c0, 0.0],
                        [c0, 0.50 * z_bottom],
                        [c0, z_bottom],
                        [c0 + 0.30 * w, z_bottom],
                        [c0 + 0.65 * w, z_bottom],
                        [c0 + w, z_bottom],
                    ],
                    dtype=float,
                ),
                "speed_scale": c_a_speed_scale,
                "corner_holds": [corner_hold_c_s, corner_hold_c_s],
            },
        ]
    )

    l0 = letter_origin(2)
    letters.append(
        [
            {
                "points": base.np.array([[l0, z_top], [l0, z_bottom], [l0 + w, z_bottom]], dtype=float),
                "speed_scale": 1.0,
                "corner_holds": [],
            }
        ]
    )

    a0 = letter_origin(3)
    letters.append(
        [
            {
                # A one-stroke path: left leg up -> right leg down -> move to crossbar and draw
                "points": base.np.array(
                    [
                        [a0, z_bottom],
                        [a0 + 0.25 * w, 0.5 * z_top],
                        [a0 + 0.5 * w, z_top],
                        [a0 + 0.75 * w, 0.5 * z_top],
                        [a0 + w, z_bottom],
                        [a0 + 0.75 * w, z_mid],
                        [a0 + 0.25 * w, z_mid],
                    ],
                    dtype=float,
                ),
                "speed_scale": c_a_speed_scale,
                "corner_holds": [corner_hold_a_s],
            },
        ]
    )

    segments = []
    cursor = None
    for letter_idx, letter_strokes in enumerate(letters):
        first_stroke = letter_strokes[0]["points"]
        if cursor is not None and base.np.linalg.norm(first_stroke[0] - cursor) > 1e-9:
            safe_z = max(float(cursor[1]), float(first_stroke[0, 1])) + pen_up_clearance
            lift_point = base.np.array([cursor[0], safe_z], dtype=float)
            travel_point = base.np.array([first_stroke[0, 0], safe_z], dtype=float)
            descend_point = first_stroke[0].copy()

            segments.append({"points": base.np.vstack([cursor, lift_point]), "pen_down": False, "speed": base.PEN_UP_TRAVEL_SPEED, "letter_idx": letter_idx})
            segments.append({"points": base.np.vstack([lift_point, travel_point]), "pen_down": False, "speed": base.PEN_UP_TRAVEL_SPEED, "letter_idx": letter_idx})
            segments.append({"points": base.np.vstack([travel_point, descend_point]), "pen_down": False, "speed": base.PEN_UP_TRAVEL_SPEED, "letter_idx": letter_idx})
            cursor = descend_point.copy()

        for stroke_spec in letter_strokes:
            stroke = stroke_spec["points"]
            stroke_speed = base.LETTER_STROKE_SPEED * float(stroke_spec.get("speed_scale", 1.0))
            if cursor is not None and base.np.linalg.norm(stroke[0] - cursor) > 1e-9:
                segments.append({"points": base.np.vstack([cursor, stroke[0]]), "pen_down": False, "speed": base.PEN_UP_TRAVEL_SPEED, "letter_idx": letter_idx})

            # Pre-stroke settle hold (pen-up) so controller can converge before contact.
            segments.append(
                {
                    "points": base.np.vstack([stroke[0], stroke[0]]),
                    "pen_down": False,
                    "speed": base.PEN_UP_TRAVEL_SPEED,
                    "duration_override": settle_before_stroke_s,
                    "letter_idx": letter_idx,
                }
            )

            # Actual writing segment.
            segments.append({"points": stroke, "pen_down": True, "speed": stroke_speed, "letter_idx": letter_idx})

            # Optional short corner hold(s) for sharp turns.
            for hold_dt in stroke_spec.get("corner_holds", []):
                segments.append(
                    {
                        "points": base.np.vstack([stroke[-1], stroke[-1]]),
                        "pen_down": True,
                        "speed": stroke_speed,
                        "duration_override": float(hold_dt),
                        "letter_idx": letter_idx,
                    }
                )
            cursor = stroke[-1].copy()

    cumulative_time = base.PAINT_START_DELAY
    timed = []
    for segment in segments:
        length = base.polyline_length(segment["points"])
        if "duration_override" in segment:
            duration = float(segment["duration_override"])
        else:
            duration = 0.0 if length < 1e-9 else length / segment["speed"]
        timed.append(
            {
                "points": segment["points"],
                "pen_down": segment["pen_down"],
                "speed": segment["speed"],
                "letter_idx": int(segment.get("letter_idx", -1)),
                "t_start": cumulative_time,
                "t_end": cumulative_time + duration,
                "length": length,
            }
        )
        cumulative_time += duration

    def local_to_world(local_yz):
        y = base.PAINT_Y_CENTER - local_yz[0] if base.TEXT_READABLE_FROM_ROBOT_SIDE else base.PAINT_Y_CENTER + local_yz[0]
        return base.np.array([base.X_NOMINAL_BASE, y, base.PAINT_Z_CENTER + local_yz[1]], dtype=float)

    total_duration = cumulative_time + base.PAINT_END_HOLD
    first_local_point = letters[0][0]["points"][0]
    last_local_point = letters[-1][-1]["points"][-1]
    local_plan = {
        "segments": timed,
        "total_duration": total_duration,
        "start_point": local_to_world(first_local_point),
        "end_point": local_to_world(last_local_point),
    }
    if bool(getattr(base, "PLAN_DEBUG_PRINT", True)):
        print("\n=== Local UCLA Plan Segments ===")
        for i, seg in enumerate(local_plan["segments"]):
            p0 = seg["points"][0]
            p1 = seg["points"][-1]
            print(
                f"[{i:02d}] letter={seg['letter_idx']} pen_down={seg['pen_down']} "
                f"t=[{seg['t_start']:.3f}, {seg['t_end']:.3f}] "
                f"start=({p0[0]:.4f},{p0[1]:.4f}) end=({p1[0]:.4f},{p1[1]:.4f})"
            )
    return local_plan


def painting_nominal_pose_local(base, plan, t):
    if t <= base.PAINT_START_DELAY:
        return plan["start_point"].copy(), False, -1

    for segment in plan["segments"]:
        if segment["t_start"] <= t <= segment["t_end"] or (abs(t - segment["t_end"]) < 1e-9 and segment["t_end"] >= segment["t_start"]):
            distance = segment["speed"] * max(0.0, t - segment["t_start"])
            local_point_yz = base.interpolate_polyline(segment["points"], distance)
            world_y = base.PAINT_Y_CENTER - local_point_yz[0] if base.TEXT_READABLE_FROM_ROBOT_SIDE else base.PAINT_Y_CENTER + local_point_yz[0]
            return (
                base.np.array([base.X_NOMINAL_BASE, world_y, base.PAINT_Z_CENTER + local_point_yz[1]], dtype=float),
                segment["pen_down"],
                int(segment.get("letter_idx", -1)),
            )

    return plan["end_point"].copy(), False, -1


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
    ex_limit = 0.015
    v_limit = 0.12
    x_cmd_rate_limit = 0.08  # m/s
    force_lpf_alpha = 0.25
    yz_track_kp = 0.65
    yz_corr_limit = 0.020
    yz_track_kp_c = 1.20
    yz_corr_limit_c = 0.032
    q_step_limit = base.np.array([0.03, 0.03, 0.03, 0.04, 0.04, 0.025], dtype=float)
    wrist_hold_gain = 0.08

    # Initial pose.
    q_seed = base.np.zeros(len(base.JOINT_NAMES))
    local_plan = build_local_ucla_plan(base)
    p0, _, _ = painting_nominal_pose_local(base, local_plan, 0.0)
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
    q_ref = q_init.copy()

    sim_time = max(base.SIM_DURATION, float(local_plan["total_duration"]))
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
    log_y_nom, log_z_nom, log_y_ee, log_z_ee = [], [], [], []
    log_pen_down = []

    for i in range(steps):
        t = i * base.DT

        nominal_pos, pen_down, letter_idx = painting_nominal_pose_local(base, local_plan, t)
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
        # Shape-first correction: keep y-z close to writing trajectory while x follows impedance.
        yz_err = nominal_pos[1:3] - ee_pos[1:3]
        if pen_down and letter_idx == 1:
            yz_corr = base.np.clip(yz_track_kp_c * yz_err, -yz_corr_limit_c, yz_corr_limit_c)
        else:
            yz_corr = base.np.clip(yz_track_kp * yz_err, -yz_corr_limit, yz_corr_limit)
        target[1:3] = target[1:3] + yz_corr

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
        # Smooth joint commands and mildly suppress free wrist spinning.
        q_cmd = q_curr + base.np.clip(q_cmd - q_curr, -q_step_limit, q_step_limit)
        q_cmd[-1] = (1.0 - wrist_hold_gain) * q_cmd[-1] + wrist_hold_gain * q_ref[-1]
        data.ctrl[act_idx] = q_cmd

        log_t.append(t)
        log_f.append(f_meas_filt)
        log_fdes.append(f_des)
        log_xcmd.append(x_cmd)
        log_xee.append(float(ee_pos[0]))
        log_xref.append(x_ref)
        log_y_nom.append(float(nominal_pos[1]))
        log_z_nom.append(float(nominal_pos[2]))
        log_y_ee.append(float(ee_pos[1]))
        log_z_ee.append(float(ee_pos[2]))
        log_pen_down.append(bool(pen_down))

        if renderer is not None and (i % base.RENDER_EVERY_N_STEPS == 0):
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(base.np.asarray(pixels).copy())

        base.mujoco.mj_step(model, data)
        if pen_down:
            ee_pos_mark = base.get_ee_pose(model, data, site_id)
            next_paint_mark_index, last_paint_mark_local_yz = base.maybe_add_paint_mark(
                model,
                paint_mark_site_ids,
                next_paint_mark_index,
                ee_pos_mark,
                last_paint_mark_local_yz,
            )

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

    # Trajectory accuracy diagnostics in wall tangential plane (y-z).
    t_arr = base.np.asarray(log_t)
    y_nom = base.np.asarray(log_y_nom)
    z_nom = base.np.asarray(log_z_nom)
    y_ee = base.np.asarray(log_y_ee)
    z_ee = base.np.asarray(log_z_ee)
    pen_down_arr = base.np.asarray(log_pen_down, dtype=bool)
    yz_err = base.np.sqrt((y_ee - y_nom) ** 2 + (z_ee - z_nom) ** 2)
    yz_err_mm = 1000.0 * yz_err

    if base.np.any(pen_down_arr):
        yz_err_pen_mm = yz_err_mm[pen_down_arr]
        mean_err_mm = float(base.np.mean(yz_err_pen_mm))
        max_err_mm = float(base.np.max(yz_err_pen_mm))
    else:
        mean_err_mm = float("nan")
        max_err_mm = float("nan")

    fig_err = base.plt.figure(figsize=(11, 5))
    ax_traj = fig_err.add_subplot(1, 2, 1)
    ax_traj.plot(y_nom, z_nom, "--", linewidth=2.0, label="Nominal YZ path")
    ax_traj.plot(y_ee, z_ee, linewidth=2.0, label="Actual YZ path")
    ax_traj.set_xlabel("y [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_title("Path Shape: Nominal vs Actual")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()
    ax_traj.axis("equal")

    ax_err = fig_err.add_subplot(1, 2, 2)
    ax_err.plot(t_arr, yz_err_mm, linewidth=2.0, label="YZ tracking error")
    if base.np.any(pen_down_arr):
        ax_err.plot(t_arr[pen_down_arr], yz_err_mm[pen_down_arr], ".", markersize=2.0, label="Pen-down samples")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("error [mm]")
    ax_err.set_title("Tangential Tracking Error")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()
    fig_err.tight_layout()
    fig_err_path = out_dir / "yz_tracking_error.png"
    fig_err.savefig(fig_err_path, dpi=160)
    base.plt.close(fig_err)
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
    print(f"YZ error figure: {fig_err_path}")
    print(f"Mean YZ error (pen-down): {mean_err_mm:.2f} mm")
    print(f"Max YZ error (pen-down): {max_err_mm:.2f} mm")
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
