# code_realworld — Piper Arm Real-Hardware Control

Real-robot control code for the **AgileX Piper 6-DOF arm**,
communicating over CAN bus via `piper_sdk`.

---

## Files

| File | Description |
|------|-------------|
| `piper_control.py` | High-level motion wrapper (`PiperArmController`) — MOVE_J / MOVE_P / MOVE_L / gripper / teach mode |
| `impedance_control_mit.py` | True joint-level impedance controller (`PiperMitImpedance`) — draws letters U / C / L / A using `JointMitCtrl` (MIT interface) with Pinocchio IK |

---

## Dependencies

### Conda environment

Both scripts run inside the `piper_work_env` conda environment:

```bash
conda activate piper_work_env
```

`impedance_control_mit.py` additionally requires **Pinocchio**:

```bash
pip install pin
```

### External requirements

| Requirement | Notes |
|-------------|-------|
| `piper_sdk` | `C_PiperInterface_V2`; must be installed in the conda env |
| CAN interface | `can0` (configured via `ip link set can0 up type can bitrate 1000000`) |
| URDF file | `~/piper_robot/piper_ros/src/piper_description/urdf/piper_description.urdf` (used only by `impedance_control_mit.py`) |

---

## Unit Conventions

| Quantity | Input to API | SDK internal |
|----------|-------------|--------------|
| Joint angles | rad | 0.001 deg |
| End-effector position | mm | 0.001 mm |
| End-effector orientation | deg (Euler) | 0.001 deg |
| Gripper opening | m | 0.001 mm |

---

## piper_control.py

`PiperArmController` wraps the low-level SDK into readable motion commands.

### Quick start

```python
from piper_control import PiperArmController

arm = PiperArmController(
    can_name="can0",
    home_joints_rad=[1.57, 0.866, -0.960, 0.0, 0.182, 1.571],
    gripper_open_m=0.03,
    gripper_close_m=0.00,
)

arm.connect()
arm.enable()
arm.gripper_enable()
arm.set_ctrl_mode2can()   # STANDBY / TEACH -> CAN control
arm.goto_home(speed=30)
```

### Key methods

| Method | Description |
|--------|-------------|
| `connect()` | Open CAN port and start SDK read thread |
| `enable()` | Enable the arm (retries until success) |
| `set_ctrl_mode2can()` | Transition from STANDBY or TEACH mode to CAN control mode |
| `goto_home(speed)` | Move to `home_joints_rad` via MOVE_J |
| `move_joints(joints_rad, speed, wait)` | Joint-space move (MOVE_J) |
| `move_end_pose(pose, speed, wait)` | Cartesian move (MOVE_P) |
| `move_line(pose, speed, wait)` | Linear Cartesian move (MOVE_L) |
| `gripper_open()` / `gripper_close()` | Gripper control |
| `teach_and_set_home()` | Enter drag-teach mode; read and save current joints as new home |
| `emergency_stop()` / `emergency_resume()` | Hardware emergency stop |
| `update()` | Pull latest feedback into `self.joint_rad`, `self.end_pose`, `self.gripper` |

### Finding the home position via drag-teach

```bash
python piper_control.py
# Enter 't' when prompted -> manually move the arm -> press Enter to record
```

The script prints the recorded `home_joints_rad` values; copy them into the
`PiperArmController` constructor.

---

## impedance_control_mit.py

`PiperMitImpedance` implements **true joint-level impedance control** using
the Piper MIT interface (`JointMitCtrl`).

### Why MIT mode instead of position control

`EndPoseCtrl` (used in `piper_control.py`) goes through the on-board
high-stiffness servo — the arm cannot be back-driven by external forces.
`JointMitCtrl` sends low-level torque commands directly:

```
τ = kp · (q_ref − q) + kd · (dq_ref − dq)
```

With small `kp` / `kd`, each joint becomes a compliant spring-damper,
enabling safe contact with the canvas.

### Letter drawing

Run from the command line with `--letter`:

```bash
conda activate piper_work_env
python impedance_control_mit.py --letter U   # default
python impedance_control_mit.py --letter C
python impedance_control_mit.py --letter L
python impedance_control_mit.py --letter A   # narrow arch shape
```

**Motion sequence:**

1. Connect and enable the arm
2. Move to init position (`joint1 = 1.57 rad`, joints 2-6 = 0)
3. Move to home / drawing start position
4. Ramp into MIT mode (2 s ramp) → trace letter stroke in the XZ plane
5. On exit (`q` or duration): ramp gains to zero → smooth handoff back to MOVE_J → return to init position

**Letter coordinate convention** (local XZ plane):
- `(0, 0)` = letter start point (top-left)
- `+x` → right
- `+z` → up (downward strokes use negative z)
- Y dimension is fixed (pen depth into canvas)

### Impedance gains (default)

```python
MitImpedanceParams(
    mit_kp    = [10.0, 8.0, 8.0, 8.0, 6.0, 6.0],  # N·m / rad per joint
    mit_kd    = [ 0.8, 0.6, 0.6, 0.6, 0.5, 0.5],  # N·m·s / rad per joint
    stiffness = 0.3,           # global scale: kp_eff = kp * stiffness
    max_joint_err_rad = 0.35,  # torque cap: τ_max = kp_eff * max_err (~20 deg)
    rate_hz   = 200.0,
    ramp_s    = 2.0,
)
```

Effective gains: `kp_eff = mit_kp × stiffness`, `kd_eff = mit_kd × √stiffness`.

### Key classes

| Class | Description |
|-------|-------------|
| `PiperKinematics` | Pinocchio wrapper for FK and damped least-squares IK (6-DOF reduced model, gripper joints locked) |
| `LetterPath` | 2D stroke builder — chain `.line_to()`, `.arc()`, `.ellipse_arc()` calls |
| `MitImpedanceParams` | Dataclass for gains and timing knobs |
| `PiperMitImpedance` | Main controller — `run_to_target_mit()`, `run_trajectory_mit()`, `run_letter_mit()` |

### URDF requirement

Pinocchio needs the Piper URDF. Default path:

```
~/piper_robot/piper_ros/src/piper_description/urdf/piper_description.urdf
```

Override with the `--urdf` argument or `urdf_path` constructor parameter.

---

## CAN Bus Setup

Before running either script, bring up the CAN interface:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
```

Verify the interface is up:

```bash
ip link show can0
```
