#!/usr/bin/env python3

# %% [markdown]
# # MuJoCo Admittance-Control Painting Demo
#
# This script is written in a notebook-friendly style so it can be copied into
# Google Colab or run directly as a `.py` file. It demonstrates a simplified
# 6-DOF "Piper-like" arm painting on a moving wall while regulating the normal
# contact force with a scalar admittance controller.
#
# Key ideas:
# - The wall first moves toward the robot, then eases back while staying no farther
#   than its original `x` position along the surface normal.
# - The brush follows a nominal tangential painting path in the wall's `y-z` plane.
# - A scalar admittance controller offsets the brush in `x` to regulate contact force.
# - If contact is lost, the controller performs a slow, safe search toward the wall.

# %% Imports and optional package installation
import importlib
import math
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
IN_COLAB = "google.colab" in sys.modules or os.getenv("COLAB_GPU") is not None
ALLOW_RUNTIME_PIP_INSTALL = IN_COLAB
FORCE_NONINTERACTIVE_MPL = sys.platform == "darwin" and not IN_COLAB

def choose_writable_dir(preferred_dir: Path, fallback_name: str) -> Path:
    """Return a writable directory, falling back to cwd or the system temp dir."""
    candidates = [
        preferred_dir,
        Path.cwd() / fallback_name,
        Path(tempfile.gettempdir()) / fallback_name,
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            continue

    raise RuntimeError(f"Could not create a writable directory for {fallback_name}.")


MATPLOTLIB_CACHE_DIR = choose_writable_dir(SCRIPT_DIR / ".matplotlib-cache", ".matplotlib-cache")
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
if FORCE_NONINTERACTIVE_MPL:
    os.environ.setdefault("MPLBACKEND", "Agg")


def ensure_package(import_name: str, pip_name: str | None = None):
    """Import a package, installing it first if needed."""
    try:
        return importlib.import_module(import_name)
    except ImportError:
        install_name = pip_name or import_name
        if ALLOW_RUNTIME_PIP_INSTALL:
            print(f"Installing missing package: {install_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            return importlib.import_module(import_name)

        raise RuntimeError(
            f"Missing Python package '{install_name}' in interpreter:\n"
            f"  {sys.executable}\n\n"
            "This script does not auto-install packages in local desktop runs.\n"
            "In PyCharm, set the interpreter to the same Python that works in your terminal,\n"
            "for example `/opt/miniconda3/bin/python3`, then install:\n"
            "  pip install mujoco numpy matplotlib imageio\n"
        ) from None


mujoco = ensure_package("mujoco")
np = ensure_package("numpy")
ensure_package("matplotlib")
plt = importlib.import_module("matplotlib.pyplot")
animation = importlib.import_module("matplotlib.animation")
imageio = ensure_package("imageio.v2", "imageio")
mujoco_viewer = importlib.import_module("mujoco.viewer")
try:
    imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
except ImportError:
    imageio_ffmpeg = None

try:
    ipython_display = importlib.import_module("IPython.display")
    IPyImage = ipython_display.Image
    display = ipython_display.display
except ImportError:
    IPyImage = None
    display = None

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def set_vis_flag(flag_container, flag_name: str, enabled: bool):
    """Set a MuJoCo viewer flag only when the enum exists in the installed version."""
    vis_flag = getattr(mujoco.mjtVisFlag, flag_name, None)
    if vis_flag is not None:
        flag_container.flags[vis_flag] = enabled


# %% Tuning parameters
# Admittance parameters:
# M_a: virtual mass. Larger values make the normal response slower.
# D_a: virtual damping. Larger values reduce oscillation.
# K_a: virtual stiffness. Larger values resist large normal offsets.
M_A = 1.0
D_A = 32.0
K_A = 55.0

# Desired normal contact force at the brush tip [N].
F_DES = 10.0

# Moving wall parameters:
# The wall surface normal points along world +x, which is also the approach direction.
# For a clean admittance-control demo, the wall uses a one-sided motion profile:
# it advances toward the brush, then retreats partway without going past its
# initial x position, which helps keep the paint stroke continuous.
WALL_X_CENTER = 0.66
WALL_HALF_THICKNESS = 0.01
WALL_HALF_WIDTH_Y = 0.26
WALL_HALF_HEIGHT_Z = 0.22
WALL_PUSH_DISTANCE = 0.011
WALL_RETREAT_DISTANCE = 0.007
WALL_MOTION_START = 2.0
WALL_PUSH_DURATION = 2.0
WALL_RETURN_DURATION = 2.4

# Contact model parameters for the virtual spring-damper estimate.
CONTACT_STIFFNESS = 2500.0
CONTACT_DAMPING = 45.0
CONTACT_LOSS_FORCE_THRESHOLD = 0.4

# Brush/tool geometry.
BRUSH_RADIUS = 0.015

# Simulation timing.
DT = 0.01
SIM_DURATION = 14.0

# Safety clamps for the admittance state and commanded normal motion.
MAX_DELTA_X = 0.05
MAX_DELTA_X_DOT = 0.15
APPROACH_SEARCH_SPEED = 0.008

# Nominal painting path in the wall's tangential plane.
PAINT_Y_CENTER = 0.0
PAINT_Z_CENTER = 0.30
PAINT_Y_AMPLITUDE = 0.04
PAINT_Z_AMPLITUDE = 0.02
PAINT_FREQUENCY = 0.12

# UCLA letter painting parameters.
PAINT_TEXT = "UCLA"
LETTER_WIDTH = 0.085
LETTER_HEIGHT = 0.135
LETTER_SPACING = 0.030
LETTER_STROKE_SPEED = 0.085
PEN_UP_TRAVEL_SPEED = 0.18
PEN_UP_CLEARANCE = 0.012
PAINT_START_DELAY = 0.8
PAINT_END_HOLD = 1.0
PAINT_MARK_SPACING = 0.005
PAINT_MARK_RADIUS = 0.005
PAINT_MARK_COUNT = 900
PAINT_MARK_OFFSET = 0.0015
PAINT_MARK_RGBA = np.array([0.15, 0.34, 0.68, 1.0], dtype=float)
TEXT_READABLE_FROM_ROBOT_SIDE = True

# IK parameters.
IK_MAX_ITERS = 25
IK_DAMPING = 2.5e-3
IK_TOL = 1.0e-4
IK_STEP_LIMIT = 0.12

# Rendering parameters.
RENDER_VIDEO = True
RENDER_WIDTH = 480
RENDER_HEIGHT = 360
RENDER_FPS = 20
RENDER_EVERY_N_STEPS = max(1, int(round(1.0 / (RENDER_FPS * DT))))
SAVE_ROBOT_3D_ANIMATION = True
ROBOT_ANIMATION_FPS = 20
ROBOT_ANIMATION_EVERY_N_STEPS = max(1, int(round(1.0 / (ROBOT_ANIMATION_FPS * DT))))

# Native MuJoCo viewer parameters.
SHOW_NATIVE_MUJOCO_VIEWER = True
VIEWER_REALTIME = True
VIEWER_SHOW_LEFT_UI = False
VIEWER_SHOW_RIGHT_UI = False
KEEP_VIEWER_OPEN_AFTER_SIM = True
SHOW_CONTACT_VISUALS = False
CAMERA_AZIMUTH = -38.0
CAMERA_ELEVATION = -11.0
CAMERA_DISTANCE = 0.96
CAMERA_LOOKAT = np.array([0.59, 0.0, PAINT_Z_CENTER + 0.02], dtype=float)
ROBOT_ANIMATION_ELEVATION = 18.0
ROBOT_ANIMATION_AZIMUTH = -40.0

# Saved render output.
SAVE_RENDER_GIF = True
SAVE_RENDER_MP4 = True

# Optional touch sensor usage. The manual contact estimate is usually more stable
# for this class-demo setup, so it is the default source.
USE_TOUCH_SENSOR = False

# If True, the demo approximates a very high-bandwidth inner joint servo by
# writing the IK solution directly into qpos before stepping MuJoCo. This keeps
# the notebook example stable and makes the admittance behavior easier to see.
USE_QUASI_STATIC_SERVO = True

# Paths and names for swapping in a real robot later.
EXTERNAL_MODEL_PATH = None
USE_REAL_PIPER_MODEL_IF_AVAILABLE = True
AUTO_FIND_LOCAL_PIPER_MODEL = True
AUTO_CLONE_PIPER_REPO = False
PIPER_REPO_URL = "https://github.com/agilexrobotics/piper_isaac_sim.git"
PIPER_REPO_BRANCH = "master"
PIPER_MODEL_RELATIVE_CANDIDATES = [
    Path("src/piper_description/mujoco_model/piper_description.xml"),
    Path("src/piper_description/mujoco_model/piper_no_gripper_description.xml"),
    Path("piper_description/mujoco_model/piper_description.xml"),
    Path("piper_description/mujoco_model/piper_no_gripper_description.xml"),
]
PIPER_ATTACHMENT_BODY_CANDIDATES = ["link6", "link7", "gripper_base", "tool_mount"]
EXTERNAL_BRUSH_OFFSET = np.array([0.0, 0.0, 0.165], dtype=float)
JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
EE_SITE_NAME = "brush_tip_site"
TOUCH_SENSOR_NAME = "brush_touch"
WALL_BODY_NAME = "moving_wall"


# Derived normal-direction references.
WALL_SURFACE_CENTER = WALL_X_CENTER - WALL_HALF_THICKNESS
DESIRED_PENETRATION = F_DES / CONTACT_STIFFNESS
X_NOMINAL_BASE = WALL_SURFACE_CENTER - BRUSH_RADIUS + DESIRED_PENETRATION
X_CMD_MIN = X_NOMINAL_BASE - 0.04
X_CMD_MAX = X_NOMINAL_BASE + 0.01


def paint_mark_sites_xml(indent="              "):
    lines = []
    for index in range(PAINT_MARK_COUNT):
        lines.append(
            f'{indent}<site name="paint_mark_{index:04d}" '
            f'pos="{-WALL_HALF_THICKNESS - PAINT_MARK_OFFSET:.6f} 0 0" '
            f'size="{PAINT_MARK_RADIUS:.6f}" '
            'rgba="0.15 0.34 0.68 0"/>'
        )
    return "\n".join(lines)


def local_to_world_yz(local_point_yz):
    world_y = PAINT_Y_CENTER - local_point_yz[0] if TEXT_READABLE_FROM_ROBOT_SIDE else PAINT_Y_CENTER + local_point_yz[0]
    return np.array(
        [X_NOMINAL_BASE, world_y, PAINT_Z_CENTER + local_point_yz[1]],
        dtype=float,
    )


def world_to_paint_plane_yz(world_y, world_z):
    local_y = PAINT_Y_CENTER - world_y if TEXT_READABLE_FROM_ROBOT_SIDE else world_y - PAINT_Y_CENTER
    local_z = world_z - PAINT_Z_CENTER
    return np.array([local_y, local_z], dtype=float)


def ucla_letter_strokes():
    """Return pen-down 2D polylines for drawing UCLA in the wall's y-z plane."""
    w = LETTER_WIDTH
    h = LETTER_HEIGHT
    s = LETTER_SPACING
    total_width = 4 * w + 3 * s
    x0 = -0.5 * total_width
    z_bottom = -0.5 * h
    z_top = 0.5 * h
    z_mid = 0.02 * h

    def letter_origin(letter_index):
        return x0 + letter_index * (w + s)

    strokes = []

    # U
    u0 = letter_origin(0)
    strokes.append(np.array([[u0, z_top], [u0, z_bottom], [u0 + w, z_bottom], [u0 + w, z_top]], dtype=float))

    # C
    c0 = letter_origin(1)
    strokes.append(
        np.array(
            [[c0 + w, z_top], [c0, z_top], [c0, z_bottom], [c0 + w, z_bottom]],
            dtype=float,
        )
    )

    # L
    l0 = letter_origin(2)
    strokes.append(np.array([[l0, z_top], [l0, z_bottom], [l0 + w, z_bottom]], dtype=float))

    # A outline and crossbar.
    a0 = letter_origin(3)
    strokes.append(np.array([[a0, z_bottom], [a0 + 0.5 * w, z_top], [a0 + w, z_bottom]], dtype=float))
    strokes.append(np.array([[a0 + 0.25 * w, z_mid], [a0 + 0.75 * w, z_mid]], dtype=float))

    return strokes


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def interpolate_polyline(points, distance_along):
    if len(points) == 1:
        return points[0].copy()

    remaining = float(np.clip(distance_along, 0.0, polyline_length(points)))
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length < 1e-9:
            continue
        if remaining <= segment_length:
            return start + (remaining / segment_length) * segment
        remaining -= segment_length
    return points[-1].copy()


def build_paint_plan():
    """Build a timed path that paints UCLA with pen-up travel between strokes."""
    strokes = ucla_letter_strokes()
    segments = []
    cursor = None

    for stroke in strokes:
        if cursor is not None and np.linalg.norm(stroke[0] - cursor) > 1e-9:
            segments.append({"points": np.vstack([cursor, stroke[0]]), "pen_down": False, "speed": PEN_UP_TRAVEL_SPEED})
        segments.append({"points": stroke, "pen_down": True, "speed": LETTER_STROKE_SPEED})
        cursor = stroke[-1].copy()

    cumulative_time = PAINT_START_DELAY
    timed_segments = []
    for segment in segments:
        length = polyline_length(segment["points"])
        duration = 0.0 if length < 1e-9 else length / segment["speed"]
        timed_segments.append(
            {
                "points": segment["points"],
                "pen_down": segment["pen_down"],
                "speed": segment["speed"],
                "length": length,
                "t_start": cumulative_time,
                "t_end": cumulative_time + duration,
            }
        )
        cumulative_time += duration

    total_duration = cumulative_time + PAINT_END_HOLD
    return {
        "segments": timed_segments,
        "total_duration": total_duration,
        "start_point": local_to_world_yz(strokes[0][0]),
        "end_point": local_to_world_yz(strokes[-1][-1]),
    }


PAINT_PLAN = build_paint_plan()


# %% Model construction
def build_simplified_piper_like_mjcf() -> str:
    """Return a self-contained MJCF string for a simple 6-DOF serial arm."""
    return dedent(
        f"""
        <mujoco model="piper_like_painting_arm">
          <compiler angle="degree" autolimits="true"/>
          <option timestep="{DT}" gravity="0 0 0" integrator="RK4" iterations="80" ls_iterations="20"/>
          <size njmax="1000" nconmax="500"/>

          <visual>
            <headlight ambient="0.45 0.45 0.45" diffuse="0.85 0.85 0.85" specular="0.15 0.15 0.15"/>
            <rgba haze="0.98 0.99 1.0 1"/>
          </visual>

          <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.92 0.96 1.0" rgb2="0.58 0.70 0.90" width="256" height="256"/>
          </asset>

          <default>
            <joint damping="3.0" armature="0.02"/>
            <geom density="550"/>
          </default>

          <worldbody>
            <light pos="0.0 -1.0 1.6" dir="0 1 -1"/>
            <geom name="ground" type="plane" pos="0 0 0" size="2 2 0.1"
                  rgba="0.96 0.96 0.96 1" contype="0" conaffinity="0"/>

            <body name="{WALL_BODY_NAME}" mocap="true" pos="{WALL_X_CENTER} 0 {PAINT_Z_CENTER}">
              <geom name="paint_wall" type="box" size="{WALL_HALF_THICKNESS} {WALL_HALF_WIDTH_Y} {WALL_HALF_HEIGHT_Z}"
                    rgba="0.75 0.82 0.92 1"
                    friction="0.8 0.02 0.002"
                    contype="1" conaffinity="1"/>
{paint_mark_sites_xml()}
            </body>

            <body name="robot_base" pos="0 0 {PAINT_Z_CENTER}">
              <geom name="base_geom" type="cylinder" size="0.055 0.05"
                    rgba="0.16 0.16 0.20 1" contype="0" conaffinity="0"/>

              <body name="link1" pos="0 0 0">
                <joint name="joint1" axis="0 0 1" range="-170 170"/>
                <geom name="link1_geom" type="capsule" fromto="0 0 0 0.14 0 0"
                      size="0.022" rgba="0.29 0.43 0.73 1" contype="0" conaffinity="0"/>

                <body name="link2" pos="0.14 0 0">
                  <joint name="joint2" axis="0 1 0" range="-110 110"/>
                  <geom name="link2_geom" type="capsule" fromto="0 0 0 0.12 0 0"
                        size="0.020" rgba="0.33 0.47 0.78 1" contype="0" conaffinity="0"/>

                  <body name="link3" pos="0.12 0 0">
                    <joint name="joint3" axis="0 1 0" range="-130 130"/>
                    <geom name="link3_geom" type="capsule" fromto="0 0 0 0.12 0 0"
                          size="0.019" rgba="0.37 0.51 0.82 1" contype="0" conaffinity="0"/>

                    <body name="link4" pos="0.12 0 0">
                      <joint name="joint4" axis="1 0 0" range="-180 180"/>
                      <geom name="link4_geom" type="capsule" fromto="0 0 0 0.10 0 0"
                            size="0.017" rgba="0.41 0.55 0.86 1" contype="0" conaffinity="0"/>

                      <body name="link5" pos="0.10 0 0">
                        <joint name="joint5" axis="0 1 0" range="-130 130"/>
                        <geom name="link5_geom" type="capsule" fromto="0 0 0 0.08 0 0"
                              size="0.015" rgba="0.46 0.60 0.90 1" contype="0" conaffinity="0"/>

                        <body name="link6" pos="0.08 0 0">
                          <joint name="joint6" axis="1 0 0" range="-180 180"/>
                          <geom name="link6_geom" type="capsule" fromto="0 0 0 0.06 0 0"
                                size="0.014" rgba="0.50 0.64 0.94 1" contype="0" conaffinity="0"/>

                          <body name="tool_mount" pos="0.06 0 0">
                            <geom name="wrist_geom" type="sphere" size="0.022"
                                  rgba="0.30 0.30 0.34 1" contype="0" conaffinity="0"/>
                            <geom name="brush_tip_geom" pos="0.02 0 0" type="sphere" size="{BRUSH_RADIUS}"
                                  rgba="0.95 0.28 0.20 1" friction="1.0 0.02 0.002"
                                  contype="1" conaffinity="1"/>
                            <site name="{EE_SITE_NAME}" pos="0.02 0 0" size="0.010"
                                  rgba="1.0 0.0 0.0 1"/>
                            <site name="brush_touch_site" pos="0.02 0 0" size="0.025"
                                  rgba="0.2 0.5 1.0 0.20"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>

          <actuator>
            <position name="act1" joint="joint1" kp="180"/>
            <position name="act2" joint="joint2" kp="180"/>
            <position name="act3" joint="joint3" kp="160"/>
            <position name="act4" joint="joint4" kp="110"/>
            <position name="act5" joint="joint5" kp="100"/>
            <position name="act6" joint="joint6" kp="90"/>
          </actuator>

          <sensor>
            <touch name="{TOUCH_SENSOR_NAME}" site="brush_touch_site"/>
          </sensor>
        </mujoco>
        """
    )


def maybe_clone_piper_repo(repo_root: Path):
    """Optionally clone the Piper ROS repository when running in Colab."""
    if repo_root.exists() or not AUTO_CLONE_PIPER_REPO:
        return repo_root if repo_root.exists() else None

    try:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                PIPER_REPO_BRANCH,
                PIPER_REPO_URL,
                str(repo_root),
            ]
        )
        return repo_root
    except Exception as exc:
        print(f"Could not clone Piper repository automatically. Falling back if needed. Reason: {exc}")
        return None


def candidate_piper_roots():
    """Common places where the Piper ROS repository may exist locally."""
    roots = []
    for env_name in ("PIPER_ASSET_REPO_DIR", "PIPER_ROS_DIR"):
        env_root = os.getenv(env_name)
        if env_root:
            roots.append(Path(env_root))

    cwd = Path.cwd()
    roots.extend(
        [
            SCRIPT_DIR,
            SCRIPT_DIR / "piper_isaac_sim",
            SCRIPT_DIR / "piper_ros",
            SCRIPT_DIR.parent / "piper_isaac_sim",
            SCRIPT_DIR.parent / "piper_ros",
            cwd,
            cwd / "piper_isaac_sim",
            cwd / "piper_ros",
            cwd.parent / "piper_isaac_sim",
            cwd.parent / "piper_ros",
            Path("/content/piper_isaac_sim"),
            Path("/content/piper_ros"),
            Path.home() / "piper_isaac_sim",
            Path.home() / "piper_ros",
            Path.home() / "Desktop" / "piper_isaac_sim",
            Path.home() / "Desktop" / "piper_ros",
        ]
    )

    unique_roots = []
    seen = set()
    for root in roots:
        if root is None:
            continue
        resolved = root.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique_roots.append(resolved)
    return unique_roots


def find_local_piper_model():
    """Return the first discovered local Piper MuJoCo XML file."""
    for root in candidate_piper_roots():
        should_try_clone = str(root).endswith("/content/piper_isaac_sim") or str(root).endswith("/content/piper_ros")
        actual_root = maybe_clone_piper_repo(root) if should_try_clone else root
        if actual_root is None:
            continue
        for relative_path in PIPER_MODEL_RELATIVE_CANDIDATES:
            candidate = actual_root / relative_path
            if candidate.exists():
                return candidate
    return None


def format_xyz(values):
    return " ".join(f"{float(v):.6f}" for v in values)


def absolutize_asset_paths(xml_root: ET.Element, source_dir: Path):
    for tag_name in ("mesh", "texture"):
        for element in xml_root.findall(f".//{tag_name}"):
            file_attr = element.attrib.get("file")
            if not file_attr:
                continue
            element.set("file", str((source_dir / file_attr).resolve()))


def ensure_xml_section(xml_root: ET.Element, section_name: str):
    section = xml_root.find(section_name)
    if section is None:
        section = ET.SubElement(xml_root, section_name)
    return section


def find_ee_attachment_body(xml_root: ET.Element):
    for body_name in PIPER_ATTACHMENT_BODY_CANDIDATES:
        body = xml_root.find(f".//body[@name='{body_name}']")
        if body is not None:
            return body

    bodies = xml_root.findall(".//body")
    if not bodies:
        raise ValueError("No robot bodies were found in the external Piper XML.")
    return bodies[-1]


def ensure_position_actuators(xml_root: ET.Element):
    actuator = ensure_xml_section(xml_root, "actuator")
    existing_joint_actuators = {node.attrib.get("joint") for node in actuator}
    for joint_name in JOINT_NAMES:
        if joint_name in existing_joint_actuators:
            continue
        ET.SubElement(
            actuator,
            "position",
            attrib={
                "name": f"{joint_name}_paint_ctrl",
                "joint": joint_name,
                "kp": "2000",
            },
        )


def patch_external_piper_model(external_model_path: Path):
    """
    Patch the vendor Piper MuJoCo XML into a self-contained painting scene.

    The vendor file provides the robot and actuators. This helper adds:
    - the moving wall,
    - a brush-tip body/geom/site,
    - a touch sensor, and
    - absolute mesh paths so the patched XML can be written anywhere.
    """
    tree = ET.parse(external_model_path)
    root = tree.getroot()
    source_dir = external_model_path.parent

    absolutize_asset_paths(root, source_dir)

    option = ensure_xml_section(root, "option")
    option.set("timestep", str(DT))

    worldbody = ensure_xml_section(root, "worldbody")

    if root.find(".//geom[@name='ground']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            attrib={
                "name": "ground",
                "type": "plane",
                "pos": "0 0 0",
                "size": "2 2 0.1",
                "rgba": "0.96 0.96 0.96 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    wall_body = root.find(f".//body[@name='{WALL_BODY_NAME}']")
    if wall_body is None:
        wall_body = ET.SubElement(
            worldbody,
            "body",
            attrib={
                "name": WALL_BODY_NAME,
                "mocap": "true",
                "pos": format_xyz([WALL_X_CENTER, 0.0, PAINT_Z_CENTER]),
            },
        )
        ET.SubElement(
            wall_body,
            "geom",
            attrib={
                "name": "paint_wall",
                "type": "box",
                "size": format_xyz([WALL_HALF_THICKNESS, WALL_HALF_WIDTH_Y, WALL_HALF_HEIGHT_Z]),
                "rgba": "0.75 0.82 0.92 1",
                "friction": "0.8 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )

    for index in range(PAINT_MARK_COUNT):
        site_name = f"paint_mark_{index:04d}"
        if wall_body.find(f".//site[@name='{site_name}']") is None:
            ET.SubElement(
                wall_body,
                "site",
                attrib={
                    "name": site_name,
                    "pos": format_xyz([-(WALL_HALF_THICKNESS + PAINT_MARK_OFFSET), 0.0, 0.0]),
                    "size": f"{PAINT_MARK_RADIUS:.6f}",
                    "rgba": "0.15 0.34 0.68 0",
                },
            )

    attachment_body = find_ee_attachment_body(root)
    if attachment_body.find(f".//site[@name='{EE_SITE_NAME}']") is None:
        tool_mount = ET.SubElement(
            attachment_body,
            "body",
            attrib={
                "name": "painting_tool_mount",
                "pos": format_xyz(EXTERNAL_BRUSH_OFFSET),
            },
        )
        ET.SubElement(
            tool_mount,
            "geom",
            attrib={
                "name": "brush_tip_geom",
                "type": "sphere",
                "size": f"{BRUSH_RADIUS:.6f}",
                "rgba": "0.95 0.28 0.20 1",
                "friction": "1.0 0.02 0.002",
                "contype": "1",
                "conaffinity": "1",
            },
        )
        ET.SubElement(
            tool_mount,
            "site",
            attrib={
                "name": EE_SITE_NAME,
                "size": "0.010",
                "rgba": "1.0 0.0 0.0 1",
            },
        )
        ET.SubElement(
            tool_mount,
            "site",
            attrib={
                "name": "brush_touch_site",
                "size": "0.025",
                "rgba": "0.2 0.5 1.0 0.20",
            },
        )

    sensor = ensure_xml_section(root, "sensor")
    existing_sensor_names = {node.attrib.get("name") for node in sensor}
    if TOUCH_SENSOR_NAME not in existing_sensor_names:
        ET.SubElement(
            sensor,
            "touch",
            attrib={
                "name": TOUCH_SENSOR_NAME,
                "site": "brush_touch_site",
            },
        )

    ensure_position_actuators(root)

    patched_dir = Path(tempfile.gettempdir())
    patched_path = patched_dir / "patched_piper_painting_scene.xml"
    tree.write(patched_path, encoding="utf-8", xml_declaration=True)
    return patched_path


def resolve_external_model_path():
    if EXTERNAL_MODEL_PATH is not None:
        candidate = Path(EXTERNAL_MODEL_PATH).expanduser()
        if candidate.is_dir():
            for relative_path in PIPER_MODEL_RELATIVE_CANDIDATES:
                nested = candidate / relative_path
                if nested.exists():
                    return nested, f"user-specified Piper repo: {nested}"
        elif candidate.exists():
            return candidate, f"user-specified model: {candidate}"

    if USE_REAL_PIPER_MODEL_IF_AVAILABLE and AUTO_FIND_LOCAL_PIPER_MODEL:
        candidate = find_local_piper_model()
        if candidate is not None:
            return candidate, f"auto-discovered Piper MuJoCo model: {candidate}"

    return None, None


def load_model():
    """Load an external model if provided, otherwise build the self-contained demo."""
    external_model_path, external_source = resolve_external_model_path()
    if external_model_path is not None:
        try:
            patched_model_path = patch_external_piper_model(external_model_path)
            model = mujoco.MjModel.from_xml_path(str(patched_model_path))
            source = f"{external_source} patched with wall and brush"
            return model, source
        except Exception as exc:
            print(f"Could not use the external Piper model at {external_model_path}. Falling back. Reason: {exc}")

    model = mujoco.MjModel.from_xml_string(build_simplified_piper_like_mjcf())
    source = "self-contained simplified 6-DOF Piper-like MJCF"
    return model, source


# %% Kinematics helpers
def get_joint_indices(model, joint_names):
    qpos_indices = []
    dof_indices = []
    lower = []
    upper = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(
                f"Joint '{name}' was not found. Update JOINT_NAMES when swapping in a real robot model."
            )
        qpos_indices.append(model.jnt_qposadr[joint_id])
        dof_indices.append(model.jnt_dofadr[joint_id])
        lower.append(model.jnt_range[joint_id, 0])
        upper.append(model.jnt_range[joint_id, 1])
    return (
        np.array(qpos_indices, dtype=int),
        np.array(dof_indices, dtype=int),
        np.array(lower, dtype=float),
        np.array(upper, dtype=float),
    )


def get_joint_actuator_indices(model, joint_names):
    """Map each commanded joint to its corresponding actuator index."""
    actuator_indices = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' was not found while searching for actuators.")

        matching_actuator_id = None
        for actuator_id in range(model.nu):
            transmission_target = int(model.actuator_trnid[actuator_id, 0])
            if transmission_target == joint_id:
                matching_actuator_id = actuator_id
                break

        if matching_actuator_id is None:
            raise ValueError(
                f"No actuator was found for joint '{joint_name}'. "
                "Check the external model actuators or update JOINT_NAMES."
            )
        actuator_indices.append(matching_actuator_id)

    return np.array(actuator_indices, dtype=int)


def get_ee_pose(model, data, site_id):
    """Return the current end-effector Cartesian position."""
    return data.site_xpos[site_id].copy()


def get_robot_chain_body_ids(model, joint_names):
    """Return body ids for the serial chain associated with the commanded joints."""
    body_ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        body_id = model.jnt_bodyid[joint_id]
        if body_id not in body_ids:
            body_ids.append(body_id)
    return body_ids


def get_robot_chain_points(model, data, body_ids, site_id):
    """Return a list of 3D points that sketch the robot chain for animation."""
    points = []
    for body_id in body_ids:
        points.append(data.xpos[body_id].copy())
    points.append(data.site_xpos[site_id].copy())
    return np.asarray(points, dtype=float)


def compute_jacobian(model, data, site_id):
    """Return translational and rotational Jacobians for the end-effector site."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp, jacr


def get_site_linear_velocity(model, data, site_id):
    """Estimate site linear velocity from the translational Jacobian."""
    jacp, _ = compute_jacobian(model, data, site_id)
    return jacp @ data.qvel


def ik_damped_least_squares(
    model,
    ik_data,
    site_id,
    target_pos,
    q_seed,
    qpos_indices,
    dof_indices,
    joint_lower,
    joint_upper,
    damping=IK_DAMPING,
    max_iters=IK_MAX_ITERS,
    tol=IK_TOL,
):
    """
    Solve position-only inverse kinematics with a damped least-squares update.

    This keeps the code simple and easy to replace if you later use a different
    IK package or a real Piper model with task-space control.
    """
    q_trial = q_seed.copy()

    for _ in range(max_iters):
        ik_data.qpos[qpos_indices] = q_trial
        ik_data.qvel[:] = 0.0
        mujoco.mj_forward(model, ik_data)

        current_pos = ik_data.site_xpos[site_id].copy()
        position_error = target_pos - current_pos
        if np.linalg.norm(position_error) < tol:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        J = jacp[:, dof_indices]

        regularized = J @ J.T + (damping**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(regularized, position_error)
        dq = np.clip(dq, -IK_STEP_LIMIT, IK_STEP_LIMIT)

        q_trial = np.clip(q_trial + dq, joint_lower, joint_upper)

    return q_trial


# %% Motion and control helpers
def smoothstep_fraction(t, t_start, duration):
    """Return a smooth 0->1 phase and its time derivative over the interval."""
    if duration <= 1e-9:
        return (1.0, 0.0) if t >= t_start else (0.0, 0.0)

    if t <= t_start:
        return 0.0, 0.0
    if t >= t_start + duration:
        return 1.0, 0.0

    s = (t - t_start) / duration
    phase = s * s * (3.0 - 2.0 * s)
    phase_dot = (6.0 * s * (1.0 - s)) / duration
    return float(phase), float(phase_dot)


def wall_motion(t):
    """
    Showcase-friendly wall motion along x.

    The wall starts at its nominal x location, moves toward the robot to
    increase contact force, then retreats partway while staying no farther away
    than its original x position. This makes the admittance response easy to see
    while keeping the painting task feasible.
    """
    retreat_distance = min(WALL_RETREAT_DISTANCE, WALL_PUSH_DISTANCE)

    push_phase, push_phase_dot = smoothstep_fraction(t, WALL_MOTION_START, WALL_PUSH_DURATION)
    return_phase, return_phase_dot = smoothstep_fraction(
        t,
        WALL_MOTION_START + WALL_PUSH_DURATION,
        WALL_RETURN_DURATION,
    )

    x = WALL_X_CENTER - WALL_PUSH_DISTANCE * push_phase + retreat_distance * return_phase
    vx = -WALL_PUSH_DISTANCE * push_phase_dot + retreat_distance * return_phase_dot
    return float(x), float(vx)


def effective_sim_duration():
    return max(SIM_DURATION, PAINT_PLAN["total_duration"])


def painting_nominal_pose(t):
    """
    Return the nominal position and pen-down state for painting UCLA.

    When the brush is between strokes, pen_down=False so the controller can back
    away from the wall slightly instead of drawing connecting travel lines.
    """
    if t <= PAINT_START_DELAY:
        return PAINT_PLAN["start_point"].copy(), False

    for segment in PAINT_PLAN["segments"]:
        if segment["t_start"] <= t <= segment["t_end"] or (
            abs(t - segment["t_end"]) < 1e-9 and segment["t_end"] >= segment["t_start"]
        ):
            distance = segment["speed"] * max(0.0, t - segment["t_start"])
            local_point_yz = interpolate_polyline(segment["points"], distance)
            return local_to_world_yz(local_point_yz), segment["pen_down"]

    return PAINT_PLAN["end_point"].copy(), False


def get_paint_mark_site_ids(model):
    site_ids = []
    for index in range(PAINT_MARK_COUNT):
        site_name = f"paint_mark_{index:04d}"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            site_ids.append(site_id)
    return site_ids


def reset_paint_marks(model, site_ids):
    hidden_pos = np.array([-(WALL_HALF_THICKNESS + PAINT_MARK_OFFSET), 0.0, 0.0], dtype=float)
    for site_id in site_ids:
        model.site_pos[site_id] = hidden_pos
        model.site_rgba[site_id] = np.array([PAINT_MARK_RGBA[0], PAINT_MARK_RGBA[1], PAINT_MARK_RGBA[2], 0.0])


def maybe_add_paint_mark(model, site_ids, next_mark_index, ee_pos, last_mark_local_yz):
    if next_mark_index >= len(site_ids):
        return next_mark_index, last_mark_local_yz

    local_yz = world_to_paint_plane_yz(ee_pos[1], ee_pos[2])
    if abs(local_yz[0]) > WALL_HALF_WIDTH_Y or abs(local_yz[1]) > WALL_HALF_HEIGHT_Z:
        return next_mark_index, last_mark_local_yz
    if last_mark_local_yz is not None and np.linalg.norm(local_yz - last_mark_local_yz) < PAINT_MARK_SPACING:
        return next_mark_index, last_mark_local_yz

    site_id = site_ids[next_mark_index]
    site_y = PAINT_Y_CENTER - local_yz[0] if TEXT_READABLE_FROM_ROBOT_SIDE else PAINT_Y_CENTER + local_yz[0]
    model.site_pos[site_id] = np.array([-(WALL_HALF_THICKNESS + PAINT_MARK_OFFSET), site_y, local_yz[1]], dtype=float)
    model.site_rgba[site_id] = PAINT_MARK_RGBA
    return next_mark_index + 1, local_yz


def get_touch_sensor_value(model, data, sensor_name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        return None
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    values = data.sensordata[adr : adr + dim]
    if dim == 1:
        return float(values[0])
    return values.copy()


def contact_force(model, data, site_id, t):
    """
    Estimate the normal contact force.

    Preferred option in many MuJoCo demos is a touch sensor, but in this
    assignment the manual spring-damper estimate is often more stable and easier
    to interpret. The helper keeps both paths available.
    """
    ee_pos = get_ee_pose(model, data, site_id)
    ee_vel = get_site_linear_velocity(model, data, site_id)

    wall_x, wall_vx = wall_motion(t)
    wall_surface_x = wall_x - WALL_HALF_THICKNESS

    penetration = ee_pos[0] + BRUSH_RADIUS - wall_surface_x
    closing_speed = ee_vel[0] - wall_vx

    if penetration > 0.0:
        force_manual = CONTACT_STIFFNESS * penetration + CONTACT_DAMPING * closing_speed
        force_manual = max(0.0, force_manual)
    else:
        force_manual = 0.0

    force_touch = get_touch_sensor_value(model, data, TOUCH_SENSOR_NAME)
    if force_touch is None:
        force_touch = 0.0

    force_used = float(force_touch) if USE_TOUCH_SENSOR else float(force_manual)

    return {
        "force_used": force_used,
        "force_manual": float(force_manual),
        "force_touch": float(force_touch),
        "penetration": float(max(0.0, penetration)),
        "wall_x": float(wall_x),
        "wall_surface_x": float(wall_surface_x),
        "wall_vx": float(wall_vx),
        "ee_x": float(ee_pos[0]),
    }


def admittance_update(delta_x, delta_x_dot, force_measured, dt, in_contact):
    """
    Update the scalar normal-direction admittance state.

    Sign convention:
    - Increasing x means the robot moves toward the wall.
    - If F_meas > F_des, force_error becomes negative, so delta_x decreases and
      the brush moves away from the wall.
    - If F_meas < F_des, force_error becomes positive, so delta_x increases and
      the brush moves toward the wall.
    """
    if not in_contact:
        delta_x_dot = min(APPROACH_SEARCH_SPEED, max(0.0, delta_x_dot) + 0.25 * APPROACH_SEARCH_SPEED)
        delta_x = np.clip(delta_x + delta_x_dot * dt, -MAX_DELTA_X, MAX_DELTA_X)
        return delta_x, delta_x_dot, 0.0

    force_error = F_DES - force_measured
    delta_x_ddot = (force_error - D_A * delta_x_dot - K_A * delta_x) / M_A
    delta_x_dot = np.clip(delta_x_dot + delta_x_ddot * dt, -MAX_DELTA_X_DOT, MAX_DELTA_X_DOT)
    delta_x = np.clip(delta_x + delta_x_dot * dt, -MAX_DELTA_X, MAX_DELTA_X)

    if abs(delta_x) >= MAX_DELTA_X - 1e-9:
        delta_x_dot = 0.0

    return float(delta_x), float(delta_x_dot), float(delta_x_ddot)


# %% Plotting and rendering
def plot_results(log, output_dir):
    time = np.asarray(log["time"])
    wall_surface_x = np.asarray(log["wall_surface_x"])
    ee_x = np.asarray(log["ee_x"])
    x_cmd = np.asarray(log["x_cmd"])
    force_meas = np.asarray(log["force_meas"])
    force_des = np.asarray(log["force_des"])
    delta_x = np.asarray(log["delta_x"])
    ee_y = np.asarray(log["ee_y"])
    ee_z = np.asarray(log["ee_z"])
    pen_down = np.asarray(log["pen_down"], dtype=bool)
    joint_positions = np.asarray(log["joint_positions"])
    joint_commands = np.asarray(log["joint_commands"])

    fig1, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(time, wall_surface_x, label="Wall surface x", linewidth=2)
    axes[0].plot(time, ee_x, label="EE x", linewidth=2)
    axes[0].plot(time, x_cmd, label="Commanded EE x", linestyle="--")
    axes[0].set_ylabel("x position [m]")
    axes[0].set_title("Wall motion vs. end-effector normal motion")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, force_meas, label="Measured contact force", linewidth=2)
    axes[1].plot(time, force_des, label="Desired contact force", linestyle="--")
    axes[1].set_ylabel("Force [N]")
    axes[1].set_title("Force tracking")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time, delta_x, color="tab:purple", linewidth=2)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Admittance $\\Delta x$ [m]")
    axes[2].set_title("Normal admittance displacement")
    axes[2].grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1_path = output_dir / "admittance_timeseries.png"
    fig1.savefig(fig1_path, dpi=180, bbox_inches="tight")

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    painted_coords = np.array([world_to_paint_plane_yz(y, z) for y, z in zip(ee_y, ee_z)])
    painted_y = painted_coords[:, 0]
    painted_z = painted_coords[:, 1]
    painted_y[~pen_down] = np.nan
    painted_z[~pen_down] = np.nan
    axes2[0].plot(painted_y, painted_z, label=f'Painted path: "{PAINT_TEXT}"', linewidth=2.5)
    axes2[0].set_xlabel("wall-horizontal coordinate [m]")
    axes2[0].set_ylabel("wall-vertical coordinate [m]")
    axes2[0].set_title(f'Painting path in the wall plane: "{PAINT_TEXT}"')
    axes2[0].axis("equal")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    for i in range(joint_positions.shape[1]):
        axes2[1].plot(time, joint_positions[:, i], label=f"q{i+1}")
        axes2[1].plot(time, joint_commands[:, i], linestyle="--", alpha=0.55)
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Joint angle [rad]")
    axes2[1].set_title("Joint trajectories (solid = actual, dashed = command)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(ncol=2, fontsize=8)

    fig2.tight_layout()
    fig2_path = output_dir / "painting_path_and_joints.png"
    fig2.savefig(fig2_path, dpi=180, bbox_inches="tight")

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot(np.asarray(log["ee_x"]), ee_y, ee_z, linewidth=2, label="Brush tip")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("y [m]")
    ax3.set_zlabel("z [m]")
    ax3.set_title("3D end-effector trajectory")
    ax3.legend()
    fig3.tight_layout()
    fig3_path = output_dir / "ee_3d_path.png"
    fig3.savefig(fig3_path, dpi=180, bbox_inches="tight")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close("all")
    return fig1_path, fig2_path, fig3_path


def make_scene_option():
    option = mujoco.MjvOption()
    set_vis_flag(option, "mjVIS_SITE", True)
    set_vis_flag(option, "mjVIS_CONTACTPOINT", SHOW_CONTACT_VISUALS)
    set_vis_flag(option, "mjVIS_CONTACTFORCE", SHOW_CONTACT_VISUALS)
    return option


def make_renderer(model):
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    camera.distance = CAMERA_DISTANCE
    camera.lookat[:] = CAMERA_LOOKAT
    return renderer, camera, make_scene_option()


def running_with_mjpython():
    return bool(os.getenv("MJPYTHON_BIN")) or "mjpython" in Path(sys.argv[0]).name.lower()


def should_launch_native_viewer():
    return SHOW_NATIVE_MUJOCO_VIEWER and not IN_COLAB


def configure_native_viewer_camera(viewer_handle):
    with viewer_handle.lock():
        viewer_handle.cam.azimuth = CAMERA_AZIMUTH
        viewer_handle.cam.elevation = CAMERA_ELEVATION
        viewer_handle.cam.distance = CAMERA_DISTANCE
        viewer_handle.cam.lookat[:] = CAMERA_LOOKAT
        set_vis_flag(viewer_handle.opt, "mjVIS_SITE", True)
        set_vis_flag(viewer_handle.opt, "mjVIS_CONTACTPOINT", SHOW_CONTACT_VISUALS)
        set_vis_flag(viewer_handle.opt, "mjVIS_CONTACTFORCE", SHOW_CONTACT_VISUALS)


def maybe_launch_native_viewer(model, data):
    """
    Launch the MuJoCo mesh viewer when possible.

    Official MuJoCo docs note that on macOS `launch_passive` must be executed via
    the special `mjpython` launcher rather than plain `python`.
    """
    if not should_launch_native_viewer():
        return None

    if sys.platform == "darwin" and not running_with_mjpython():
        print(
            "Native MuJoCo viewer not launched because macOS passive viewer requires `mjpython`.\n"
            "Run this script with:\n"
            f"  /opt/miniconda3/bin/mjpython {SCRIPT_DIR / 'MuJoCo_test.py'}"
        )
        return None

    viewer_handle = mujoco_viewer.launch_passive(
        model,
        data,
        show_left_ui=VIEWER_SHOW_LEFT_UI,
        show_right_ui=VIEWER_SHOW_RIGHT_UI,
    )
    configure_native_viewer_camera(viewer_handle)
    return viewer_handle


def should_attempt_rendering():
    """Attempt offscreen MuJoCo rendering whenever video export is enabled."""
    return RENDER_VIDEO


def save_render_outputs(frames, output_dir):
    if not frames:
        return None, None

    mp4_path = None
    gif_path = None

    if SAVE_RENDER_MP4:
        candidate_path = output_dir / "painting_demo.mp4"
        if imageio_ffmpeg is None:
            print("imageio-ffmpeg is not available, so MP4 export is skipped and GIF export will be used.")
        else:
            try:
                writer = imageio.get_writer(candidate_path, fps=RENDER_FPS, codec="libx264", macro_block_size=None)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                mp4_path = candidate_path
                print(f"Saved rendered video to: {mp4_path}")
            except Exception as exc:
                print(f"Could not save MP4 render. Falling back to GIF. Reason: {exc}")

    if SAVE_RENDER_GIF:
        gif_path = output_dir / "painting_demo.gif"
        imageio.mimsave(gif_path, frames, fps=RENDER_FPS)
        print(f"Saved rendered GIF to: {gif_path}")
        if IPyImage is not None and display is not None:
            display(IPyImage(filename=str(gif_path)))

    return mp4_path, gif_path


def wall_polygon(surface_x):
    z_min = PAINT_Z_CENTER - WALL_HALF_HEIGHT_Z
    z_max = PAINT_Z_CENTER + WALL_HALF_HEIGHT_Z
    y_min = -WALL_HALF_WIDTH_Y
    y_max = WALL_HALF_WIDTH_Y
    return np.array(
        [
            [surface_x, y_min, z_min],
            [surface_x, y_max, z_min],
            [surface_x, y_max, z_max],
            [surface_x, y_min, z_max],
        ],
        dtype=float,
    )


def save_robot_3d_animation(log, output_dir):
    """Save a matplotlib 3D animation that shows the robot painting the wall."""
    if not SAVE_ROBOT_3D_ANIMATION or not log["robot_chain_points"]:
        return None, None

    sample_indices = list(range(0, len(log["time"]), ROBOT_ANIMATION_EVERY_N_STEPS))
    if sample_indices[-1] != len(log["time"]) - 1:
        sample_indices.append(len(log["time"]) - 1)

    robot_points = np.asarray(log["robot_chain_points"])
    ee_positions = np.column_stack([log["ee_x"], log["ee_y"], log["ee_z"]])
    time = np.asarray(log["time"])
    force_meas = np.asarray(log["force_meas"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D robot animation while painting")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(0.0, 0.80)
    ax.set_ylim(-0.35, 0.35)
    ax.set_zlim(0.0, 0.70)
    ax.view_init(elev=ROBOT_ANIMATION_ELEVATION, azim=ROBOT_ANIMATION_AZIMUTH)
    ax.grid(True, alpha=0.3)

    robot_line, = ax.plot([], [], [], "-o", color="tab:blue", linewidth=3, markersize=5, label="Robot")
    ee_trail_line, = ax.plot([], [], [], color="tab:red", linewidth=1.8, alpha=0.8, label="Brush trail")
    brush_marker, = ax.plot([], [], [], "o", color="crimson", markersize=7, label="Brush tip")
    wall_artist = {"poly": None}
    status_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    def init():
        robot_line.set_data([], [])
        robot_line.set_3d_properties([])
        ee_trail_line.set_data([], [])
        ee_trail_line.set_3d_properties([])
        brush_marker.set_data([], [])
        brush_marker.set_3d_properties([])
        status_text.set_text("")
        return robot_line, ee_trail_line, brush_marker, status_text

    def update(frame_number):
        idx = sample_indices[frame_number]
        points = robot_points[idx]
        trail = ee_positions[: idx + 1]

        robot_line.set_data(points[:, 0], points[:, 1])
        robot_line.set_3d_properties(points[:, 2])

        ee_trail_line.set_data(trail[:, 0], trail[:, 1])
        ee_trail_line.set_3d_properties(trail[:, 2])

        brush_marker.set_data([points[-1, 0]], [points[-1, 1]])
        brush_marker.set_3d_properties([points[-1, 2]])

        if wall_artist["poly"] is not None:
            wall_artist["poly"].remove()
        wall_vertices = [wall_polygon(log["wall_surface_x"][idx])]
        wall_artist["poly"] = Poly3DCollection(
            wall_vertices,
            facecolors=(0.55, 0.75, 0.95, 0.28),
            edgecolors=(0.25, 0.45, 0.75, 0.9),
            linewidths=1.0,
        )
        ax.add_collection3d(wall_artist["poly"])

        status_text.set_text(
            f"t = {time[idx]:.2f} s\n"
            f"F = {force_meas[idx]:.2f} N\n"
            f"x_cmd = {log['x_cmd'][idx]:.3f} m"
        )
        return robot_line, ee_trail_line, brush_marker, status_text, wall_artist["poly"]

    animation_object = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(sample_indices),
        interval=1000 / ROBOT_ANIMATION_FPS,
        blit=False,
    )

    gif_path = output_dir / "robot_painting_3d.gif"
    animation_object.save(gif_path, writer=animation.PillowWriter(fps=ROBOT_ANIMATION_FPS), dpi=120)
    plt.close(fig)

    print(f"Saved matplotlib 3D animation to: {gif_path}")
    if IPyImage is not None and display is not None:
        display(IPyImage(filename=str(gif_path)))

    mp4_path = None
    if imageio_ffmpeg is not None:
        try:
            mp4_path = output_dir / "robot_painting_3d.mp4"
            writer = imageio.get_writer(mp4_path, fps=ROBOT_ANIMATION_FPS, codec="libx264", macro_block_size=None)
            reader = imageio.get_reader(gif_path)
            for frame in reader:
                writer.append_data(np.asarray(frame))
            reader.close()
            writer.close()
            print(f"Saved matplotlib 3D MP4 to: {mp4_path}")
        except Exception as exc:
            print(f"Could not convert the 3D GIF into MP4. Reason: {exc}")

    return gif_path, mp4_path


def get_output_dir():
    return choose_writable_dir(SCRIPT_DIR / "mujoco_painting_outputs", "mujoco_painting_outputs")


# %% Main simulation
def main():
    print(f"Python interpreter: {sys.executable}")
    print(f"Script directory: {SCRIPT_DIR}")

    model, model_source = load_model()
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
    if site_id < 0:
        raise ValueError(
            f"Site '{EE_SITE_NAME}' was not found. Update EE_SITE_NAME after swapping in a real robot model."
        )

    wall_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WALL_BODY_NAME)
    if wall_body_id < 0:
        raise ValueError(
            f"Wall body '{WALL_BODY_NAME}' was not found. Keep the moving wall name consistent with this script."
        )
    mocap_id = model.body_mocapid[wall_body_id]
    if mocap_id < 0:
        raise ValueError("The wall body is expected to be a mocap body in this demo.")

    paint_mark_site_ids = get_paint_mark_site_ids(model)
    reset_paint_marks(model, paint_mark_site_ids)

    qpos_indices, dof_indices, joint_lower, joint_upper = get_joint_indices(model, JOINT_NAMES)
    actuator_indices = get_joint_actuator_indices(model, JOINT_NAMES)
    robot_chain_body_ids = get_robot_chain_body_ids(model, JOINT_NAMES)

    # Initialize the wall pose.
    initial_wall_x, _ = wall_motion(0.0)
    data.mocap_pos[mocap_id] = np.array([initial_wall_x, 0.0, PAINT_Z_CENTER])
    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
    ik_data.mocap_pos[mocap_id] = data.mocap_pos[mocap_id].copy()
    ik_data.mocap_quat[mocap_id] = data.mocap_quat[mocap_id].copy()

    # Use IK to find a reasonable starting joint configuration at the initial painting pose.
    q_seed = np.zeros(len(JOINT_NAMES))
    initial_target, _ = painting_nominal_pose(0.0)
    initial_target[0] = X_NOMINAL_BASE - PEN_UP_CLEARANCE
    q_init = ik_damped_least_squares(
        model,
        ik_data,
        site_id,
        initial_target,
        q_seed,
        qpos_indices,
        dof_indices,
        joint_lower,
        joint_upper,
    )

    data.qpos[qpos_indices] = q_init
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[actuator_indices] = q_init
    mujoco.mj_forward(model, data)

    ik_data.qpos[qpos_indices] = q_init
    ik_data.qvel[:] = 0.0
    ik_data.ctrl[:] = 0.0
    ik_data.ctrl[actuator_indices] = q_init
    mujoco.mj_forward(model, ik_data)

    renderer = None
    camera = None
    scene_option = None
    frames = []
    viewer_handle = maybe_launch_native_viewer(model, data)
    if should_attempt_rendering():
        try:
            renderer, camera, scene_option = make_renderer(model)
        except Exception as exc:
            print(f"Renderer could not be created. Continuing without video. Reason: {exc}")

    log = {
        "time": [],
        "wall_x": [],
        "wall_surface_x": [],
        "ee_x": [],
        "ee_y": [],
        "ee_z": [],
        "x_cmd": [],
        "force_meas": [],
        "force_manual": [],
        "force_touch": [],
        "force_des": [],
        "delta_x": [],
        "delta_x_dot": [],
        "joint_positions": [],
        "joint_commands": [],
        "contact_lost": [],
        "robot_chain_points": [],
        "pen_down": [],
    }

    delta_x = 0.0
    delta_x_dot = 0.0
    q_command = q_init.copy()
    num_steps = int(effective_sim_duration() / DT)
    next_paint_mark_index = 0
    last_paint_mark_local_yz = None
    previous_pen_down = False

    try:
        for step in range(num_steps):
            if viewer_handle is not None and not viewer_handle.is_running():
                print("Viewer window was closed early; ending simulation loop.")
                break

            step_wallclock_start = time.time()
            t = step * DT

            wall_x, _ = wall_motion(t)
            data.mocap_pos[mocap_id] = np.array([wall_x, 0.0, PAINT_Z_CENTER])
            data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
            mujoco.mj_forward(model, data)

            contact = contact_force(model, data, site_id, t)
            p_nominal, pen_down = painting_nominal_pose(t)

            if pen_down and not previous_pen_down:
                delta_x = 0.0
                delta_x_dot = 0.0

            if pen_down:
                force_measured = contact["force_used"]
                in_contact = force_measured > CONTACT_LOSS_FORCE_THRESHOLD
                delta_x, delta_x_dot, _ = admittance_update(delta_x, delta_x_dot, force_measured, DT, in_contact)
                force_desired = F_DES
                x_cmd = float(np.clip(p_nominal[0] + delta_x, X_CMD_MIN, X_CMD_MAX))
            else:
                force_measured = contact["force_used"]
                in_contact = False
                delta_x = -PEN_UP_CLEARANCE
                delta_x_dot = 0.0
                force_desired = 0.0
                x_cmd = float(np.clip(X_NOMINAL_BASE - PEN_UP_CLEARANCE, X_CMD_MIN, X_CMD_MAX))

            p_command = p_nominal.copy()
            p_command[0] = x_cmd

            q_current = data.qpos[qpos_indices].copy()
            q_command = ik_damped_least_squares(
                model,
                ik_data,
                site_id,
                p_command,
                q_current,
                qpos_indices,
                dof_indices,
                joint_lower,
                joint_upper,
            )
            data.ctrl[actuator_indices] = q_command

            if USE_QUASI_STATIC_SERVO:
                data.qpos[qpos_indices] = q_command
                data.qvel[dof_indices] = 0.0
                mujoco.mj_forward(model, data)

            ee_pos = get_ee_pose(model, data, site_id)
            robot_chain_points = get_robot_chain_points(model, data, robot_chain_body_ids, site_id)

            if pen_down and in_contact:
                next_paint_mark_index, last_paint_mark_local_yz = maybe_add_paint_mark(
                    model,
                    paint_mark_site_ids,
                    next_paint_mark_index,
                    ee_pos,
                    last_paint_mark_local_yz,
                )

            log["time"].append(t)
            log["wall_x"].append(contact["wall_x"])
            log["wall_surface_x"].append(contact["wall_surface_x"])
            log["ee_x"].append(ee_pos[0])
            log["ee_y"].append(ee_pos[1])
            log["ee_z"].append(ee_pos[2])
            log["x_cmd"].append(x_cmd)
            log["force_meas"].append(force_measured)
            log["force_manual"].append(contact["force_manual"])
            log["force_touch"].append(contact["force_touch"])
            log["force_des"].append(force_desired)
            log["delta_x"].append(delta_x)
            log["delta_x_dot"].append(delta_x_dot)
            log["joint_positions"].append(q_current.copy())
            log["joint_commands"].append(q_command.copy())
            log["contact_lost"].append(0 if in_contact else 1)
            log["robot_chain_points"].append(robot_chain_points)
            log["pen_down"].append(1 if pen_down else 0)
            previous_pen_down = pen_down

            if renderer is not None and step % RENDER_EVERY_N_STEPS == 0:
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                frames.append(np.asarray(pixels).copy())

            if viewer_handle is not None:
                viewer_handle.sync()

            mujoco.mj_step(model, data)

            if viewer_handle is not None and VIEWER_REALTIME:
                elapsed = time.time() - step_wallclock_start
                remaining = DT - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        if viewer_handle is not None and KEEP_VIEWER_OPEN_AFTER_SIM and viewer_handle.is_running():
            print("Simulation finished. Close the MuJoCo viewer window when you are done inspecting it.")
            while viewer_handle.is_running():
                viewer_handle.sync()
                time.sleep(0.02)
    finally:
        if renderer is not None:
            renderer.close()
        if viewer_handle is not None:
            viewer_handle.close()

    output_dir = get_output_dir()

    fig_paths = plot_results(log, output_dir)
    mp4_path, gif_path = save_render_outputs(frames, output_dir) if frames else (None, None)
    robot_animation_gif_path, robot_animation_mp4_path = save_robot_3d_animation(log, output_dir)

    sim_time = np.asarray(log["time"])
    wall_x_log = np.asarray(log["wall_x"])
    force_meas = np.asarray(log["force_meas"])
    delta_x_log = np.asarray(log["delta_x"])
    contact_loss_fraction = float(np.mean(np.asarray(log["contact_lost"])))

    print("\n=== Simulation summary ===")
    print(f"Model source: {model_source}")
    print(f"Simulated time: {sim_time[-1]:.2f} s")
    print(f"Wall x range: [{np.min(wall_x_log):.3f}, {np.max(wall_x_log):.3f}] m")
    print(f"Mean measured force: {np.mean(force_meas):.2f} N")
    print(f"Max measured force: {np.max(force_meas):.2f} N")
    print(f"Mean |force error|: {np.mean(np.abs(force_meas - F_DES)):.2f} N")
    print(f"Max |admittance displacement|: {np.max(np.abs(delta_x_log)):.4f} m")
    print(f"Contact-lost fraction: {100.0 * contact_loss_fraction:.1f}%")
    print(f"Plots saved to: {output_dir}")
    print(f'Painted text: "{PAINT_TEXT}"')
    if mp4_path is not None:
        print(f"Rendered MP4 saved to: {mp4_path}")
    if gif_path is not None:
        print(f"Rendered GIF saved to: {gif_path}")
    if robot_animation_gif_path is not None:
        print(f"3D robot animation GIF saved to: {robot_animation_gif_path}")
    if robot_animation_mp4_path is not None:
        print(f"3D robot animation MP4 saved to: {robot_animation_mp4_path}")
    print("\nInterpretation:")
    print("- This demo uses a one-sided wall motion: the wall first advances toward the robot, then relaxes back while staying no farther than its starting x position.")
    print("- When the wall moves toward the robot, the brush-wall penetration increases, so the measured contact force rises.")
    print("- The admittance controller responds by decreasing the commanded x position, which moves the end-effector away from the wall.")
    print("- When the wall eases back, the force drops, so the controller increases the commanded x position to keep the brush in contact and finish the stroke.")
    print("- If contact is lost, force alone cannot reveal the exact wall distance, so this demo uses a safe search motion toward the wall and a known wall pose in the contact estimate.")

    print("\nSaved figures:")
    for path in fig_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
