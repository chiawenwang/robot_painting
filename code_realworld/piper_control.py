# piper_arm_controller.py
# -*- coding: utf-8 -*-

import time
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from piper_sdk import C_PiperInterface_V2


# -----------------------------
# Data structures (optional)
# -----------------------------
@dataclass
class EndPose:
    # 单位：mm, deg（欧拉角）
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0


@dataclass
class GripperState:
    # 单位：m, N·m（内部反馈是 0.001mm / 0.001N·m，这里做了换算）
    opening_m: float = 0.0
    effort_nm: float = 0.0
    foc_status: int = 0
    raw: object = None


@dataclass
class ArmStatus:
    ctrl_mode: int = 0
    arm_status: int = 0
    mode_feed: int = 0
    teach_status: int = 0
    motion_status: int = 0
    trajectory_num: int = 0
    err_code: int = 0
    raw: object = None


class PiperArmController:
    """
    Piper 机械臂高层控制封装（基于 C_PiperInterface_V2 + demos）

    约定输入单位：
      - joints: rad
      - end pose: mm + deg (Euler)
      - gripper: meter (m)

    内部会按 demo 的方式转换到 SDK 原始单位：
      - joint: 0.001 degree
      - end pose: 0.001 mm / 0.001 deg
      - gripper: 0.001 mm（即 m -> *1e6）
    """

    # 与 demo 一致：rad -> 0.001deg
    _RAD_TO_MDEG = 1000.0 * 180.0 / math.pi  # ~= 57295.7795
    # 与 demo 一致：mm/deg -> 0.001(mm/deg)
    _MM_DEG_TO_1E3 = 1000.0
    # gripper: meter -> 0.001mm   (1 m = 1e6 * 0.001mm)
    _M_TO_UMM = 1_000_000.0

    # mode_feed 枚举（来自 status 注释）
    MOVE_P = 0x00
    MOVE_J = 0x01
    MOVE_L = 0x02
    MOVE_C = 0x03

    def __init__(
        self,
        can_name: str = "can0",
        home_joints_rad: Optional[List[float]] = None,
        gripper_open_m: float = 0.05,   # demo 里 0.05m=50mm 的写法很常见
        gripper_close_m: float = 0.0,
    ):
        self.can_name = can_name
        self.piper = C_PiperInterface_V2(can_name)
        # self.piper = C_PiperInterface(can_name)

        # ---------- self state ----------
        self.joint_rad: List[float] = [0.0] * 6
        self.end_pose: EndPose = EndPose()
        self.status: ArmStatus = ArmStatus()
        self.gripper: GripperState = GripperState()

        # ---------- configuration ----------
        self.home_joints_rad = home_joints_rad if home_joints_rad is not None else [0.0] * 6
        self.gripper_open_m = float(gripper_open_m)
        self.gripper_close_m = float(gripper_close_m)

    # -----------------------------
    # Connection / enable
    # -----------------------------
    def connect(self, start_thread: bool = True, piper_init: bool = True) -> None:
        """连接 CAN，并启动 SDK 读线程（等价 demo 的 ConnectPort）。"""
        self.piper.ConnectPort(start_thread=start_thread, piper_init=piper_init)
        print("Piper connected on", self.can_name)

    def disconnect(self) -> None:
        self.piper.DisconnectPort()

    def enable(self, retry_interval_s: float = 0.01) -> None:
        """循环使能，直到成功（等价 demo）。"""
        while not self.piper.EnablePiper():
            time.sleep(retry_interval_s)
        print("Piper enabled")

    # -----------------------------
    # Feedback update
    # -----------------------------
    def update(self) -> None:
        """拉取一次反馈，更新 self.xxx"""
        # joints (feedback in 0.001 deg)
        joint_msg = self.piper.GetArmJointMsgs()
        js = joint_msg.joint_state
        j_mdeg = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        self.joint_rad = [math.radians(v / 1000.0) for v in j_mdeg]  # 0.001deg -> deg -> rad

        # end pose (feedback通常也是 0.001 mm / 0.001 deg；这里按常见约定换算)
        end_msg = self.piper.GetArmEndPoseMsgs()
        ep = end_msg.end_pose
        self.end_pose = EndPose(
            x_mm=ep.X_axis / 1000.0,
            y_mm=ep.Y_axis / 1000.0,
            z_mm=ep.Z_axis / 1000.0,
            rx_deg=ep.RX_axis / 1000.0,
            ry_deg=ep.RY_axis / 1000.0,
            rz_deg=ep.RZ_axis / 1000.0,
        )

        # status
        st_msg = self.piper.GetArmStatus()
        st = st_msg.arm_status
        self.status = ArmStatus(
            ctrl_mode=st.ctrl_mode,
            arm_status=st.arm_status,
            mode_feed=st.mode_feed,
            teach_status=st.teach_status,
            motion_status=st.motion_status,
            trajectory_num=st.trajectory_num,
            err_code=st.err_code,
            raw=st,
        )

        # gripper feedback (0.001mm / 0.001N·m)  :contentReference[oaicite:5]{index=5}
        g_msg = self.piper.GetArmGripperMsgs()
        g = g_msg.gripper_state
        self.gripper = GripperState(
            opening_m=(g.grippers_angle / 1000.0) / 1000.0,  # 0.001mm -> mm -> m
            effort_nm=g.grippers_effort / 1000.0,            # 0.001 N·m -> N·m
            foc_status=g.foc_status,
            raw=g,
        )

    # -----------------------------
    # Motion helpers
    # -----------------------------
    def set_mode(self, mode_feed: int, speed: int = 100) -> None:
        """
        切换模式（等价 demo：MotionCtrl_2(0x01, mode, speed, 0x00)）
        mode_feed:
          0x00 MOVE P, 0x01 MOVE J, 0x02 MOVE L, 0x03 MOVE C  :contentReference[oaicite:6]{index=6}
        """
        self.piper.MotionCtrl_2(0x01, mode_feed, speed, 0x00)

    def wait_reached(self, timeout_s: float = 10.0, poll_s: float = 0.02) -> bool:
        """
        等待到位：motion_status 0x00=到达，0x01=未到达  :contentReference[oaicite:7]{index=7}
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st = self.piper.GetArmStatus().arm_status
            if st.motion_status == 0x00:
                return True
            time.sleep(poll_s)
        return False

    # -----------------------------
    # High-level motions
    # -----------------------------
    def goto_zero(self, speed: int = 30, gripper_m: Optional[float] = None) -> None:
        """去零位：完全按 demo 的做法（MOVE J + JointCtrl 全 0）。:contentReference[oaicite:8]{index=8}"""
        print("Going to zero position...")
        self.set_mode(self.MOVE_J, speed=speed)
        self.piper.JointCtrl(0, 0, 0, 0, 0, 0)
        if gripper_m is not None:
            self.gripper_set(gripper_m)

    def goto_home(self, speed: int = 30, gripper_m: Optional[float] = None) -> None:
        """去 home（默认 home=全 0；你可以在 init 里传入 home_joints_rad）。"""
        self.move_joints(self.home_joints_rad, speed=speed, wait=True)
        if gripper_m is not None:
            self.gripper_set(gripper_m)

    def move_joints(self, joints_rad: List[float], speed: int = 100, wait: bool = False) -> None:
        """
        关节空间 MOVE J：demo 的做法是 MotionCtrl_2(MOVE_J) + JointCtrl(...) :contentReference[oaicite:9]{index=9}
        joints_rad: 长度=6
        """
        if len(joints_rad) != 6:
            raise ValueError("joints_rad 需要长度为 6")

        self.set_mode(self.MOVE_J, speed=speed)
        j = [int(round(v * self._RAD_TO_MDEG)) for v in joints_rad]
        self.piper.JointCtrl(j[0], j[1], j[2], j[3], j[4], j[5])

        if wait:
            self.wait_reached()

    def move_end_pose(self, pose: EndPose, speed: int = 100, wait: bool = False) -> None:
        """
        笛卡尔 MOVE P：demo 的做法是 MotionCtrl_2(MOVE_P) + EndPoseCtrl(X,Y,Z,RX,RY,RZ) :contentReference[oaicite:10]{index=10}
        pose 单位：mm + deg（欧拉角）
        """
        self.set_mode(self.MOVE_P, speed=speed)
        X = int(round(pose.x_mm * self._MM_DEG_TO_1E3))
        Y = int(round(pose.y_mm * self._MM_DEG_TO_1E3))
        Z = int(round(pose.z_mm * self._MM_DEG_TO_1E3))
        RX = int(round(pose.rx_deg * self._MM_DEG_TO_1E3))
        RY = int(round(pose.ry_deg * self._MM_DEG_TO_1E3))
        RZ = int(round(pose.rz_deg * self._MM_DEG_TO_1E3))
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        time.sleep(0.01)

        if wait:
            self.wait_reached()

    def move_line(self, pose: EndPose, speed: int = 100, wait: bool = False) -> None:
        """直线 MOVE L：demo 用 MOVE_L + EndPoseCtrl。:contentReference[oaicite:11]{index=11}"""
        self.set_mode(self.MOVE_L, speed=speed)
        self.move_end_pose(pose, speed=speed, wait=wait)

    def move_c(self, pose: EndPose, c_axis_update: int, speed: int = 30) -> None:
        """
        圆弧 MOVE C（demo：MOVE_C + EndPoseCtrl + MoveCAxisUpdateCtrl）
        c_axis_update: 0x01/0x02/0x03 对应 demo 的 MoveCAxisUpdateCtrl
        :contentReference[oaicite:12]{index=12}
        """
        self.set_mode(self.MOVE_C, speed=speed)
        X = int(round(pose.x_mm * self._MM_DEG_TO_1E3))
        Y = int(round(pose.y_mm * self._MM_DEG_TO_1E3))
        Z = int(round(pose.z_mm * self._MM_DEG_TO_1E3))
        RX = int(round(pose.rx_deg * self._MM_DEG_TO_1E3))
        RY = int(round(pose.ry_deg * self._MM_DEG_TO_1E3))
        RZ = int(round(pose.rz_deg * self._MM_DEG_TO_1E3))
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        self.piper.MoveCAxisUpdateCtrl(c_axis_update)

    # -----------------------------
    # Gripper control
    # -----------------------------
    def gripper_enable(self) -> None:
        """
        demo 里常见：先 0x02 再 0x01（像是初始化/使能序列） :contentReference[oaicite:13]{index=13}
        """
        self.piper.GripperCtrl(0, 1000, 0x02, 0)
        self.piper.GripperCtrl(0, 1000, 0x01, 0)

    def gripper_set(self, opening_m: float, effort: int = 1000, enable: int = 0x01, block: int = 0) -> None:
        """
        opening_m: 夹爪开口（m）
        发送给 SDK 的 range 单位按 demo：m -> *1000*1000 （等价 0.001mm） :contentReference[oaicite:14]{index=14}
        """
        rng = int(round(abs(opening_m) * self._M_TO_UMM))
        self.piper.GripperCtrl(rng, effort, enable, block)

    def gripper_open(self, effort: int = 1000) -> None:
        self.gripper_set(self.gripper_open_m, effort=effort)

    def gripper_close(self, effort: int = 1000) -> None:
        self.gripper_set(self.gripper_close_m, effort=effort)

    # -----------------------------
    # Drag teach: find home pose
    # -----------------------------
    def teach_and_set_home(self) -> List[float]:
        """进入拖动示教模式，让用户手动摆好姿态后读取关节角并更新 home_joints_rad。

        返回新的 home_joints_rad（rad）。
        用法：在 connect/enable 之后、goto_home 之前调用。
        """
        self.piper.MotionCtrl_1(0x00, 0x00, 0x01)   # 开启拖动示教
        print("进入拖动模式，现在可以手动移动机械臂到目标 home 位置...")
        input("摆好后按 Enter 读取当前关节角度...")

        self.piper.MotionCtrl_1(0x00, 0x00, 0x02)   # 退出拖动示教
        time.sleep(0.2)

        js = self.piper.GetArmJointMsgs().joint_state
        j_mdeg = [js.joint_1, js.joint_2, js.joint_3,
                  js.joint_4, js.joint_5, js.joint_6]
        new_home = [math.radians(v / 1000.0) for v in j_mdeg]

        self.home_joints_rad = new_home
        print("新 home_joints_rad =", [round(v, 4) for v in new_home])
        print("请把上面的值填入 PiperArmController 初始化的 home_joints_rad 参数。")
        return new_home

    # -----------------------------
    # Safety: emergency stop
    # -----------------------------
    def emergency_stop(self) -> None:
        """快速急停（MotionCtrl: emergency_stop=0x01）。:contentReference[oaicite:15]{index=15}"""
        self.piper.MotionCtrl(emergency_stop=0x01, track_ctrl=0x00, grag_teach_ctrl=0x00)

    def emergency_resume(self) -> None:
        """急停恢复（MotionCtrl: emergency_stop=0x02）。:contentReference[oaicite:16]{index=16}"""
        self.piper.MotionCtrl(emergency_stop=0x02, track_ctrl=0x00, grag_teach_ctrl=0x00)

    # get current control mode
    def get_ctrl_mode(self) -> int:
        """
        返回当前控制模式 ctrl_mode:
        0x00 待机
        0x01 CAN指令控制模式
        0x02 示教模式
        :contentReference[oaicite:2]{index=2}
        """
        return self.piper.GetArmStatus().arm_status.ctrl_mode

    def get_ctrl_mode_str(self) -> str:
        m = self.get_ctrl_mode()
        return {0x00: "STANDBY(0x00)", 0x01: "CAN_CTRL(0x01)", 0x02: "TEACHING(0x02)"}.get(m, f"UNKNOWN({hex(m)})")
    
    def switch_to_can_control(self) -> None:
        self.piper.MotionCtrl_1(0x00, 0x00, 0x02)  # exit teach
        time.sleep(0.05)
        self.piper.MotionCtrl_1(0x00, 0x06, 0x00)  # terminate execution
        time.sleep(0.05)

        while not self.piper.EnablePiper():
            time.sleep(0.01)

        joint_msg = self.piper.GetArmJointMsgs()
        js = joint_msg.joint_state
        hold_j = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]  # 单位：0.001deg


        dt = 1.0 / max(50.0, 1.0)
        t0 = time.time()
        while time.time() - t0 < 2.0:
            self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
            # self.piper.JointCtrl(hold_j[0], hold_j[1], hold_j[2], hold_j[3], hold_j[4], hold_j[5])
            time.sleep(dt)
            # print('Switching to CAN control mode...')

        if self.get_ctrl_mode() != 0x01:
            print("Failed to switch to CAN control mode.")
        else:
            print("Switched to CAN control mode.")

    def set_ctrl_mode2can(self):
        """
        切换机械臂到can控制模式
        """
        print(f"当前机械臂状态为{self.piper.GetArmStatus().arm_status.ctrl_mode}")

        if self.piper.GetArmStatus().arm_status.ctrl_mode == 0x00:#如果是待机
            print("尝试从 待机模式->can控制模式")
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            time.sleep(1)
            while not self.piper.EnablePiper():  # 使能机械臂
                time.sleep(0.01)
            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)  # 设置can模式
            t0 = time.time()
            while self.piper.GetArmStatus().arm_status.ctrl_mode != 0x01:
                if time.time() - t0 > 10.0:
                    print("Failed to switch to CAN control mode (timeout).")
                    break
                time.sleep(0.01)
            else:
                print("成功切换到can控制模式")
        elif self.piper.GetArmStatus().arm_status.ctrl_mode == 0x02:#如果是示教模式
            print("尝试从 示教模式->can控制模式")
            self.piper.MotionCtrl_1(0x02,0,0)#恢复，示教模式->待机模式
            # print(piper.GetArmStatus().arm_status.ctrl_mode)
            time.sleep(1)#这里必须要等切换到待机模式
            # print(piper.GetArmStatus().arm_status.ctrl_mode)

            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            time.sleep(1)#这里也必须要等

            while( not self.piper.EnablePiper()):#使能机械臂
                time.sleep(0.01)

            self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)#设置can模式
            while(self.piper.GetArmStatus().arm_status.ctrl_mode != 0x01):#等待进入can控制模式
                time.sleep(0.01)
            print("成功切换到can控制模式")


        elif self.piper.GetArmStatus().arm_status.ctrl_mode == 0x01:#如果是can控制模式
            print("can控制模式") 
            while( not self.piper.EnablePiper()):#使能机械臂
                time.sleep(0.01)
            
        else:
            print(self.piper.GetArmStatus().arm_status.ctrl_mode)
        #清除夹爪错误使其可以被正常控制
        self.piper.GripperCtrl(0,1000,0x02, 0)
        self.piper.GripperCtrl(0,1000,0x01, 0)

if __name__ == "__main__":
    arm = PiperArmController(
        can_name="can0",
        home_joints_rad=[1.57, 0.866, -0.960, 0.0, 0.182, 1.571],
        gripper_open_m=0.03,
        gripper_close_m=0.00,
    )

    arm.connect()
    arm.enable()
    # arm.gripper_enable()
    time.sleep(0.1)  # 等待使能完成

    arm.gripper_enable()

    print("Current control mode:", arm.get_ctrl_mode_str())
    # print('status:', arm.status)

    arm.set_ctrl_mode2can()

    # exit()

    # arm.update()
    # cur_joint = arm.joint_rad
    # print("Current joints (rad):", cur_joint)
    # cur_pose = arm.end_pose
    # print("Current pose:", cur_pose)

    # time.sleep(0.5)

    arm.gripper_open()
    time.sleep(0.5)

    mode = input("输入 't' 进入拖动示教调整home位，其他键直接去home位: ").strip().lower()
    if mode == 't':
        arm.teach_and_set_home()

    arm.set_ctrl_mode2can()
    arm.goto_home(speed=30)
    arm.update()
    print("Home joints (rad):", arm.joint_rad)
    print("Home end pose:", arm.end_pose)
    input("机械臂已到 home 位。按 Enter 退出...")
