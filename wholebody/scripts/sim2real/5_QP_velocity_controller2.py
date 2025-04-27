import torch
import numpy as np
import qpsolvers as qp
import spatialmath as sm
import time
import matplotlib.pyplot as plt
import roboticstoolbox as rtb

import torchcontrol as toco
from polymetis.utils.data_dir import get_full_path_to_urdf

def step_robot(q, Tep, frankie, robot_model, ax):

    pos, quat = robot_model.forward_kinematics(torch.Tensor(q))
    quat_np = quat.numpy()
    quat_wxyz = np.r_[quat_np[3], quat_np[:3]]
    rot = sm.UnitQuaternion(quat_wxyz).SO3()
    wTe = sm.SE3.Rt(rot, pos.numpy()).A

    eTep = np.linalg.inv(wTe) @ Tep
    # eTep = Tep @ np.linalg.inv(wTe)
    # twist_ee = sm.SE3(eTep).log()

    # T_ee_in_world = sm.SE3(wTe)
    # Ad_wTe = T_ee_in_world.Ad()
    # v = Ad_wTe @ twist_ee

    et = np.sum(np.abs(eTep[:3, -1]))

    Y = 0.01
    n = 7  # Franka is 7-DOF

    Q = np.eye(n + 6)
    Q[:n, :n] *= Y
    Q[:2, :2] *= 1.0 / et
    
    Q[n:, n:] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)

    v[3:] *= 0.05

    # if ax is not None:
    #     ax.cla()  # 이전 프레임 지우기

    #     ee_pos = wTe[:3, -1]
    #     target_pos = Tep[:3, -1]

    #     ax.scatter(*ee_pos, c='blue', label='Current EE Pose')
    #     ax.scatter(*target_pos, c='red', label='Target Pose')
    #     ax.plot([ee_pos[0], target_pos[0]], [ee_pos[1], target_pos[1]], [ee_pos[2], target_pos[2]], 'k--')

    #     R_wTe = wTe[:3, :3]
    #     v_ee = v[:3]
    #     v_dir = R_wTe @ v_ee
    #     norm = np.linalg.norm(v_dir)

    #     if norm > 1e-6:
    #         v_scaled = (v_dir / norm) * 0.1
    #         ax.quiver(ee_pos[0], ee_pos[1], ee_pos[2],
    #                   v_scaled[0], v_scaled[1], v_scaled[2],
    #                   color='green', length=1.0, normalize=False, linewidth=2, label='Translation Velocity')

    #     ax.set_xlim([-0.5, 0.5])
    #     ax.set_ylim([-0.5, 0.5])
    #     ax.set_zlim([0.0, 1.0])
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("EE Pose vs Target Pose")
    #     ax.legend()
    #     ax.set_box_aspect([1,1,1])

    #     plt.draw()
    #     plt.pause(0.001)


    # print(f"v trans L2 norm: {np.linalg.norm(v[:3])}")

    jac = robot_model.compute_jacobian(torch.Tensor(q), "panda_link8")  # torch.Tensor
    # print(f"jac: {jac}")
    Aeq = np.c_[frankie.jacobe(q), np.eye(6)]  # numpy 변환 후 QP용으로 사용
    beq = v.reshape((6,))

    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    ps = 0.1
    pi = 0.9
    # joint_velocity_damper()를 직접 구현하거나 skip

    c = np.zeros(n + 6)

    lb = -np.r_[1*np.ones(n), 10*np.ones(6)]  # Franka velocity limit 대략 +-2.0 rad/s
    ub = np.r_[1*np.ones(n), 10*np.ones(6)]

    qd_solution = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='osqp')
    if qd_solution is None:
        raise RuntimeError("QP solver failed.")

    qd_command = qd_solution[:n]

    # if et > 0.5:
    #     qd_command *= 0.7 / et
    # else:
    #     qd_command *= 1.4


    arrived = et < 0.02
    return arrived, torch.Tensor(qd_command)


# -- 메인 루프 --

from polymetis import RobotInterface

if __name__ == "__main__":
    # Initialize robot interface
    robot = RobotInterface(
        ip_address="localhost",
    )

    frankie = rtb.models.Panda()
    frankie.q = frankie.qr

    robot.go_home()

    time.sleep(1)
    print("start")

    q_current = robot.get_joint_positions().numpy()

    pos, quat = robot.robot_model.forward_kinematics(torch.Tensor(q_current))

    # Quaternion 변환 (x, y, z, w) → (w, x, y, z)
    quat_np = quat.numpy()
    quat_wxyz = np.r_[quat_np[3], quat_np[:3]]

    # UnitQuaternion 생성 및 SO3 변환
    uq = sm.UnitQuaternion(quat_wxyz)
    rot = uq.SO3()  # 이게 핵심

    # SE3 생성
    wTep = sm.SE3.Rt(rot, pos.numpy()).A

    # 예: 40cm 뒤로 이동
    wTep = wTep @ sm.SE3.Trans(0.2, 0.0, 0.0).A

    arrived = False

    # Velocity control 시작 (초기화, 이건 빈 벡터로 먼저 시작 가능)
    robot.start_joint_velocity_control(joint_vel_desired=torch.zeros(7))

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while not arrived:
        q_current = robot.get_joint_positions().numpy()
        # print(q_current)
        pos, quat = robot.robot_model.forward_kinematics(torch.Tensor(q_current))

        # print(pos)
        arrived, qd_command = step_robot(q_current, wTep, frankie, robot.robot_model, ax=ax)
        # print(f"qd_command_dim: {qd_command.shape}")

        robot.update_desired_joint_velocities(qd_command)
        time.sleep(0.02)  # 50Hz 제어
     

    # 도착하면 정지
    print("도착 완료.")
    robot.update_desired_joint_velocities(torch.zeros(7))

