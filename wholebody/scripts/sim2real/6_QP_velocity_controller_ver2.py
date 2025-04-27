import torch
import numpy as np
import qpsolvers as qp
import spatialmath as sm
import time
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from polymetis import RobotInterface

def step_robot(q, Tep, frankie, robot_model, ax=None):
    pos, quat = robot_model.forward_kinematics(torch.Tensor(q))
    quat_np = quat.numpy()
    quat_wxyz = np.r_[quat_np[3], quat_np[:3]]
    rot = sm.UnitQuaternion(quat_wxyz).SO3()
    wTe = sm.SE3.Rt(rot, pos.numpy()).A

    eTep = np.linalg.inv(wTe) @ Tep
    et = np.sum(np.abs(eTep[:3, -1]))

    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    v[3:] *= 0.05  # 회전 속도 축소

    # 시각화
    if ax is not None:
        ax.cla()
        ee_pos = wTe[:3, -1]
        target_pos = Tep[:3, -1]
        ax.scatter(*ee_pos, c='blue', label='Current EE Pose')
        ax.scatter(*target_pos, c='red', label='Target Pose')
        ax.plot([ee_pos[0], target_pos[0]], [ee_pos[1], target_pos[1]], [ee_pos[2], target_pos[2]], 'k--')
        R_wTe = wTe[:3, :3]
        v_dir = R_wTe @ v[:3]
        norm = np.linalg.norm(v_dir)
        if norm > 1e-6:
            v_scaled = (v_dir / norm) * 0.1
            ax.quiver(*ee_pos, *v_scaled, color='green', length=1.0, normalize=False, linewidth=2, label='EE Velocity')
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0.0, 1.0])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("EE Pose vs Target Pose")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        plt.draw(); plt.pause(0.001)

    # QP 구성
    n = 7
    Q = np.eye(n + 6)
    Q[:n, :n] *= 0.01
    Q[:2, :2] *= 1.0 / et
    Q[n:, n:] = (1.0 / et) * np.eye(6)
    Aeq = np.c_[frankie.jacobe(q), np.eye(6)]
    beq = v.reshape((6,))
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)
    c = np.zeros(n + 6)
    lb = -np.r_[np.ones(n), 10 * np.ones(6)]
    ub = np.r_[np.ones(n), 10 * np.ones(6)]

    qd_solution = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='osqp')
    if qd_solution is None:
        raise RuntimeError("QP solver failed.")
    qd_command = qd_solution[:n]

    arrived = et < 0.02
    return arrived, torch.Tensor(qd_command)

# === Main 실행부 ===
if __name__ == "__main__":
    robot = RobotInterface(ip_address="localhost")
    frankie = rtb.models.Panda()
    frankie.q = frankie.qr
    robot.go_home()
    time.sleep(1)

    q_current = robot.get_joint_positions().numpy()
    pos, quat = robot.robot_model.forward_kinematics(torch.Tensor(q_current))
    quat_np = quat.numpy()
    quat_wxyz = np.r_[quat_np[3], quat_np[:3]]
    uq = sm.UnitQuaternion(quat_wxyz)
    rot = uq.SO3()

    # ✅ 목표 위치 (월드 좌표계 기준)
    target_xyz = [0.3, 0.1, 0.4]
    wTep = sm.SE3.Rt(rot, np.array(target_xyz)).A

    # 제어 시작
    arrived = False
    robot.start_joint_velocity_control(joint_vel_desired=torch.zeros(7))

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while not arrived:
        q_current = robot.get_joint_positions().numpy()
        arrived, qd_command = step_robot(q_current, wTep, frankie, robot.robot_model, ax=ax)
        robot.update_desired_joint_velocities(qd_command)
        time.sleep(0.02)

    robot.update_desired_joint_velocities(torch.zeros(7))
    print("도착 완료.")
    plt.ioff()
