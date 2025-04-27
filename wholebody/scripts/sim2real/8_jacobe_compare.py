import torch
import numpy as np
import spatialmath as sm
import roboticstoolbox as rtb
from polymetis import RobotInterface

def adjoint_transformation(T):
    """
    4x4 Transformation matrix -> 6x6 adjoint matrix
    """
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])
    adj = np.zeros((6, 6))
    adj[0:3, 0:3] = R
    adj[3:6, 0:3] = p_hat @ R
    adj[3:6, 3:6] = R
    return adj

def compare_jacobians(q_current, panda, robot_model):
    # Polymetis FK: position + quaternion
    pos, quat = robot_model.forward_kinematics(torch.Tensor(q_current))
    quat_np = quat.numpy()
    quat_wxyz = np.r_[quat_np[3], quat_np[:3]]
    rot = sm.UnitQuaternion(quat_wxyz).SO3()
    wTe = sm.SE3.Rt(rot, pos.numpy()).A  # World to EE transformation

    # ✅ Adjoint 변환 만들기 (Body → World)
    Ad_wTe = adjoint_transformation(wTe)

    # Polymetis Jacobian (Body frame 기준이라고 가정)
    jacobe_body = robot_model.compute_jacobian(torch.Tensor(q_current)).numpy()

    # ✅ 변환: Spatial frame 기준으로 변환
    jacobe_spatial = Ad_wTe @ jacobe_body

    # RTB Jacobian (World frame Spatial Jacobian)
    jacobe_rtb = panda.jacobe(q_current)

    # 결과 출력
    print("RTB EE Pose:\n", panda.fkine(q_current).A[:4, :4])
    print("Polymetis EE Pose (Tool added):\n", wTe[:4, :4])
    print("\nPolymetis Jacobian (converted to Spatial Frame):\n", jacobe_spatial)
    print("\nRTB Jacobian:\n", jacobe_rtb)

    # Difference 확인
    diff = jacobe_spatial - jacobe_rtb
    print("\nJacobian Difference (Polymetis - RTB):\n", diff)
    print("\nMax Difference:", np.max(np.abs(diff)))

if __name__ == "__main__":
    robot = RobotInterface(ip_address="localhost")
    panda = rtb.models.Panda()
    robot.go_home()

    # 현재 joint position 받아오기
    q_current = robot.get_joint_positions().numpy()

    # Jacobian 비교
    compare_jacobians(q_current, panda, robot.robot_model)
