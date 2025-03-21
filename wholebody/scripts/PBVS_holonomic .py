import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math
import matplotlib.pyplot as plt

def step_robot(r: rtb.ERobot, Tep):
    wTe = r.fkine(r.q)
    eTep = np.linalg.inv(wTe) @ Tep
    et = np.sum(np.abs(eTep[:3, -1]))
    Y = 0.01
    Q = np.eye(r.n + 6)
    Q[: r.n, : r.n] *= Y
    Q[:3, :3] *= 1.0 / et
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)
    v, _ = rtb.p_servo(wTe, Tep, 1.5)
    v[3:] *= 1.3
    Aeq = np.c_[r.jacobe(r.q), np.eye(6)]
    beq = v.reshape((6,))
    Aeq = np.vstack((Aeq, np.zeros((1, Aeq.shape[1]))))
    Aeq[-1, 2] = 1
    beq = np.append(beq, 0)
    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)
    ps = 0.1
    pi = 0.9
    Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)
    c = np.concatenate(
        (np.zeros(3), -r.jacobm(start=r.links[5]).reshape((r.n - 3,)), np.zeros(6))
    )
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε
    lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='osqp')
    qd = qd[: r.n]
    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4
    if et < 0.02:
        return True, qd
    else:
        return False, qd

env = swift.Swift()
env.launch(realtime=True)
ax_goal = sg.Axes(0.1)
env.add(ax_goal)
frankie = rtb.models.FrankieOmni()
frankie.q = frankie.qr
env.add(frankie)
arrived = False
dt = 0.025
env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
wTep = frankie.fkine(frankie.q) * sm.SE3.Rz(np.pi)
wTep.A[:3, :3] = np.diag([-1, 1, -1])
wTep.A[0, -1] -= 4.0
wTep.A[2, -1] -= 0.25
ax_goal.T = wTep
env.step()

qd_values = []
while not arrived:
    arrived, frankie.qd = step_robot(frankie, wTep.A)
    qd_values.append(frankie.qd[:3])
    env.step(dt)
    base_new = frankie.fkine(frankie._q, end=frankie.links[3]).A
    frankie._T = base_new
    frankie.q[:3] = 0


qd_values = np.array(qd_values)
plt.figure(figsize=(10, 6))
plt.plot(qd_values[:, 0], label="qd[0]")
plt.plot(qd_values[:, 1], label="qd[1]")
plt.plot(qd_values[:, 2], label="qd[2]")
plt.xlabel("Time step")
plt.ylabel("Joint velocity")
plt.title("Joint Velocities over Time")
plt.legend()
plt.grid(True)
plt.show()

env.hold()