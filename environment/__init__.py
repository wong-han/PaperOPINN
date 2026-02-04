from gymnasium.envs.registration import register


register(
    id="double_mass-v0",
    entry_point="environment.double_mass:DoubleMass",
)

register(
    id="linear2d-v0",
    entry_point="environment.linear_2d:Linear2D",
)

register(
    id="nonlinear2d-v0",
    entry_point="environment.nonlinear_2d:Nonlinear2D",
)

register(
    id="angle3d-v0",
    entry_point="environment.angle_3d:Angle3D",
)

register(
    id="quadrotor2d-v0",
    entry_point="environment.quadrotor2d:Quadrotor2D",
)

register(
    id="angle_rate-v0",
    entry_point="environment.angle_rate:AngleRate",
)

register(
    id="quadrotor3d-v0",
    entry_point="environment.quadrotor3d:Quadrotor3D",
)

register(
    id="quadrotor3d_TM-v0",
    entry_point="environment.quadrotor3d_TM:Quadrotor3D_TM",
)

register(
    id="quadrotor3d_TR-v0",
    entry_point="environment.quadrotor3d_TR:Quadrotor3D_TR",
)

register(
    id="custom_pendulum-v0",
    entry_point="environment.custom_pendulum:Pendulum",
)

register(
    id="winged_cone-v0",
    entry_point="environment.winged_cone:WingedCone",
)

register(
    id="winged_cone-v2",
    entry_point="environment.winged_cone_v2:WingedCone",
)