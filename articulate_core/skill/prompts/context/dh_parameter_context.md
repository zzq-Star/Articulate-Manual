# DH Parameter Context
# ====================
# Standard Denavit-Hartenberg convention for robotic arm kinematics.

## Standard DH Convention
Each joint i is described by 4 parameters:
- a_i: link length (distance along x_i from z_i to z_{i+1})
- alpha_i: link twist (angle about x_i from z_i to z_{i+1})
- d_i: link offset (distance along z_{i-1} from x_{i-1} to x_i)
- theta_i: joint angle (angle about z_{i-1} from x_{i-1} to x_i)

## Transformation Matrix
The homogeneous transformation from frame i-1 to frame i is:
T_i = Rot(z, theta_i) * Trans(z, d_i) * Trans(x, a_i) * Rot(x, alpha_i)

## Notes
- For revolute joints, theta_i is the variable and d_i is fixed
- For prismatic joints, d_i is the variable and theta_i is fixed
- Base frame is frame 0, end-effector is frame n
- FK: T_0_n = T_1 * T_2 * ... * T_n
