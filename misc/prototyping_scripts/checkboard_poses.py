import numpy as np
from scipy.spatial.transform import Rotation

from point_cloud_world import Checkerboard

import pdb



def calc_gsc(Rsb, Tsb, Rbc, Tbc):
    Tsb = np.reshape(Tsb, (3,1))
    Tbc = np.reshape(Tbc, (3,1))
    Rsc = Rsb @ Rbc
    Tsc = Rsb @ Tbc + Tsb
    gsc = np.hstack((Rsc, Tsc))
    return gsc


def pprint(output_list):
    for item in output_list:
        Xs = item[1]
        Xc = item[2]
        xc = np.array([ Xc[0]/Xc[2], Xc[1]/Xc[2]])
        print("{}, Xs={}, xc={}, Z={}".format(item[0], Xs, xc, Xc[2]))



nsquares_width = 7
nsquares_height = 6
square_width = 0.05
half_width = square_width * nsquares_width / 2
half_height = square_width * nsquares_height / 2
board_y = 0.25

PCW = Checkerboard(square_width=square_width,
                   dim_squares=(nsquares_width, nsquares_height),
                   bot_right_coord=(-half_width, board_y, -half_height),
                   plane="xz")


# Camera-IMU extrinsics
Rbc = Rotation.from_euler('XYZ', [-np.pi/2, 0, 0], degrees=False).as_matrix()
Tbc = np.zeros((3,1))
gbc = np.hstack((Rbc, Tbc))


# Camera intrinsics
fx = 275.0
fy = 275.0
imw = 640.0
imh = 480.0
cx = imw / 2
cy = imh / 2
K = np.array([[  fx,   0,  cx ],
              [   0,  fy,  cy ],
              [   0,   0,   1 ]])

# Useful matrix
Rx_delta = Rotation.from_euler('XYZ', [-20, 0, 0], degrees=True).as_matrix()
Ry_90 = Rotation.from_euler('XYZ', [0, np.pi/2, 0]).as_matrix()
Ry_delta = Rotation.from_euler('XYZ', [0, 1, 0], degrees=True).as_matrix()


# Initial position
Rsb0 = np.eye(3)
Tsb0 = np.zeros((3,1))
check0 = PCW.allCornersVisible(calc_gsc(Rsb0, Tsb0, Rbc, Tbc), K, imw, imh)
print("Initial position")
pprint(check0)

# Initial position with 90 degrees cyclorotation
Rsb0_90 = Ry_90 @ Rsb0
Tsb0_90 = Tsb0
check0_90 = PCW.allCornersVisible(calc_gsc(Rsb0_90, Tsb0_90, Rbc, Tbc), K, imw, imh)
print("\nInitial position rotated 90 degrees")
pprint(check0_90)

# Center, top
Rsb1 = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi]).as_matrix()
Tsb1 = np.array([0, board_y, half_height+0.175])
check1 = PCW.allCornersVisible(calc_gsc(Rsb1, Tsb1, Rbc, Tbc), K, imw, imh)
print("\nTop")
pprint(check1)

# Top right corner
Rsb2 = Rotation.from_euler('XYZ', [np.pi/2, 0, 3*np.pi/4]).as_matrix()
Tsb2 = np.array([ half_width+0.05, board_y, half_height+0.05])
check2 = PCW.allCornersVisible(calc_gsc(Rsb2, Tsb2, Rbc, Tbc), K, imw, imh)
print("\nTop-right corner")
pprint(check2)

# Center, bottom
Rsb3 = Rotation.from_euler('XYZ', [np.pi/2, 0, 0]).as_matrix()
Tsb3 = np.array([ 0, board_y, -half_height-0.175])
check3 = PCW.allCornersVisible(calc_gsc(Rsb3, Tsb3, Rbc, Tbc), K, imw, imh)
print("\nBottom")
pprint(check3)

# Left side
Rsb4 = Rotation.from_euler('XYZ', [np.pi/2, 0, -np.pi/2]).as_matrix()
Tsb4 = np.array([ -half_width-0.175, board_y, 0])
check4 = PCW.allCornersVisible(calc_gsc(Rsb4, Tsb4, Rbc, Tbc), K, imw, imh)
print("\nLeft side")
pprint(check4)

# Right side
Rsb5 = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi/2]).as_matrix()
Tsb5 = np.array([ half_width+0.175, board_y, 0])
check5 = PCW.allCornersVisible(calc_gsc(Rsb5, Tsb5, Rbc, Tbc), K, imw, imh)
print("\nRight side")
pprint(check5)

# Top left corner
Rsb6 = Rotation.from_euler('XYZ', [np.pi/2, 0, -3*np.pi/4]).as_matrix()
Tsb6 = np.array([ -half_width-0.05, board_y, half_height+0.05])
check6 = PCW.allCornersVisible(calc_gsc(Rsb6, Tsb6, Rbc, Tbc), K, imw, imh)
print("\nTop-Left corner")
pprint(check6)

# Bottom left corner
Rsb7 = Rotation.from_euler('XYZ', [np.pi/2, 0, -np.pi/4]).as_matrix()
Tsb7 = np.array([ -half_width-0.05, board_y, -half_height-0.05])
check7 = PCW.allCornersVisible(calc_gsc(Rsb7, Tsb7, Rbc, Tbc), K, imw, imh)
print("\nBottom-Left corner")
pprint(check7)

# Bottom right corner
Rsb8 = Rotation.from_euler('XYZ', [np.pi/2, 0, np.pi/4]).as_matrix()
Tsb8 = np.array([ half_width+0.05, board_y, -half_height-0.05])
check8 = PCW.allCornersVisible(calc_gsc(Rsb8, Tsb8, Rbc, Tbc), K, imw, imh)
print("\nBottom-Right corner")
pprint(check8)