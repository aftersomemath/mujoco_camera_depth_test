import traceback

import mujoco
import mujoco.viewer as viewer
import numpy as np
import cv2

import matplotlib.pyplot as plt

RES_X = 1280
RES_Y = 720
# Accuracy degrades rapidly after the values in the Z buffer reach a certain point
# Choose this at your discretion
Z_BUF_MAX = 0.993

# Linearize Depth from an OpenGL buffer
def linearize_depth(depth, znear, zfar):
    zlinear = (znear * zfar) / (zfar + depth * (znear - zfar))
    return zlinear

def depth_on_control(m, d, gl_ctx, scn, cam, vopt, pert, ctx, viewport, cam_x_over_z, cam_y_over_z, sample_list):
  try:
    # Render the simulated camera
    mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(viewport, scn, ctx)
    image = np.empty((RES_Y, RES_X, 3), dtype=np.uint8)
    depth = np.empty((RES_Y, RES_X, 1),    dtype=np.float32)
    mujoco.mjr_readPixels(image, depth, viewport, ctx)
    image = cv2.flip(image, 0) # OpenGL renders with inverted y axis
    depth = cv2.flip(depth, 0) # OpenGL renders with inverted y axis

    # Show the simulated camera image
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Check XML reference, choice of zfar and znear can have big effect on accuracy
    zfar  = m.vis.map.zfar * m.stat.extent
    znear = m.vis.map.znear * m.stat.extent
    depth_linear = linearize_depth(depth, znear=znear, zfar=zfar)

    # ground truth depth up to the box width which is not accounted for (set to 0.0000001 in XML)
    depth_gt = d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1]
    depth_hat = depth_linear[int(RES_Y/2), int(RES_X/2)] # Center of screen

    error = depth_hat - depth_gt

    sample_list.append(error)

    z_target_max = linearize_depth(Z_BUF_MAX, znear, zfar) # Accuracy degrades rapidly after the values in the Z buffer reach a certain point
    print('znear {:0.1f} zfar {:0.1f} max_gt {:0.2f} est {:0.2f} gt {:0.2f} instant abs {:0.2e} mean {:0.2e} std dev {:0.2e} max {:0.2e} min {:0.2e} N {:0.02e}'.format(
      znear, zfar, z_target_max, depth_hat, depth_gt,
      np.abs(depth_hat-depth_gt),
      np.mean(sample_list),
      np.std(sample_list),
      np.max(np.abs(sample_list)),
      np.min(np.abs(sample_list)),
      len(sample_list)))

    # For visualization
    depth_linear[depth_linear > m.vis.map.zfar - 0.0005] = 0 # Zero out depths farther than the z buffer

    # Set the position of the moving target
    next_target_depth = (d.time * 2) % (z_target_max - znear) + znear
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1] = next_target_depth

    cv2.imshow('depth', depth_linear / np.max(depth_linear))
    cv2.waitKey(1)

    # Save off some pointclouds for visualization later
    p_X = cam_x_over_z * depth_linear
    p_Y = cam_y_over_z * depth_linear
    p_Z = depth_linear
    p = np.dstack((p_X, p_Y, p_Z))

    cloud_id = int(d.time / m.opt.timestep)
    cloud_file_name = 'cloud_{:06d}.npy'.format(cloud_id)
    np.save(cloud_file_name, p)

  except Exception as e:
    traceback.print_exc()
    print(e)
    raise e

def load_callback(m=None, d=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path('./camera_depth_test.xml')
  d = mujoco.MjData(m)

  if m is not None:
    # Make all the things needed to render a simulated camera
    gl_ctx = mujoco.GLContext(RES_X, RES_Y)
    gl_ctx.make_current()

    scn = mujoco.MjvScene(m, maxgeom=100)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'cam')

    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

    viewport = mujoco.MjrRect(0, 0, RES_X, RES_Y)

    yfov = m.cam_fovy[cam.fixedcamid]
    fy = (RES_Y/2) / np.tan(yfov * np.pi / 180 / 2)
    fx = fy
    cx = (RES_X-1) / 2.0
    cy = (RES_Y-1) / 2.0

    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Get the 3D direction vector for each pixel in the simulated sensor
    # in the format (x, y, 1)
    cam_x_over_z, cam_y_over_z = cv2.initInverseRectificationMap(
            cam_K, # Intrinsics
            None, # Distortion (0 for GPU rendered images)
            np.eye(3), # Rectification
            np.eye(3), # Unity rectification intrinsics (we want direction vector)
            (RES_X, RES_Y), # Test all pixels in physical sensor
            m1type=cv2.CV_32FC1)

    sample_list = []

    # Set the callback and capture all variables needed for rendering
    mujoco.set_mjcb_control(
      lambda m, d: depth_on_control(
        m, d, gl_ctx, scn, cam, vopt, pert, ctx, viewport, cam_x_over_z, cam_y_over_z, sample_list))

  return m , d

if __name__ == '__main__':
  viewer.launch(loader=load_callback)
