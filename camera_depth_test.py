import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # Otherwise numpy spawns way too many threads

import argparse
import glob
import random
import time
import traceback

# import cv2
import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import scipy

# TODO read these from mjModel
RES_X = 1280
RES_Y = 720

def glFrustum_CD_float32(znear, zfar):
  zfar  = np.float32(zfar)
  znear = np.float32(znear)
  C = -(zfar + znear)/(zfar - znear)
  D = -(np.float32(2)*zfar*znear)/(zfar - znear)
  return C, D

def ogl_zbuf_projection(zlinear, C, D):
  zbuf = -C + (1/zlinear)*D # TODO why -C?
  return zbuf

def ogl_zbuf_projection_inverse(zbuf, C, D):
  zlinear = 1 / ((zbuf - (-C)) / D) # TODO why -C?
  return zlinear

def ogl_zbuf_default(zlinear, znear=None, zfar=None, C=None, D=None):
  if C is None:
    C, D = glFrustum_CD_float32(znear, zfar)
  zbuf = ogl_zbuf_projection(zlinear, C, D)
  zbuf_scaled = 0.5 * zbuf + 0.5
  return zbuf_scaled

def ogl_zbuf_negz(zlinear, znear=None, zfar=None, C=None, D=None):
  if C is None:
    C, D = glFrustum_CD_float32(znear, zfar)
    C = np.float32(-0.5)*C - np.float32(0.5)
    D = np.float32(-0.5)*D
  zlinear = ogl_zbuf_projection(zlinear, C, D)
  return zlinear

def ogl_zbuf_default_inv(zbuf_scaled, znear=None, zfar=None, C=None, D=None):
  if C is None:
    C, D = glFrustum_CD_float32(znear, zfar)
  zbuf = 2.0 * zbuf_scaled - 1.0
  zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
  return zlinear

def ogl_zbuf_negz_inv(zbuf, znear=None, zfar=None, C=None, D=None):
  if C is None:
    C, D = glFrustum_CD_float32(znear, zfar)
    C = np.float32(-0.5)*C - np.float32(0.5)
    D = np.float32(-0.5)*D
  zlinear = ogl_zbuf_projection_inverse(zbuf, C, D)
  return zlinear

def plot_errors(errors):
  rows = len(errors)
  cols = 3

  for i, (name, error) in enumerate(errors.items()):
      plt.subplot(rows, cols, 1 + i*3 + 0)
      plt.title(name)
      plt.plot(error[:, 0], error[:, 1])
      plt.ylabel('mean')
      plt.grid(True)
      plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
      plt.xlabel('target z')

      plt.subplot(rows, cols, 1 + i*3 + 1)
      plt.title(name)
      plt.plot(error[:, 0], error[:, 2])
      plt.ylabel('std dev')
      plt.grid(True)
      plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
      plt.xlabel('target z')

      # plt.subplot(rows, cols, 1 + i*4 + 2)
      # plt.title(name)
      # plt.plot(error[:, 0], error[:, 3])
      # plt.ylabel('min')
      # plt.grid(True)
      # plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
      # plt.xlabel('target z')

      plt.subplot(rows, cols, 1 + i*3 + 2)
      plt.title(name)
      plt.plot(error[:, 0], error[:, 4])
      plt.ylabel('max')
      plt.grid(True)
      plt.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
      plt.xlabel('target z')

  # plt.tight_layout()
  # plt.show()

def plot_errors_all():
  files = sorted(glob.glob('errors_*.npz'))

  plt.figure()
  for f in files:
    plot_errors(np.load(f))
  plt.legend(files)
  plt.show()

def objective_one_frame(cam_x_over_z, cam_y_over_z, optimization_dictionary):
  pn_c      = optimization_dictionary['pn_c']
  n_c       = optimization_dictionary['n_c']
  depth_hat = optimization_dictionary['depth_hat']
  depth_gt = (pn_c.T @ n_c) / (n_c[0] * cam_x_over_z + n_c[1] * cam_y_over_z + n_c[2])
  error     = depth_hat - depth_gt
  return error.flatten()

# TODO optimize C and D?
def optimize_intrinsics(optimization_dictionaries, N):
  args_i = np.round(np.linspace(0, len(optimization_dictionaries)-1, N)).astype(np.int64)
  optimization_dictionaries_decimated = [optimization_dictionaries[i] for i in args_i]

  RES_X = optimization_dictionaries_decimated[0]['RES_X']
  RES_Y = optimization_dictionaries_decimated[0]['RES_Y']
  x0    = optimization_dictionaries_decimated[0]['intrinsics']

  x = np.arange(0, RES_X)
  y = np.arange(0, RES_Y)
  xx, yy = np.meshgrid(x, y)
  xx_yy_one = np.dstack((xx, yy, np.ones((RES_Y, RES_X))))
  def objective(x):
    fx, fy, cx, cy = x
    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cam_K_inv = np.linalg.inv(cam_K)

    cam_p_over_z = ((cam_K_inv @ xx_yy_one.reshape((-1, 3)).T).T).reshape((720, 1280, 3))
    cam_x_over_z = cam_p_over_z[:, :, 0]
    cam_y_over_z = cam_p_over_z[:, :, 1]

    residuals = []
    for optimization_dictionary in optimization_dictionaries_decimated:
      pn_c      = optimization_dictionary['pn_c']
      n_c       = optimization_dictionary['n_c']
      depth_hat = optimization_dictionary['depth_hat']
      depth_gt = (pn_c.T @ n_c) / (n_c[0] * cam_x_over_z + n_c[1] * cam_y_over_z + n_c[2])
      error     = depth_hat - depth_gt
      residuals.append(error.flatten())

    residuals = np.concatenate(residuals)
    return residuals

  result = scipy.optimize.least_squares(objective, x0=x0, verbose=2)
  print(result)
  print('delta intrinsics', result.x - x0)
  return result.x

def collect_data(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, view, m, d, random_angle):
  scn = mujoco.MjvScene(m, maxgeom=100)

  # Turn on segmented rendering
  scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
  scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
  cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'cam')

  vopt = mujoco.MjvOption()
  pert = mujoco.MjvPerturb()

  ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150, depth_mapping, depth_precision)
  mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

  viewport = mujoco.MjrRect(0, 0, RES_X, RES_Y)

  yfov = m.cam_fovy[cam.fixedcamid]

  # Get the center pixel right
  # Currently can't explain the extra -0.5 pixel offset
  # this article might be relevant, but after working it through
  # I still get that the -0.5 offset should not be helping as it is (3-4x more accurate with offset)
  # https://www.realtimerendering.com/blog/the-center-of-the-pixel-is-0-50-5/#:~:text=OpenGL%20has%20always%20considered%20the,the%20program%20with%20DirectX%2010.
  # https://lmb.informatik.uni-freiburg.de/people/reisert/opengl/doc/glFrustum.html
  # https://registry.khronos.org/OpenGL-Refpages/gl4/html/glViewport.xhtml
  # https://stackoverflow.com/a/25468051
  # https://community.khronos.org/t/multisampled-depth-renderbuffer/55751/8
  # https://www.khronos.org/opengl/wiki/Fragment_Shader#System_inputs
  # khronos.org/opengl/wiki/Type_Qualifier_(GLSL)#Interpolation_qualifiers
  # https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_SamplePosition.xhtml
  # "When rendering to a non-multisample buffer, or if multisample rasterization is disabled, gl_SamplePosition will be (0.5, 0.5)."
  # https://community.amd.com/t5/archives-discussions/how-depth-is-interpolated-in-rasterizer-linear-depth-instead-of/td-p/390440
  # https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/
  # http://www.humus.name/Articles/Persson_CreatingVastGameWorlds.pdf
  # https://developer.nvidia.com/content/depth-precision-visualized
  # https://www.lighthouse3d.com/tutorials/glsl-tutorial/rasterization-and-interpolation

  if intrinsics is None:
    fy = (RES_Y/2) / np.tan(yfov * np.pi / 180 / 2)
    fx = fy
    cx = (RES_X - 1) / 2.0 + 0.125       # These offsets are very close to what the optimization returns when using
    cy = (RES_Y - 1) / 2.0 - 0.5 + 0.125 # a float32 reversed Z buffer, 24 bit non reversed results in different offsets
  else:
    fx, fy, cx, cy = intrinsics

  x = np.arange(0, RES_X)
  y = np.arange(0, RES_Y)
  xx, yy = np.meshgrid(x, y)
  xx_yy_one = np.dstack((xx, yy, np.ones((RES_Y, RES_X))))
  cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
  cam_K_inv = np.linalg.inv(cam_K)
  cam_p_over_z = ((cam_K_inv @ xx_yy_one.reshape((-1, 3)).T).T).reshape((720, 1280, 3))
  cam_x_over_z = cam_p_over_z[:, :, 0]
  cam_y_over_z = cam_p_over_z[:, :, 1]

  sample_list = []

  error_list        = []
  error_buf_list    = []
  if C is not None:
    error_CD_list     = []
    error_buf_CD_list = []

  optimization_dictionaries = []

  # Check XML reference, choice of zfar and znear can have big effect on accuracy
  zfar  = m.vis.map.zfar * m.stat.extent
  znear = m.vis.map.znear * m.stat.extent
  z_target_max = ogl_zbuf_inv(z_max_buf, znear, zfar)

  while True:
    if view is not None and not view.is_running():
      break

    # Render the simulated camera
    mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(viewport, scn, ctx)
    image = np.empty((RES_Y, RES_X, 3), dtype=np.uint8)
    depth_hat_buf = np.empty((RES_Y, RES_X, 1),    dtype=np.float32)
    mujoco.mjr_readPixels(image, depth_hat_buf, viewport, ctx)

    # OpenGL renders with inverted y axis
    image         = np.flip(image, axis=0).squeeze()
    depth_hat_buf = np.flip(depth_hat_buf, axis=0).squeeze()

    target_mujoco_id = None
    for vgeom in scn.geoms:
      if vgeom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,vgeom.objid)
        if name == 'target':
          target_mujoco_id = vgeom.segid
    assert target_mujoco_id is not None

    if d.time > 0.0:
      target_pixels = image[:, :, 0] == target_mujoco_id + 1
      # cv2.imshow('target_pixels', target_pixels.astype(np.uint8) * 255)
      # cv2.waitKey(1)
      assert np.all(target_pixels)

      depth_hat_buf = depth_hat_buf.astype(np.float64)
      depth_hat = ogl_zbuf_inv(depth_hat_buf, znear, zfar)
      if C is not None:
        depth_hat_CD = ogl_zbuf_inv(depth_hat_buf, C=C, D=D)

      # For visualization
      # cv2.imshow('depth', depth_hat / np.max(depth_hat))
      # cv2.waitKey(1)

      # # Save off some pointclouds for visualization later
      # p_X = cam_x_over_z * depth_linear
      # p_Y = cam_y_over_z * depth_linear
      # p_Z = depth_linear
      # #p = np.dstack((p_X, p_Y, p_Z))

      # The 3D normal of the visible plane in the planes coordinate frame (this is time invariant)
      R_wp = d.body('target').xmat.reshape((3,3))
      t_wp = d.body('target').xpos

      R_wg = d.camera('cam').xmat.reshape((3,3))
      t_wg = d.camera('cam').xpos

      # Get the target odometry in the drone frame
      # that will be used for state estimation
      # OpenGL camera frame to camera
      R_gc = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])
      R_wc = R_wg @ R_gc
      t_wc = t_wg

      R_cp =  R_wc.T @ R_wp
      t_cp = -R_wc.T @ t_wc + R_wc.T @ t_wp

      # Normal in the planes coordinate frame
      n_p  = np.array([0.0, -1.0, 0.0])
      # position of the normal in the planes coordinate frame
      pn_n = np.array([0.0, 0.0, 0.0])

      # Normal and position of normal in the camera frame
      n_c  = R_cp @ n_p
      pn_c = R_cp @ pn_n + t_cp

      # Todo it seems like we are not as close as expected
      # print(pn_c, n_c, depth_gt)

      # Calculate depth of intersection
      # points on plane satisfy
      # (x - pn_c)^T n_c = 0
      # implying
      # x^T n_c - pn_c^T n_c = 0
      #
      # rays from the camera satisfy
      # x_1 = cam_x_over_z * x_3
      # x_2 = cam_y_over_z * x_3
      # x_3 = x_3
      #
      # So we have
      # x_3 * [cam_x_over_z, cam_y_over_z, 1] n_c - pn_c^T n_c = 0
      # x_3 = (pn_c^T n_c) / ([cam_x_over_z, cam_y_over_z, 1] n_c)
      depth_gt = (pn_c.T @ n_c) / (n_c[0] * cam_x_over_z + n_c[1] * cam_y_over_z + n_c[2])
      depth_gt_buf = ogl_zbuf(depth_gt, znear, zfar)
      if C is not None:
        depth_gt_buf_CD = ogl_zbuf(depth_gt, C=C, D=D)

      optimization_dictionary = {
        'intrinsics': np.array([cam_K[0, 0], cam_K[1,1], cam_K[0, 2], cam_K[1, 2]]),
        'RES_X': RES_X, 'RES_Y': RES_Y,
        'pn_c': pn_c,
        'n_c': n_c,
        'znear': znear,
        'zfar': zfar,
        'depth_hat': depth_hat,
      }

      if C is not None:
        optimization_dictionary['C'] = C
        optimization_dictionary['D'] = D

      optimization_dictionaries.append(optimization_dictionary)

      # depth_gt = d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1]
      
      # Assume all pixels on target
      error     = depth_hat    - depth_gt
      error_buf = depth_hat_buf - depth_gt_buf
      if C is not None:
        error_CD     = depth_hat_CD  - depth_gt
        error_buf_CD = depth_hat_buf - depth_gt_buf_CD

      abs_error = np.abs(error)
      error_list.append([
        d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1],
        np.mean(error),
        np.std (error),
        np.min (abs_error),
        np.max (abs_error),
      ])

      abs_error_buf = np.abs(error_buf)
      error_buf_list.append([
        d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1],
        np.mean(error_buf),
        np.std (error_buf),
        np.min (abs_error_buf),
        np.max (abs_error_buf),
      ])

      if C is not None:
        abs_error_CD = np.abs(error_CD)
        error_CD_list.append([
          d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1],
          np.mean(error_CD),
          np.std (error_CD),
          np.min (abs_error_CD),
          np.max (abs_error_CD),
        ])

        abs_error_buf_CD = np.abs(error_buf_CD)
        error_buf_CD_list.append([
          d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1],
          np.mean(error_buf_CD),
          np.std (error_buf_CD),
          np.min (abs_error_buf_CD),
          np.max (abs_error_buf_CD),
        ])

    mujoco.mj_step(m, d)
    # time.sleep(0.5)

    if view is not None:
      view.sync()

    # Set the position of the moving target
    if d.time * 2 > z_target_max - znear:
      break

    next_target_depth = (d.time * 2) % (z_target_max - znear) + znear
    # next_target_depth = (d.time * 2) % (z_target_max - 0.9 * z_target_max) + 0.9 * z_target_max
    # next_target_depth = z_target_max
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1] = next_target_depth

    if random_angle:
      rot_x = random.uniform(-1.0, 1.0)
      rot_y = random.uniform( 0.0, 1.0)
      angle = random.uniform(-30.0, 30.0)
    else:
      rot_x = 1.0
      rot_y = 1.0
      angle = 30.0

    next_q_wp = R.from_rotvec(np.array([rot_x, rot_y, 0.0]) * angle * np.pi / 180.0).as_quat()
    d.mocap_quat[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][0]   = next_q_wp[3]
    d.mocap_quat[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1:4] = next_q_wp[0:3]

  errors = {}
  errors['error']        = np.array(error_list)
  # errors['error_buf']    = np.array(error_buf_list)
  # if C is not None:
  #   errors['error_CD']     = np.array(error_CD_list)
  #   errors['error_buf_CD'] = np.array(error_buf_CD_list)

  return errors, optimization_dictionaries

def run_test(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, save_name, use_viewer):
  m = mujoco.MjModel.from_xml_path('./camera_depth_test.xml')
  d = mujoco.MjData(m)

  gl_ctx = mujoco.GLContext(RES_X, RES_Y)
  gl_ctx.make_current()

  if use_viewer:
    view = viewer.launch_passive(m, d)
  else:
    view = None

  random.seed(None) # Use system time as seed for data (random training set)
  errors, optimization_dictionaries = collect_data(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, view, m, d, random_angle=True)
  if view:
    view.close()

  new_intrinsics = optimize_intrinsics(optimization_dictionaries, 20) # TODO parameter

  d = mujoco.MjData(m)
  if use_viewer:
    view = viewer.launch_passive(m, d)
  random.seed(0) # Seed random angle with 0 (non random test set, which will be the same for all versions of z buffer)
  new_errors, new_optimization_dictionaries = collect_data(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, new_intrinsics, view, m, d, random_angle=True)
  if view:
    view.close()

  # Optimize intrinsics again if you want to verify that
  # changing the data didn't change the intrinsics much
  # optimize_intrinsics(new_optimization_dictionaries, 10)

  np.savez(save_name, **new_errors)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  #parser.add_argument('--z_buf_type', type=str, help='ogl_default, ogl_negz')
  #parser.add_argument('--plot', action='store_true')
  parser.add_argument('--viewer', action='store_true')
  args = parser.parse_args()

  z_max_buf_negz = (0.0035 / 2) + (0.0035 / 4) + (0.0035 / 8)

  # mjDB_NEGONETOONE, mjDB_INT24
  depth_mapping =  mujoco.mjtDepthMapping.mjDB_NEGONETOONE
  depth_precision = mujoco.mjtDepthPrecision.mjDB_INT24
  ogl_zbuf     = ogl_zbuf_default
  ogl_zbuf_inv = ogl_zbuf_default_inv
  z_max_buf = 1.0 - z_max_buf_negz
  C = None #-1.0080322027e+00
  D = None #-2.0080322027e-01
  intrinsics = None #np.array([869.11688245,  869.12557739,  639.,         359.12569043])
  save_name = 'errors_default_int24'
  run_test(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, save_name, args.viewer)

  # mjDB_NEGONETOONE, mjDB_FLOAT32
  depth_mapping =  mujoco.mjtDepthMapping.mjDB_NEGONETOONE
  depth_precision = mujoco.mjtDepthPrecision.mjDB_FLOAT32
  ogl_zbuf     = ogl_zbuf_default
  ogl_zbuf_inv = ogl_zbuf_default_inv
  z_max_buf = 1.0 - z_max_buf_negz
  C = None
  D = None
  intrinsics = None
  save_name = 'errors_default_float32'
  run_test(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, save_name, args.viewer)

  # mjDB_ONETOZERO, mjDB_INT24
  depth_mapping =  mujoco.mjtDepthMapping.mjDB_ONETOZERO
  depth_precision = mujoco.mjtDepthPrecision.mjDB_INT24
  ogl_zbuf     = ogl_zbuf_negz
  ogl_zbuf_inv = ogl_zbuf_negz_inv
  z_max_buf = z_max_buf_negz
  C = None
  D = None
  intrinsics = None
  save_name = 'errors_revz_int24'
  run_test(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, save_name, args.viewer)

  # mjDB_ONETOZERO, mjDB_FLOAT32
  depth_mapping =  mujoco.mjtDepthMapping.mjDB_ONETOZERO
  depth_precision = mujoco.mjtDepthPrecision.mjDB_FLOAT32
  ogl_zbuf     = ogl_zbuf_negz
  ogl_zbuf_inv = ogl_zbuf_negz_inv
  z_max_buf = z_max_buf_negz
  C = None #4.0161013603e-03
  D = None #1.0040161014e-01
  intrinsics = None #np.array([869.11688245, 869.11653695, 639.        , 359.12489134])
  save_name = 'errors_revz_float32'
  run_test(depth_mapping, depth_precision, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D, intrinsics, save_name, args.viewer)

  plot_errors_all()
