import argparse
import random
import traceback

import cv2
import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

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

def depth_on_control(m, d, gl_ctx, scn, cam, vopt, pert, ctx, viewport, cam_x_over_z, cam_y_over_z, sample_list,
                     ogl_zbuf=None, ogz_zbuf_inv=None, z_max_buf=None, C=None, D=None):
  try:
    test_pix = (int(RES_Y/2), int(RES_X/2))

    # Render the simulated camera
    mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    mujoco.mjr_render(viewport, scn, ctx)
    image = np.empty((RES_Y, RES_X, 3), dtype=np.uint8)
    depth = np.empty((RES_Y, RES_X, 1),    dtype=np.float32)
    mujoco.mjr_readPixels(image, depth, viewport, ctx)
    image = cv2.flip(image, 0) # OpenGL renders with inverted y axis
    depth = cv2.flip(depth, 0) # OpenGL renders with inverted y axis

    # Check XML reference, choice of zfar and znear can have big effect on accuracy
    zfar  = m.vis.map.zfar * m.stat.extent
    znear = m.vis.map.znear * m.stat.extent
    z_target_max = ogz_zbuf_inv(z_max_buf, znear, zfar) # Accuracy degrades rapidly after the values in the Z buffer reach a certain point

    # Show the simulated camera image
    # cv2.imshow('image', image / np.max(image)) # the color corresponds with the id

    if d.time > 0.0:
      depth = depth.astype(np.float64)
      depth_hat_buf = depth[test_pix]
      depth_linear    = ogz_zbuf_inv(depth, znear, zfar)
      depth_hat = depth_linear[test_pix] # Center of screen
      if C is not None:
        depth_linear_CD = ogz_zbuf_inv(depth, C=C, D=D)
        depth_hat_CD = depth_linear_CD[test_pix]

      # For visualization
      # depth_linear[depth_linear > m.vis.map.zfar - 0.0005] = 0 # Zero out depths farther than the z buffer

      # cv2.imshow('depth', depth_linear / np.max(depth_linear))

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
      depth_gt_plane = (pn_c.T @ n_c) / (n_c[0] * cam_x_over_z + n_c[1] * cam_y_over_z + n_c[2])
      depth_gt = depth_gt_plane[test_pix] # Center of screen

      # depth_gt = d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1]
      depth_gt_buf = ogl_zbuf(depth_gt, znear, zfar)
      error_buf    = depth_hat_buf - depth_gt_buf
      error = depth_hat - depth_gt

      if C is not None:
        depth_gt_buf_CD = ogl_zbuf(depth_gt, C=C, D=D)
        # print('CD buf', depth_gt_buf_CD - depth_gt_buf, depth_gt_buf_CD, depth_gt_buf)
        error_CD = depth_hat_CD - depth_gt
        error_buf_CD = depth_hat_buf - depth_gt_buf_CD
        # print('buf CD', error_buf_CD, depth_hat_buf, depth_gt_buf_CD)

        sample_list.append(error_CD)
      else:
        error_CD = 0.0
        error_buf_CD = 0.0
        sample_list.append(error)

      print('znear {:0.8f} zfar {:0.1f} hat_buf {:0.2e} max_gt {:0.2f} est {:0.2f} gt {:0.2f} instant {:0.2e} instant CD {:0.2e} instant buf {:0.2e} instant buf CD {:0.2e} mean {:0.2e} std dev {:0.2e} max {:0.2e} min {:0.2e} N {:0.02e}'.format(
        znear, zfar, depth_hat_buf, z_target_max, depth_hat, depth_gt,
        error,
        error_CD,
        error_buf,
        error_buf_CD,
        np.mean(sample_list),
        np.std(sample_list),
        np.max(np.abs(sample_list)),
        np.min(np.abs(sample_list)),
        len(sample_list)))

    target_mujoco_id = None
    for vgeom in scn.geoms:
      if vgeom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,vgeom.objid)
        if name == 'target':
          target_mujoco_id = vgeom.segid
    assert target_mujoco_id is not None

    threshed = image[:, :, 0] == target_mujoco_id + 1
    # cv2.imshow('threshed', threshed.astype(np.uint8) * 255)
    # cv2.waitKey(1)

    # Set the position of the moving target
    next_target_depth = (d.time * 2) % (z_target_max - znear) + znear
    # next_target_depth = (d.time * 2) % (z_target_max - 0.9 * z_target_max) + 0.9 * z_target_max
    # next_target_depth = z_target_max
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1] = next_target_depth

    # next_q_wp = R.from_rotvec(np.array([1.0, 0.0, 0.0]) * np.sin(d.time)).as_quat()
    next_q_wp = R.from_rotvec(np.array([1.0, 0.0, 0.0]) * 0 * np.pi / 180.0).as_quat()
    d.mocap_quat[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][0]   = next_q_wp[3]
    d.mocap_quat[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target')], :][1:4] = next_q_wp[0:3]

    # cloud_id = int(d.time / m.opt.timestep)
    # cloud_file_name = 'cloud_{:06d}.npy'.format(cloud_id)
    # np.save(cloud_file_name, p)

  except Exception as e:
    traceback.print_exc()
    print(e)
    raise e

def load_callback(m=None, d=None, ogl_zbuf=None, ogl_zbuf_inv=None, z_max_buf=None, C=None, D=None):
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

    # Turn on segmented rendering
    scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
    scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'cam')

    vopt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()

    ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
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

    fy = (RES_Y/2) / np.tan(yfov * np.pi / 180 / 2)
    fx = fy
    cx = (RES_X - 1) / 2.0 - 0.5 # Check this??
    cy = (RES_Y - 1) / 2.0 - 0.5
    cam_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Debugging, build cam_x_over_z and cam_y_over_z from perspective matrix?
    # camProject
    # 1.3579951525e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 
    # 0.0000000000e+00 2.4142136574e+00 0.0000000000e+00 0.0000000000e+00 
    # 0.0000000000e+00 0.0000000000e+00 5.0250887871e-03 -1.0000000000e+00 
    # 0.0000000000e+00 0.0000000000e+00 1.1471571028e-01 0.0000000000e+00

    # Get the 3D direction vector for each pixel in the simulated sensor
    # in the format (x, y, 1)
    cam_x_over_z, cam_y_over_z = cv2.initInverseRectificationMap(
            cam_K, # Intrinsics
            None, # Distortion (0 for GPU rendered images)
            np.eye(3), # Rectification
            np.eye(3), # Unity rectification intrinsics (we want direction vector)
            (RES_X, RES_Y), # Test all pixels in physical sensor
            m1type=cv2.CV_32FC1)
    cam_x_over_z = cam_x_over_z.astype(np.float64)
    cam_y_over_z = cam_y_over_z.astype(np.float64)

    sample_list = []

    # Set the callback and capture all variables needed for rendering
    mujoco.set_mjcb_control(
      lambda m, d: depth_on_control(
        m, d, gl_ctx, scn, cam, vopt, pert, ctx, viewport, cam_x_over_z, cam_y_over_z, sample_list,
        ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D))

  return m , d

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--z_buf_type', type=str, help='ogl_default, ogl_negz', required=True)
  args = parser.parse_args()

  # Accuracy degrades rapidly after the values in the Z buffer reach a certain point
  # Choose this at your discretion
  if args.z_buf_type == 'ogl_default':
    ogl_zbuf     = ogl_zbuf_default
    ogl_zbuf_inv = ogl_zbuf_default_inv
    z_max_buf = 0.993
    C = -1.0100501776e+00
    D = -2.2943142056e-01
  elif args.z_buf_type == 'ogl_negz':
    ogl_zbuf     = ogl_zbuf_negz
    ogl_zbuf_inv = ogl_zbuf_negz_inv
    C = 5.0250887871e-03
    D = 1.1471571028e-01
    z_max_buf = 0.007
  else:
    raise Exception('Unrecognized z_buf_type')

  viewer.launch(loader=lambda m=None, d=None: load_callback(m, d, ogl_zbuf, ogl_zbuf_inv, z_max_buf, C, D))
