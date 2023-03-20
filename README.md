# MuJoCo Camera Depth Test

A quick script to test the accuracy of the depth from an OpenGL camera visualized with MuJoCo.

To run:

```bash
python camera_depth_test.py
```

Should print something like:

```
znear 0.1 zfar 22.8 est 1.08 gt 1.08 instant abs 7.58e-07 mean -1.30e-06 std dev 1.69e-05 max 7.84e-05 min 2.38e-09 N 1.06e+03
```

The script saves off point clouds for visualization with pyqtgraph. After running `camera_depth_test.py`, run 
```bash
python visualize_cloud.py
```
