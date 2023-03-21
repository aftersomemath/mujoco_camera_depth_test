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

And display something like
![Screenshot from 2023-03-20 19-55-25](https://user-images.githubusercontent.com/6125615/226490672-d33769ee-668e-4f88-8be0-579ca271e5cd.png)

The script saves off point clouds for visualization with pyqtgraph. After running `camera_depth_test.py`, run 
```bash
python visualize_cloud.py
```

The cloud should look something like this:
![Screenshot from 2023-03-20 19-52-25](https://user-images.githubusercontent.com/6125615/226490707-691c093c-1058-481d-a79b-3a223e076935.png)
