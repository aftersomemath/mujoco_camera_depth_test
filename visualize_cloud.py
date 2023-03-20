"""
pyqtgraph GLScatterPlotItem demo adapted to plot point clouds

Demonstrates use of GLScatterPlotItem with rapidly-updating plots.

Copyright (c) 2012  University of North Carolina at Chapel Hill
Luke Campagnola    ('luke.campagnola@%s.com' % 'gmail')

The MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Footer

"""

import glob
import time

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("GLScatterPlotItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
w.setCameraPosition(distance=20)

g = gl.GLGridItem()
w.addItem(g)

pos3 = np.zeros((720*1280,3))
sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.01, pxMode=False)

w.addItem(sp3)

files = sorted(glob.glob('cloud_*.npy'))

i = 0
def update():
    global sp3, pos3, i

    file = files[i % len(files)]
    p = np.load(files[i % len(files)])
    i += 1

    p_list = p.reshape((1280*720, 3))
    p_list = p_list[p_list[:, 2] > 0.001]

    # Swap some axis to make the visualization nicer
    pos3[:p_list.shape[0], 0] = p_list[:, 0]
    pos3[:p_list.shape[0], 1] = p_list[:, 2]
    pos3[:p_list.shape[0], 2] = -p_list[:, 1]
    pos3[p_list.shape[0]:, ...] = 0

    sp3.setData(pos=pos3)
    
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)

if __name__ == '__main__':
    pg.exec()
