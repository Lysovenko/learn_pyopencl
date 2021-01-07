"""
Mandelbrot's set example
"""
from sys import argv
from pyopencl import (CommandQueue, mem_flags as MEM, Program,
                      enqueue_copy,
                      Image, ImageFormat, channel_type as CHANNEL,
                      channel_order as CHO)
from numpy import zeros, uint8, int32, float32, pi
from misc import create_some_context, load_cl_text, show_img
try:
    from time import process_time as perf_counter
except ImportError:
    from time import perf_counter


TIMES = {}
try:
    angle = -float(argv[1]) / 180. * pi
except (IndexError, ValueError):
    h = w = int32(800)
    x0, y0 = -2., 1.5
    dx = dy = 3. / 800
ctx = create_some_context()

fmt = ImageFormat(CHO.RGBA, CHANNEL.UNSIGNED_INT8)
out_img_buf = Image(ctx, MEM.WRITE_ONLY, fmt, shape=(w, h))
pt = perf_counter()
prg = Program(ctx, load_cl_text("mandelbrot.cl")).build()
TIMES["Compilation"] = perf_counter() - pt
pt = perf_counter()
with CommandQueue(ctx) as queue:
    prg.mandelbrot(queue, (w, h), None,
                   out_img_buf,
                   w, h, float32(x0), float32(y0), float32(dx), float32(dy))
    TIMES["Execution"] = perf_counter() - pt
    pt = perf_counter()
    dest = zeros((h, w, 4), dtype=uint8)
    enqueue_copy(queue, dest, out_img_buf, origin=(0, 0), region=(w, h))
    TIMES["Copying"] = perf_counter() - pt
out_img_buf.release()
print("\n".join("%s:\t%g" % i for i in TIMES.items()))
show_img(dest[:, :, :3], True)
