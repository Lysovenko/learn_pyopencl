"""
Image rotation example
"""
from sys import argv
from pyopencl import (CommandQueue, mem_flags as MEM, Program,
                      enqueue_copy,
                      Image, ImageFormat, channel_type as CHANNEL,
                      channel_order as CHO,
                      image_from_array)
from numpy import zeros, uint8, int32, float32, pi
from misc import create_some_context, load_cl_text, lena, show_img
try:
    from time import process_time as perf_counter
except ImportError:
    from time import perf_counter


TIMES = {}
try:
    angle = -float(argv[1]) / 180. * pi
except (IndexError, ValueError):
    angle = pi / 4
ctx = create_some_context()
in_img = lena()
h, w = map(int32, in_img.shape[:2])
# in pyopencl 2018.2.2 channel orders other than RGBA cause segmentation fault
i4 = zeros((h, w, 4), dtype=uint8)
i4[:, :, 0] = in_img
in_img_buf = image_from_array(ctx, i4, 4)
fmt = ImageFormat(CHO.RGBA, CHANNEL.UNSIGNED_INT8)
out_img_buf = Image(ctx, MEM.WRITE_ONLY, fmt, shape=(w, h))
pt = perf_counter()
prg = Program(ctx, load_cl_text("rotation.cl")).build()
TIMES["Compilation"] = perf_counter() - pt
pt = perf_counter()
with CommandQueue(ctx) as queue:
    prg.rotate_(queue, (w, h), None,
                in_img_buf,
                out_img_buf,
                w, h,
                float32(angle))
    TIMES["Execution"] = perf_counter() - pt
    pt = perf_counter()
    dest = zeros(i4.shape, dtype=uint8)
    enqueue_copy(queue, dest, out_img_buf, origin=(0, 0), region=(w, h))
    TIMES["Copying"] = perf_counter() - pt
in_img_buf.release()
out_img_buf.release()
print("\n".join("%s:\t%g" % i for i in TIMES.items()))
show_img(dest[:, :, 0])
