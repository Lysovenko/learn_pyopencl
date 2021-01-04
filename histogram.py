from pyopencl import (CommandQueue, mem_flags as mf, Buffer, Program,
                      enqueue_copy)
from numpy import zeros, int32
from misc import create_some_context, load_cl_text, lena
import matplotlib.pyplot as plt
try:
    from time import process_time as perf_counter
except ImportError:
    from time import perf_counter


TIMES = {}
ctx = create_some_context()
lenar = lena().astype(int32).flatten()
len_buf = Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lenar)
histogram = zeros(256, dtype=int32)
h_buf = Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=histogram)
pt = perf_counter()
prg = Program(ctx, load_cl_text("histogram.cl")).build()
TIMES["Compilation"] = perf_counter() - pt
pt = perf_counter()
with CommandQueue(ctx) as queue:
    prg.histogram(queue, lenar.shape, None,
                  len_buf, int32(len(lenar)), h_buf)
    TIMES["Execution"] = perf_counter() - pt
    pt = perf_counter()
    enqueue_copy(queue, histogram, h_buf)
    TIMES["Copying"] = perf_counter() - pt
h_buf.release()
len_buf.release()
print("\n".join("%s:\t%g" % i for i in TIMES.items()))
plt.plot(histogram, ",")
plt.show()
