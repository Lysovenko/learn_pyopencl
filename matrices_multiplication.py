import pyopencl as cl
import numpy as np
from misc import create_some_context, load_cl_text
from sys import argv
try:
    from time import process_time as perf_counter
except ImportError:
    from time import perf_counter


try:
    n, m, p = map(int, argv[1:])
except ValueError:
    (n, m, p) = (3, 4, 5)

a = np.random.randint(2, size=(n*m)).astype(np.float32)
b = np.random.randint(2, size=(m*p)).astype(np.float32)
c = np.zeros((n*p), dtype=np.float32)
TIMES = {}
ctx = create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
pt = perf_counter()
prg = cl.Program(ctx, load_cl_text("multiply_matr.cl")).build()
TIMES["Compilation"] = perf_counter() - pt
pt = perf_counter()
prg.multiply(queue, c.shape, None,
             np.uint16(n), np.uint16(m), np.uint16(p),
             a_buf, b_buf, c_buf)
TIMES["Execution"] = perf_counter() - pt
a_mul_b = np.empty_like(c)
pt = perf_counter()
cl.enqueue_copy(queue, a_mul_b, c_buf)
TIMES["Copying"] = perf_counter() - pt

print("matrix A:")
print(a.reshape(n, m))
print("matrix B:")
print(b.reshape(m, p))
print("multiplied A*B:")
print(a_mul_b.reshape(n, p))
print("\n".join("%s:\t%g" % i for i in TIMES.items()))
