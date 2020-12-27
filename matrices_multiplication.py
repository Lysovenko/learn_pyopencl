import pyopencl as cl
import numpy as np
from misc import create_some_context, load_cl_text
from time import time

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

(n, m, p) = (3, 4, 5)

a = np.random.randint(2, size=(n*m)).astype(np.float32)
b = np.random.randint(2, size=(m*p)).astype(np.float32)
c = np.zeros((n*p), dtype=np.float32)

ctx = create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.Buffer\
   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer\
   (ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
prg = cl.Program(ctx, load_cl_text("multiply_matr.cl")).build()
st = time()
prg.multiply(queue, c.shape, None,
             np.uint16(n), np.uint16(m), np.uint16(p),
             a_buf, b_buf, c_buf)
a_mul_b = np.empty_like(c)
cl.enqueue_copy(queue, a_mul_b, c_buf)
elapsed = time() - st

print("matrix A:")
print(a.reshape(n, m))
print("matrix B:")
print(b.reshape(m, p))
print("multiplied A*B:")
print(a_mul_b.reshape(n, p))
print(elapsed)
