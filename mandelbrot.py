"""
Mandelbrot's set example
"""
from pyopencl import (CommandQueue, mem_flags as MEM, Program,
                      enqueue_copy,
                      Image, ImageFormat, channel_type as CHANNEL,
                      channel_order as CHO)
from numpy import zeros, uint8, int32, float64
from misc import create_some_context, load_cl_text, show_img
try:
    from time import process_time as perf_counter
except ImportError:
    from time import perf_counter


def find_set(x0, y0, dx, dy, w, h, ctx, buf, prg):
    """Find Mandelbrot set"""
    times = {}
    pt = perf_counter()
    with CommandQueue(ctx) as queue:
        prg.mandelbrot(queue, (w, h), None, buf, int32(w), int32(h),
                       float64(x0), float64(y0), float64(dx), float64(dy))
        times["Execution"] = perf_counter() - pt
        pt = perf_counter()
        dest = zeros((h, w, 4), dtype=uint8)
        enqueue_copy(queue, dest, buf, origin=(0, 0), region=(w, h))
        times["Copying"] = perf_counter() - pt
    print("\n".join("%s:\t%g" % i for i in times.items()))
    return dest


class Interactor:
    def __init__(self):
        self.h = self.w = 800
        self.dx = self.dy = 3. / 800.
        self.x0 = -2
        self.y0 = 1.5
        ctx = create_some_context()
        fmt = ImageFormat(CHO.RGBA, CHANNEL.UNSIGNED_INT8)
        self.buf = Image(ctx, MEM.WRITE_ONLY, fmt, shape=(self.w, self.h))
        prg = Program(ctx, load_cl_text("mandelbrot.cl")).build()
        self.params = (self.w, self.h, ctx, self.buf, prg)

    def __call__(self, key=None):
        if key == "Key_UP":
            self.y0 += self.h * self.dy / 10.
        elif key == "Key_Down":
            self.y0 -= self.h * self.dy / 10.
        elif key == "Key_Left":
            self.x0 -= self.w * self.dx / 10.
        elif key == "Key_Right":
            self.x0 += self.w * self.dx / 10.
        elif key == "Key_PageUp":
            dx1 = self.dx
            dy1 = self.dy
            self.dx /= 4. / 3.
            self.dy /= 4. / 3.
            self.x0 += self.w / 2. * (dx1 - self.dx)
            self.y0 += self.h / 2. * (self.dy - dy1)
        elif key == "Key_PageDown":
            dx1 = self.dx
            dy1 = self.dy
            self.dx *= 4. / 3.
            self.dy *= 4. / 3.
            self.x0 += self.w / 2. * (dx1 - self.dx)
            self.y0 += self.h / 2. * (self.dy - dy1)
        extent = (self.x0, self.x0 + self.w * self.dx,
                  self.y0 - self.h * self.dy, self.y0)
        return {"img": find_set(self.x0, self.y0, self.dx, self.dy,
                                *self.params)[:, :, :3],
                "extent": extent}

    def __del__(self):
        self.buf.release()


if __name__ == "__main__":
    from sys import argv
    try:
        xm, ym, xw, w, h = map(float, argv[1:])
        h = int32(h)
        w = int32(w)
        dx = dy = xw / w
        x0 = xm - xw / 2.
        y0 = ym + xw * (h / w) / 2.
    except (IndexError, ValueError):
        h = w = int32(800)
        x0, y0 = -2., 1.5
        dx = dy = 3. / 800
    ctx = create_some_context()
    fmt = ImageFormat(CHO.RGBA, CHANNEL.UNSIGNED_INT8)
    buf = Image(ctx, MEM.WRITE_ONLY, fmt, shape=(w, h))
    prg = Program(ctx, load_cl_text("mandelbrot.cl")).build()
    res = find_set(x0, y0, dx, dy, w, h, ctx, buf, prg)
    buf.release()
    show_img(res[:, :, :3], True)
