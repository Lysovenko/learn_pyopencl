"""
Image rotation example
"""
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


def rotate(angle, ctx, in_img_buf, out_img_buf, h, w, prg):
    times = {}
    pt = perf_counter()
    with CommandQueue(ctx) as queue:
        prg.rotate_(queue, (w, h), None,
                    in_img_buf,
                    out_img_buf,
                    w, h,
                    float32(angle))
        times["Execution"] = perf_counter() - pt
        pt = perf_counter()
        dest = zeros((w, h, 4), dtype=uint8)
        enqueue_copy(queue, dest, out_img_buf, origin=(0, 0), region=(w, h))
        times["Copying"] = perf_counter() - pt
    print("\n".join("%s:\t%g" % i for i in times.items()))
    return dest


class Interactor:
    def __init__(self):
        self.angle = 0.
        self.ch_angles = {"Key_UP": pi / 18., "Key_Down": -pi / 18.,
                          "Key_Right": -pi / 180., "Key_Left": pi / 180.}
        ctx = create_some_context()
        in_img = lena()
        h, w = map(int32, in_img.shape[:2])
        # in pyopencl 2018.2.2 channel orders other than RGBA
        # cause segmentation fault
        i4 = zeros((h, w, 4), dtype=uint8)
        i4[:, :, 0] = in_img
        self.in_img_buf = image_from_array(ctx, i4, 4)
        fmt = ImageFormat(CHO.RGBA, CHANNEL.UNSIGNED_INT8)
        self.out_img_buf = Image(ctx, MEM.WRITE_ONLY, fmt, shape=(w, h))
        prg = Program(ctx, load_cl_text("rotation.cl")).build()
        self.params = (ctx, self.in_img_buf, self.out_img_buf, h, w, prg)

    def __call__(self, key):
        self.angle += self.ch_angles.get(key, 0.)
        return {"img": rotate(self.angle, *self.params)[:, :, 0]}

    def __del__(self):
        print("delete rotate interactor")
        self.in_img_buf.release()
        self.out_img_buf.release()


if __name__ == "__main__":
    from sys import argv
    try:
        angle = -float(argv[1]) / 180. * pi
    except (IndexError, ValueError):
        angle = pi / 4
    ctx = create_some_context()
    in_img = lena()
    h, w = map(int32, in_img.shape[:2])
    # in pyopencl 2018.2.2 channel orders other than RGBA
    # cause segmentation fault
    i4 = zeros((h, w, 4), dtype=uint8)
    i4[:, :, 0] = in_img
    in_img_buf = image_from_array(ctx, i4, 4)
    fmt = ImageFormat(CHO.RGBA, CHANNEL.UNSIGNED_INT8)
    out_img_buf = Image(ctx, MEM.WRITE_ONLY, fmt, shape=(w, h))
    prg = Program(ctx, load_cl_text("rotation.cl")).build()
    res = rotate(angle, ctx, in_img_buf, out_img_buf, h, w, prg)
    in_img_buf.release()
    out_img_buf.release()
    show_img(res[:, :, 0])
