import pyopencl as cl
from os.path import dirname, join


def create_some_context():
    """Create some context"""
    platform = cl.get_platforms()
    if not platform:
        raise RuntimeError("No OpenCL platforms found")
    print("Used OpenCL platform:", platform[0].name)
    devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    if not devices:
        devices = platform[0].get_devices(device_type=cl.device_type.CPU)
    if not devices:
        raise RuntimeError("No OpenCL devices found")
    print("Devices:", ", ".join(i.name for i in devices))
    ctx = cl.Context(devices=devices)
    return ctx


def load_cl_text(fname):
    with open(join(dirname(__file__), fname)) as fp:
        return fp.read()


def lena():
    import pickle
    from os.path import join, dirname
    from numpy import array, float32
    fname = join(dirname(__file__), "lena.dat")
    with open(fname, "rb") as f:
        lena = array(pickle.load(f))
    return lena


def show_img(img, colored=False):
    import matplotlib.pyplot as plt
    if not colored:
        plt.gray()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    show_img(lena())
