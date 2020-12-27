import pyopencl as cl


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
