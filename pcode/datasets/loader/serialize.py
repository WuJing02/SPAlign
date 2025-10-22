
import os

__all__ = ["loads", "dumps"]

def create_dummy_func(func, dependency):
    
    if isinstance(dependency, (list, tuple)):
        dependency = ",".join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError(
            "Cannot import '{}', therefore '{}' is not available".format(
                dependency, func
            )
        )

    return _dummy

def dumps_msgpack(obj):
    
    return msgpack.dumps(obj, use_bin_type=True)

def loads_msgpack(buf):
    
    return msgpack.loads(buf, raw=False)

def dumps_pyarrow(obj):
    
    return pa.serialize(obj).to_buffer()

def loads_pyarrow(buf):
    
    return pa.deserialize(buf)

try:
    import pyarrow as pa
except ImportError:
    pa = None
    dumps_pyarrow = create_dummy_func("dumps_pyarrow", ["pyarrow"])
    loads_pyarrow = create_dummy_func("loads_pyarrow", ["pyarrow"])

try:
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
except ImportError:
    assert pa is not None, "pyarrow is a dependency of tensorpack!"
    loads_msgpack = create_dummy_func(
        "loads_msgpack", ["msgpack", "msgpack_numpy"]
    )
    dumps_msgpack = create_dummy_func(
        "dumps_msgpack", ["msgpack", "msgpack_numpy"]
    )

if os.environ.get("TENSORPACK_SERIALIZE", "msgpack") == "msgpack":
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow
