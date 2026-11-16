import ctypes
import os
import subprocess
import tempfile
import numpy as np


def get_strides(shape):
    if not shape:
        return []
    strides = []
    s = 1
    for d in reversed(shape):
        strides.insert(0, s)
        s *= d
    return strides


def render_reduce(shape, axes, op="sum", name="reduce"):
    """
    Renders a C function that performs a reduction over specified axes.
    """
    n = len(shape)
    kept_axes = [i for i in range(n) if i not in axes]
    reduced_axes = list(axes)
    
    in_strides = get_strides(shape)
    kept_shape = [shape[i] for i in kept_axes]
    kept_strides = get_strides(kept_shape)
    
    lines = []
    lines.append(f"void {name}(float* restrict out, const float* restrict in) {{")
    
    indent = 2
    
    # 1. Output loops (over kept axes in order)
    for axis in kept_axes:
        space = " " * indent
        lines.append(f"{space}for (int r{axis} = 0; r{axis} < {shape[axis]}; r{axis}++) {{")
        indent += 2
        
    # 2. Accumulator initialization
    space = " " * indent
    if op == "sum":
        lines.append(f"{space}float acc = 0.0f;")
    elif op == "max":
        lines.append(f"{space}float acc = -INFINITY;")
    else:
        raise ValueError(f"Unsupported reduction operator: {op}")
        
    # 3. Reduce loops (over axes in order)
    for axis in reduced_axes:
        space = " " * indent
        lines.append(f"{space}for (int r{axis} = 0; r{axis} < {shape[axis]}; r{axis}++) {{")
        indent += 2
        
    # 4. Update statement
    space = " " * indent
    in_terms = [f"r{i} * {in_strides[i]}" for i in range(n)]
    in_flat = " + ".join(in_terms)
    
    if op == "sum":
        lines.append(f"{space}acc = acc + in[{in_flat}];")
    elif op == "max":
        lines.append(f"{space}acc = fmaxf(acc, in[{in_flat}]);")
        
    # Close reduce loops
    for axis in reversed(reduced_axes):
        indent -= 2
        space = " " * indent
        lines.append(f"{space}}}")
        
    # 5. Output write
    space = " " * indent
    if kept_axes:
        kept_terms = [f"r{kept_axes[j]} * {kept_strides[j]}" for j in range(len(kept_axes))]
        out_flat = " + ".join(kept_terms)
    else:
        out_flat = "0"
    lines.append(f"{space}out[{out_flat}] = acc;")
    
    # Close output loops
    for axis in reversed(kept_axes):
        indent -= 2
        space = " " * indent
        lines.append(f"{space}}}")
        
    lines.append("}")
    return "\n".join(lines)


def run_reduce(shape, axes, x, op="sum"):
    """
    Renders the C code, compiles it into a shared library, and executes it on input array x.
    """
    code = render_reduce(shape, axes, op=op, name="reduce")
    full_code = "#include <math.h>\n\n" + code
    
    x = np.ascontiguousarray(x, dtype=np.float32)
    kept_shape = tuple(shape[i] for i in range(len(shape)) if i not in axes)
    out = np.zeros(kept_shape, dtype=np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "reduce.c")
        so_file = os.path.join(tmpdir, "reduce.so")
        
        with open(c_file, "w") as f:
            f.write(full_code)
            
        subprocess.run(
            ["cc", "-O2", "-shared", "-fPIC", "-w", c_file, "-o", so_file, "-lm"],
            check=True
        )
        
        lib = ctypes.CDLL(so_file)
        reduce_fn = lib.reduce
        reduce_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        reduce_fn.restype = None
        
        reduce_fn(out.ctypes.data_as(ctypes.c_void_p), x.ctypes.data_as(ctypes.c_void_p))
        
    return out