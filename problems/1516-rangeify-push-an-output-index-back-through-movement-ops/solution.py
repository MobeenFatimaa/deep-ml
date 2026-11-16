import numpy as np


def get_strides(shape):
    """Compute row-major strides for a given shape."""
    if not shape:
        return []
    strides = []
    s = 1
    for d in reversed(shape):
        strides.insert(0, s)
        s *= d
    return strides


def shape_after(shape0, ops):
    """Calculate the resulting shape after applying a sequence of movement ops."""
    shape = tuple(shape0)
    for op_type, arg in ops:
        if op_type == "reshape":
            shape = tuple(arg)
        elif op_type == "expand":
            shape = tuple(arg)
        elif op_type == "permute":
            shape = tuple(shape[p] for p in arg)
        elif op_type == "flip":
            shape = shape
        elif op_type == "pad":
            shape = tuple(s + p[0] + p[1] for s, p in zip(shape, arg))
        elif op_type == "shrink":
            shape = tuple(r[1] - r[0] for r in arg)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    return shape


def source_index(shape0, ops, idx):
    """
    Walks an output index backwards through the ops to find the source buffer 
    flat offset and validity flag.
    """
    # Keep track of intermediate shapes
    shapes = [tuple(shape0)]
    for op_type, arg in ops:
        shapes.append(shape_after(shapes[-1], [(op_type, arg)]))
        
    curr_idx = list(idx)
    valid = True
    
    # Walk backwards from last op to first
    for k in range(len(ops) - 1, -1, -1):
        op_type, arg = ops[k]
        prev_shape = shapes[k]
        curr_shape = shapes[k + 1]
        
        if op_type == "reshape":
            curr_strides = get_strides(curr_shape)
            flat = sum(c * s for c, s in zip(curr_idx, curr_strides))
            prev_strides = get_strides(prev_shape)
            curr_idx = [(flat // s) % sz for sz, s in zip(prev_shape, prev_strides)]
            
        elif op_type == "expand":
            new_idx = []
            for p_sz, c_sz, c_idx in zip(prev_shape, curr_shape, curr_idx):
                if p_sz == 1 and c_sz > 1:
                    new_idx.append(0)
                else:
                    new_idx.append(c_idx)
            curr_idx = new_idx
            
        elif op_type == "permute":
            new_idx = [0] * len(prev_shape)
            for i, p in enumerate(arg):
                new_idx[p] = curr_idx[i]
            curr_idx = new_idx
            
        elif op_type == "flip":
            new_idx = list(curr_idx)
            for axis in arg:
                sz = prev_shape[axis]
                new_idx[axis] = sz - 1 - curr_idx[axis]
            curr_idx = new_idx
            
        elif op_type == "pad":
            new_idx = []
            for axis, (p_low, p_high) in enumerate(arg):
                c_idx = curr_idx[axis]
                p_coord = c_idx - p_low
                if p_coord < 0 or p_coord >= prev_shape[axis]:
                    valid = False
                new_idx.append(p_coord)
            curr_idx = new_idx
            
        elif op_type == "shrink":
            new_idx = []
            for axis, (start, end) in enumerate(arg):
                c_idx = curr_idx[axis]
                p_coord = c_idx + start
                new_idx.append(p_coord)
            curr_idx = new_idx
            
    # Compute row-major flat offset into the original buffer
    orig_strides = get_strides(shape0)
    flat_offset = sum(c * s for c, s in zip(curr_idx, orig_strides))
    return (flat_offset, valid)


def materialize(shape0, ops, buf):
    """Build the output array by querying source_index for every output index."""
    final_shape = shape_after(shape0, ops)
    out = np.zeros(final_shape, dtype=float)
    buf_arr = np.asarray(buf)
    
    it = np.nditer(out, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        flat, valid = source_index(shape0, ops, idx)
        if valid:
            out[idx] = buf_arr.flat[flat]
        else:
            out[idx] = 0.0
        it.iternext()
    return out