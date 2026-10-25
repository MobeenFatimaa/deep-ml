def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.

    Args:
        obs_space: dict with 'type' and type-specific keys defining the observation space
        act_space: dict with 'type' and type-specific keys defining the action space
        hidden_layers: list of ints specifying hidden layer widths

    Returns:
        dict with keys:
          'input_dim': int
          'output_dim': int
          'layer_shapes': list of ((weight_rows, weight_cols), (bias_dim,)) tuples
          'total_params': int
    """
    def get_dim(space):
        stype = space.get('type')
        if stype == 'Box':
            shape = space.get('shape')
            dim = 1
            for s in shape:
                dim *= s
            return dim
        elif stype in ('Discrete', 'MultiBinary'):
            return space.get('n')
        else:
            raise ValueError(f"Unknown space type: {stype}")

    input_dim = get_dim(obs_space)
    output_dim = get_dim(act_space)

    dims = [input_dim] + list(hidden_layers) + [output_dim]
    
    layer_shapes = []
    total_params = 0

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        w_shape = (in_dim, out_dim)
        b_shape = (out_dim,)
        layer_shapes.append((w_shape, b_shape))
        total_params += in_dim * out_dim + out_dim

    return {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'layer_shapes': layer_shapes,
        'total_params': total_params
    }