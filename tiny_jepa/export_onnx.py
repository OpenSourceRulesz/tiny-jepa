"""Export TinyJEPA sub-networks to ONNX.

Each exported graph is a plain MLP (Gemm + Relu chains), runnable in any
ONNX Runtime, in any language::

    from tiny_jepa import TinyJEPA
    from tiny_jepa.export_onnx import export

    model = TinyJEPA.load("tiny_jepa/checkpoints/orby_v1.npz")
    export(model, "tiny_jepa_onnx/")   # encoder.onnx, predictor.onnx, ...

Requires the optional ``onnx`` package: pip install 'tiny-jepa[onnx]'.
"""

import os

import numpy as np


def _mlp_graph(name, sizes, params, input_name, output_name):
    import onnx
    from onnx import helper, TensorProto

    nodes, inits, inputs, outputs = [], [], [], []
    cur = input_name
    for i in range(len(sizes) - 1):
        w_name, b_name = f"{name}.W{i}", f"{name}.b{i}"
        # Our numpy MLP does x @ W with W shaped (in, out); onnx Gemm with
        # transB=0 computes Y = A @ B, so W is used as-is.
        W = np.ascontiguousarray(params[f"{i}.W"])
        b = np.ascontiguousarray(params[f"{i}.b"])
        inits.append(helper.make_tensor(w_name, TensorProto.FLOAT,
                                       W.shape, W.ravel().tolist()))
        inits.append(helper.make_tensor(b_name, TensorProto.FLOAT,
                                       b.shape, b.ravel().tolist()))
        gemm_out = f"{name}.g{i}" if i < len(sizes) - 2 else output_name
        nodes.append(helper.make_node("Gemm", [cur, w_name, b_name],
                                      [gemm_out], name=f"{name}.gemm{i}"))
        if i < len(sizes) - 2:
            relu_out = f"{name}.r{i}"
            nodes.append(helper.make_node("Relu", [gemm_out], [relu_out],
                                          name=f"{name}.relu{i}"))
            cur = relu_out
    inputs.append(helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, ["batch", sizes[0]]))
    outputs.append(helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, ["batch", sizes[-1]]))
    graph = helper.make_graph(nodes, f"tiny_jepa_{name}", inputs, outputs, inits)
    opset = helper.make_operatorsetid("", 17)
    onnx_model = helper.make_model(graph, producer_name="tiny-jepa",
                                   opset_imports=[opset])
    onnx_model.ir_version = 9  # max compat with onnxruntime releases
    return onnx_model


EXPORTS = {
    # name: (input_name, output_name)
    "enc": ("frame_patches", "latent"),       # (B,1024) -> (B,32)
    "pred": ("latent_action", "next_latent"),  # (B,37)   -> (B,32)
    "reward": ("latent_action", "reward"),     # (B,37)   -> (B,1)
    "dec": ("latent", "frame_patches"),        # (B,32)   -> (B,1024)
    "pellet": ("latent", "pellet_xy"),         # (B,32)   -> (B,2) (x/32)
}


def export(model, out_dir):
    """Export encoder/predictor/reward/decoder/pellet-head to ONNX files."""
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ONNX export needs the optional 'onnx' package: "
            "pip install 'tiny-jepa[onnx]'") from e
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for net_name, (inp, outp) in EXPORTS.items():
        mlp = model.nets[net_name]
        onnx_model = _mlp_graph(net_name, mlp.sizes, mlp.parameters(), inp, outp)
        path = os.path.join(out_dir, f"{net_name}.onnx")
        import onnx as _onnx
        _onnx.save(onnx_model, path)
        paths[net_name] = path
    return paths
