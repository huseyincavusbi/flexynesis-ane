"""
MIL (Model Intermediate Language) program generators for ANE kernels.

Dynamic weight approach: activations and weights are both packed into the
spatial dimension of a single input tensor [1, IC, 1, batch+OC].
No recompilation needed when weights change — only the IOSurface data changes.
"""

_MIL_HDR = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({'
    '{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, '
    '{"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    '{\n'
)


def dynamic_linear(ic: int, oc: int, batch: int) -> str:
    """
    Dynamic-weight linear layer: output = activation @ W

    Input tensor layout  [1, IC, 1, batch+OC]  fp16:
      spatial[0 : batch]        = activations  as [IC, batch]
      spatial[batch : batch+OC] = weights       as [IC, OC]

    Output tensor layout [1, OC, 1, batch]  fp16

    Used for both forward pass (y = x @ W) and backward dx pass
    (dx = grad_out @ W^T) by swapping the roles of IC/OC and passing W^T.
    """
    sp = batch + oc

    m  = _MIL_HDR
    m += f'    func main<ios18>(tensor<fp16, [1, {ic}, 1, {sp}]> x) {{\n'

    # Slice activations: [1, IC, 1, batch]
    m += f'        tensor<int32, [4]> ba = const()[name=string("ba"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    m += f'        tensor<int32, [4]> sa = const()[name=string("sa"), val=tensor<int32, [4]>([1,{ic},1,{batch}])];\n'
    m += f'        tensor<fp16, [1,{ic},1,{batch}]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string("act")];\n'

    # Slice weights: [1, IC, 1, OC]
    m += f'        tensor<int32, [4]> bw = const()[name=string("bw"), val=tensor<int32, [4]>([0,0,0,{batch}])];\n'
    m += f'        tensor<int32, [4]> sw = const()[name=string("sw"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];\n'
    m += f'        tensor<fp16, [1,{ic},1,{oc}]> W = slice_by_size(x=x,begin=bw,size=sw)[name=string("W")];\n'

    # Reshape act: [1,IC,1,batch] → [1,1,IC,batch] → transpose → [1,1,batch,IC]
    m += f'        tensor<int32, [4]> ra = const()[name=string("ra"), val=tensor<int32, [4]>([1,1,{ic},{batch}])];\n'
    m += f'        tensor<fp16, [1,1,{ic},{batch}]> a2 = reshape(shape=ra,x=act)[name=string("a2")];\n'
    m += f'        tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];\n'
    m += f'        tensor<fp16, [1,1,{batch},{ic}]> at = transpose(perm=pm,x=a2)[name=string("at")];\n'

    # Reshape W: [1,IC,1,OC] → [1,1,IC,OC]
    m += f'        tensor<int32, [4]> rw = const()[name=string("rw"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n'
    m += f'        tensor<fp16, [1,1,{ic},{oc}]> W2 = reshape(shape=rw,x=W)[name=string("W2")];\n'

    # Matmul: [1,1,batch,IC] @ [1,1,IC,OC] → [1,1,batch,OC]
    m += f'        bool bF = const()[name=string("bF"), val=bool(false)];\n'
    m += f'        tensor<fp16, [1,1,{batch},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=at,y=W2)[name=string("yh")];\n'

    # Transpose back: [1,1,batch,OC] → [1,1,OC,batch] → reshape → [1,OC,1,batch]
    m += f'        tensor<fp16, [1,1,{oc},{batch}]> yt = transpose(perm=pm,x=yh)[name=string("yt")];\n'
    m += f'        tensor<int32, [4]> ro = const()[name=string("ro"), val=tensor<int32, [4]>([1,{oc},1,{batch}])];\n'
    m += f'        tensor<fp16, [1,{oc},1,{batch}]> y = reshape(shape=ro,x=yt)[name=string("y")];\n'

    m += '    } -> (y);\n}\n'
    return m


def input_bytes(ic: int, oc: int, batch: int) -> int:
    """Byte size of the packed input IOSurface for dynamic_linear."""
    return ic * (batch + oc) * 2  # fp16 = 2 bytes


def output_bytes(oc: int, batch: int) -> int:
    """Byte size of the output IOSurface for dynamic_linear."""
    return oc * batch * 2  # fp16 = 2 bytes
