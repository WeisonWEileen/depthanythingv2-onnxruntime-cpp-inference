import onnx
# from onnx import TensorProto
# from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
# from onnx.checker import check_model

# # inputs

# # 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
# X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
# A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
# B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])

# # outputs, the shape is left undefined

# Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

# # nodes

# # It creates a node defined by the operator type MatMul,
# # 'X', 'A' are the inputs of the node, 'XA' the output.
# node1 = make_node("MatMul", ["X", "A"], ["XA"])
# node2 = make_node("Add", ["XA", "B"], ["Y"])

# # from nodes to graph
# # the graph is built from the list of nodes, the list of inputs,
# # the list of outputs and a name.

# graph = make_graph(
#     [node1, node2], "lr", [X, A, B], [Y]  # nodes  # a name  # inputs
# )  # outputs

# # onnx graph
# # there is no metadata in this case.

# onnx_model = make_model(graph)
# onnx.save_model(onnx_model, "test.onnx")


# # Let's check the model is consistent,
# # this function is described in section
# # Checker and Shape Inference.
# check_model(onnx_model)

# # the work is done, let's display it...
# print(onnx_model)


from onnx import TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)


X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
node1 = make_node("MatMul", ["X", "A"], ["XA"])
node2 = make_node("Add", ["XA", "B"], ["Y"])
graph = make_graph([node1, node2], "lr", [X, A, B], [Y])
onnx_model = make_model(graph)
check_model(onnx_model)

# the list of inputs
print("** inputs **")
print(onnx_model.graph.input)

# in a more nicely format
print("** inputs **")
for obj in onnx_model.graph.input:
    print(
        "name=%r dtype=%r shape=%r"
        % (
            obj.name,
            obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape),
        )
    )

# the list of outputs
print("** outputs **")
print(onnx_model.graph.output)

# in a more nicely format
print("** outputs **")
for obj in onnx_model.graph.output:
    print(
        "name=%r dtype=%r shape=%r"
        % (
            obj.name,
            obj.type.tensor_type.elem_type,
            shape2tuple(obj.type.tensor_type.shape),
        )
    )

# the list of nodes
print("** nodes **")
print(onnx_model.graph.node)

# in a more nicely format
print("** nodes **")
for node in onnx_model.graph.node:
    print(
        "name=%r type=%r input=%r output=%r"
        % (node.name, node.op_type, node.input, node.output)
    )

onnx.save_model(onnx_model, "test.onnx")