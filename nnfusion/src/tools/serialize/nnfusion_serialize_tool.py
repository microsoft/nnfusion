"""A python tool to parse *.pb file of nnfusion serialization.
TODO(gbxu): merge this into nnfusion adapter.
"""
import graph_def_pb2
import sys

def import_nnfusion():
    """Read the existing nnfusion graph.
    """
    graph_def = graph_def_pb2.GraphDef()
    try:
        with open(sys.argv[1], "rb") as f:
            graph_def.ParseFromString(f.read())
    except IOError:
        print(sys.argv[1] + ": File not found.  Creating a new file.")
    with open(sys.argv[2], "w+") as f:
        f.write(str(graph_def))
    return graph_def

def export_nnfusion(graph_def):
    """Write the new nnfusion graph back to disk.
    """
    with open(sys.argv[1], "wb") as f:
        f.write(graph_def.SerializeToString())

if __name__ == "__main__":
    import_nnfusion()
