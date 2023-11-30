import networkx as nx
import numpy as np

def adjacency(n, tri):
    A = np.zeros((n, n))

    for tr in tri:
        A[tr[0], tr[1]] = 1
        A[tr[1], tr[0]] = 1
        A[tr[0], tr[2]] = 1
        A[tr[2], tr[0]] = 1
        A[tr[1], tr[2]] = 1
        A[tr[2], tr[1]] = 1

    return A

def make_graph(A, c_types):
    G = nx.from_numpy_array(A)
    nx.set_node_attributes(G, {i: {'type': c} for i, c in enumerate(c_types)})
    return G

def remove_heterotypic_edges(G):
    heterotypic_edges = [(u, v) for u, v, data in G.edges(data=True) if G.nodes[u]['type'] != G.nodes[v]['type']]
    G.remove_edges_from(heterotypic_edges)

def compute_components(G):
    remove_heterotypic_edges(G)
    return nx.number_connected_components(G), nx.connected_components(G)

def cc_per_dt(c_types, tri_save):
    cc_arr = np.empty((tri_save.shape[0]))

    for t in range(tri_save.shape[0]):
        tri = tri_save[t]
        n = c_types.shape[0]
        A = adjacency(n, tri)
        G = make_graph(A, c_types)

        # connected components
        cc, _ = compute_components(G)
        cc_arr[t] = cc

    return cc_arr
