import os
import sys
from math import asin, cos, pi, sqrt

import bezier
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection

from utils import clean_str

SAVE_FIG = False
DEBUG = True

def get_graph_metrics(graph):
    graph_degree = dict(graph.degree)
    print("Graph Summary:")
    print(f"Number of nodes : {len(graph.nodes)}")
    print(f"Number of edges : {len(graph.edges)}")
    print(f"Maximum degree : {np.max(list(graph_degree.values()))}")
    print(f"Minimum degree : {np.min(list(graph_degree.values()))}")
    print(f"Average degree : {np.mean(list(graph_degree.values()))}")
    print(f"Median degree : {np.median(list(graph_degree.values()))}")
    print("")
    print("Graph Connectivity")
    try:
        print(f"Connected Components : {nx.number_connected_components(graph)}")
    except:
        print(
            f"Strongly Connected Components : {nx.number_strongly_connected_components(graph)}"
        )
        print(
            f"Weakly Connected Components : {nx.number_weakly_connected_components(graph)}"
        )
    print("")
    print("Graph Distance")
    print(f"Average Distance : {nx.average_shortest_path_length(graph)}")
    print(f"Diameter : {nx.algorithms.distance_measures.diameter(graph)}")
    print("")
    print("Graph Clustering")
    print(f"Transitivity : {nx.transitivity(graph)}")
    print(f"Average Clustering Coefficient : {nx.average_clustering(graph)}")

    return None


def distance(lat1, lon1, lat2, lon2):
    p = pi / 180
    a = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(a))


def calculate_distance(input_network):
    """
    Add weights to the edges of a network based on the degrees of the connecting
    vertices, and return the network.
    Args:
        input_network: A NetworkX graph object
    Returns:
        G: A weighted NetworkX graph object.
    """
    G = input_network.copy()

    # Add weights to edges
    for node, successor in G.edges():
        dist = distance(
            G.nodes[node]["lat"],
            G.nodes[node]["long"],
            G.nodes[successor]["lat"],
            G.nodes[successor]["long"],
        )
        edge_dict = {(node, successor): dist}
        nx.set_edge_attributes(G, edge_dict, "distance")

    return G

def add_loc(G, long_attr = "long", lat_attr = "lat"):
    '''Add loc(longitude, latitude) attribute.'''
    for node in G.nodes():
        loc = (G.nodes[node][long_attr], G.nodes[node][lat_attr])
        node_dict = {node: loc}
        nx.set_node_attributes(G, node_dict, "pos")
    
    return G


def create_network(path):
    """
    Create a NetworkX graph object using the airport and route databases.
    Args:
        nodes: The file path to the nodes .csv file.
        edges: The file path to the edges .csv file.
    Returns:
        G: A NetworkX DiGraph object populated with the nodes and edges assigned
           by the data files from the arguments.
    """
    if DEBUG: print("Creating network.")
    G = nx.read_gexf(path)

    # Add pos attribute in nodes
    G = add_loc(G, long_attr = "long", lat_attr = "lat")

    # Calculate the edge weights
    if DEBUG: print("\tCalculating edge weights", end="")
    degree_network = nx.Graph(G)
    ldegree = degree_network.degree
    for i, j in G.edges():
        degree_sum = ldegree[i] + ldegree[j]
        G[i][j]["weight"] = degree_sum

    # Clean nodes names
    G = nx.relabel_nodes(G, clean_str)

    # Calculate the edge distances
    if DEBUG: print("\tCalculating edge distance", end="")
    G = calculate_distance(G)

    # Add clustering data
    if DEBUG: print("\tCalculating clustering coefficents", end="")
    cluster_network = nx.Graph(G)
    lcluster = nx.clustering(cluster_network)
    for i, j in G.edges():
        cluster_sum = lcluster[i] + lcluster[j]
        G[i][j]["cluster"] = cluster_sum

    return G

def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity="random"):
    # Get nodes into np array
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == "random":
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l) == 0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(
            np.mod(np.vectorize(hash)(edges[:, 0]) + np.vectorize(hash)(edges[:, 1]), 2)
            == 0,
            -1,
            1,
        )

    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse=True)
    coords = np.array([pos[x] for x in u])[inv].reshape(
        [edges.shape[0], 2, edges.shape[1]]
    )
    coords_node1 = coords[:, 0, :]
    coords_node2 = coords[:, 1, :]

    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:, 0] > coords_node2[:, 0]
    coords_node1[should_swap], coords_node2[should_swap] = (
        coords_node2[should_swap],
        coords_node1[should_swap],
    )

    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1 - coords_node2) ** 2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:, 1] - coords_node1[:, 1]) / (
        coords_node2[:, 0] - coords_node1[:, 0]
    )
    m2 = -1 / m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist / np.sqrt(1 + m1**2)
    v1 = np.array([np.ones(l), m1])
    coords_node1_displace = coords_node1 + (v1 * t1).T
    coords_node2_displace = coords_node2 - (v1 * t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist / np.sqrt(1 + m2**2)
    v2 = np.array([np.ones(len(edges)), m2])
    coords_node1_ctrl = coords_node1_displace + (rnd * v2 * t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd * v2 * t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array(
        [coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2]
    )

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:, i, :].T
        curveplots.append(
            bezier.Curve.from_nodes(nodes).evaluate_multi(np.linspace(0, 1, bezier_precision))
            .T
        )

    # Return an array of these curves
    curves = np.array(curveplots)

    return curves


def visualize_curved_edges(network, title, pos, curves):
    """
    Visualize the network given an array of posisitons.
    """
    print("-- Starting to Visualize --")

    colors = []
    alphas = []
    edge_colors = []
    for node in network.nodes():
        colors.append(network.nodes[node]["color"])
        alphas.append(network.nodes[node]["alpha"])
    for edge in network.edges(data=True):
        edge_colors.append(network[edge[0]][edge[1]]["color"])

    linestyle_list = [
        "dotted" if x[2]["aereo"] == "sim" else "solid"
        for x in list(network.edges(data=True))
    ]
    positions = nx.get_node_attributes(network, "pos")
    lc = LineCollection(curves, color=edge_colors, alpha=0.1, linestyle=linestyle_list)
    # lc = LineCollection(curves, color=edge_colors, alpha=0.1)
    # capital = nx.get_node_attributes(network, 'capital')
    # sizes = [100 if x == 'sim' else 15 for x in list(capital.values())]
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(
        network, positions, node_size=5, alpha=alphas, node_color=colors, ax=ax
    )
    plt.gca().add_collection(lc)
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    number_files = str(len(os.listdir()))
    while len(number_files) < 3:
        number_files = "0" + number_files
    if SAVE_FIG:
        plt.savefig(
        "infection_real-{0}.png".format(number_files), bbox_inches="tight", dpi=600
        )
    plt.show()


def infection_yes_no(input_network, cases_bz, vis=False, title=""):

    print("Replicating infection.")

    network = input_network.copy()
    positions = nx.get_node_attributes(network, "pos")
    curves = curved_edges(network, positions)

    # Set the default to susceptible
    sys.stdout.flush()
    for node in network.nodes():
        nx.set_node_attributes(network, values={node: "s"}, name="status")
        nx.set_node_attributes(network, values={node: "#C1CDCD"}, name="color")
        nx.set_node_attributes(network, values={node: 0.4}, name="alpha")
    for edge in network.edges(data=True):
        nx.set_edge_attributes(network, {(edge[0], edge[1]): "#C1CDCD"}, "color")
    if vis:
        pos = nx.get_node_attributes(network, "pos")

    epi_weeks = sorted(list(set(cases_bz.epidemiological_week.tolist())))

    palette = sns.color_palette("flare")

    for week in epi_weeks:
        susceptible, infected = 0, 0
        weekDf = cases_bz[cases_bz.epidemiological_week == week]
        cities = list(set(weekDf.city_state.tolist()))
        for city in cities:
            nx.set_node_attributes(network, {city: "i"}, "status")
            nx.set_node_attributes(network, {city: palette[5]}, "color")
            # nx.set_node_attributes(network, {city: 0.8}, 'alpha')
        for edge in network.edges(data=True):
            if (
                network.nodes[edge[0]]["status"] == "i"
                and network.nodes[edge[1]]["status"] == "i"
            ):
                nx.set_edge_attributes(
                    network, {(edge[0], edge[1]): "#C1CDCD"}, "color"
                )

        for node in network.nodes():
            status = network.nodes[node]["status"]
            if status == "s":
                susceptible += 1
            elif status == "i":
                infected += 1
        print(
            "Week {0}, S: {1}, I: {2}".format(
                week, susceptible / len(network.nodes()), infected / len(network.nodes())
            )
        )

        if susceptible == 0:
            break

        if vis:
            visualize_curved_edges(network, title, pos, curves)

    print(
        "\t----------\n\tS: {0}, I: {1}, {2}".format(
            susceptible, infected, infected / len(network.nodes())
        )
    )

    return {"Susceptible": susceptible, "Infected": infected}


def infection_cases(
    input_network,
    cases_bz,
    DELAY=0,
    vis=False,
    file_name="sir.csv",
    title="",
    RECALCULATE=True,
):
    print("Replicating infection.")

    network = input_network.copy()
    positions = nx.get_node_attributes(network, "pos")
    curves = curved_edges(network, positions)

    # Set the default to susceptable
    sys.stdout.flush()
    for node in network.nodes():
        nx.set_node_attributes(network, values={node: "s"}, name="status")
        nx.set_node_attributes(network, values={node: "#C1CDCD"}, name="color")
        nx.set_node_attributes(network, values={node: 0.4}, name="alpha")
    for edge in network.edges(data=True):
        nx.set_edge_attributes(network, {(edge[0], edge[1]): "#C1CDCD"}, "color")

    if vis:
        pos = nx.get_node_attributes(network, "pos")

    epi_weeks = sorted(list(set(cases_bz.epidemiological_week.tolist())))
    palette = sns.color_palette("flare")
    # Iterate through the evolution of the disease.
    i = 0
    for week in epi_weeks:
        i = i + 1
        print(i)
        # Create variables to hold the outcomes as they happen
        S, queda, estab, aumento = 0, 0, 0, 0

        casosWeekDf = cases_bz[cases_bz.epidemiological_week == week]
        cities = list(set(casosWeekDf.city_state.tolist()))

        for city in cities:
            new_per = casosWeekDf[
                casosWeekDf.city_state == city
            ].new_deaths_percent.values[0]
            if new_per < -0.01:
                nx.set_node_attributes(network, {city: "em queda"}, "status")
                nx.set_node_attributes(network, {city: palette[4]}, "color")
                # nx.set_node_attributes(network, {city: 0.7}, 'alpha')

            elif new_per < 0.10:
                nx.set_node_attributes(network, {city: "estabilidade"}, "status")
                nx.set_node_attributes(network, {city: palette[2]}, "color")
                # nx.set_node_attributes(network, {city: 0.7}, 'alpha')

            else:
                nx.set_node_attributes(network, {city: "aumento"}, "status")
                nx.set_node_attributes(network, {city: palette[0]}, "color")
                # nx.set_node_attributes(network, {city: 0.8}, 'alpha')

        # Loop twice to prevent bias.
        for edge in network.edges(data=True):
            if (
                network.nodes[edge[0]]["status"] == "i"
                and network.nodes[edge[1]]["status"] == "i"
            ):
                nx.set_edge_attributes(
                    network,
                    {(edge[0], edge[1]): (0.888292, 0.40830288, 0.36223756)},
                    "color",
                )

        for node in network.nodes():
            status = network.nodes[node]["status"]
            color = network.nodes[node]["color"]

            if status == "s":
                S += 1
            if status == "em queda":
                queda += 1
            if status == "estabilidade":
                estab += 1
            elif status == "aumento":
                aumento += 1

        print(
            "Semana: {0},Suscetivel: {1},Em queda: {2},Estabilidade: {3},Aumento: {4}".format(
                week, S, queda, estab, aumento
            )
        )

        if vis:
            visualize_curved_edges(network, title, pos, curves)

    print("\t----------\n\tS: {0}, I: {1}, R: {2}".format(S, queda, aumento))

    return {"Suscceptable": S, "Infected": queda, "Recovered": aumento}


def visualize(network, title, pos):
    """
    Visualize the network given an array of posisitons.
    """
    print("-- Starting to Visualize --")

    colors = []
    i_edge_colors = []
    d_edge_colors = []
    default = []
    infected = []
    for node in network.nodes():
        colors.append(network.nodes[node]["color"])
    for i, j in network.edges():
        color = network.nodes[i]["color"]
        alpha = 0.75
        if color == (0.42355299, 0.16934709, 0.42581586) or color == (
            0.48942421,
            0.72854938,
            0.56751036,
        ):
            color = "#A6A6A6"
            default.append((i, j))
            d_edge_colors.append(color)
        else:
            color = "#A6A6A6"  # 29A229"
            infected.append((i, j))
            i_edge_colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Fist pass - Gray lines
    nx.draw_networkx_edges(
        network,
        pos,
        connectionstyle="arc3,rad=0.9",
        edgelist=default,
        width=0.5,
        edge_color=d_edge_colors,
        alpha=0.5,
        arrows=False,
        ax=ax,
    )

    # Second Pass - Colored lines
    nx.draw_networkx_edges(
        network,
        pos,
        connectionstyle="arc3,rad=0.9",
        edgelist=infected,
        width=0.5,
        edge_color=i_edge_colors,
        alpha=0.75,
        arrows=False,
        ax=ax,
    )

    positions = nx.get_node_attributes(network, "pos")
    nx.draw_networkx_nodes(
        network,
        pos,
        linewidths=0.5,
        node_size=15,
        alpha=0.5,
        # with_labels=False,
        node_color=colors,
        ax=ax,
    )

    # Adjust the plot limits
    cut = 1.05
    xmax = cut * max(xx for xx, yy in pos.values())
    xmin = min(xx for xx, yy in pos.values())
    xmin = xmin - (cut * xmin)

    ymax = cut * max(yy for xx, yy in pos.values())
    ymin = (cut) * min(yy for xx, yy in pos.values())
    ymin = ymin - (cut * ymin)

    number_files = str(len(os.listdir()))
    while len(number_files) < 3:
        number_files = "0" + number_files

    plt.show()
    plt.close()
