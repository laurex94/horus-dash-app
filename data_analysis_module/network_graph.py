"""Module to handle Complex Networks Graph relations in ancient Greece.
    This module must create a graph given a set of nodes and edges
    The set of nodes and edges will be pandas data frames
    In this map_graph nodes are locations and edges are people traveling
    from one location to another

.. platform:: Unix, Windows, Mac
.. moduleauthor:: César Pernalete <pernalete.cg@gmail.com> and Elio Linárez <elinarezv@gmail.com>
"""

import igraph
import pyvis
import json
import copy
import pandas as pd
import numpy as np
from dataclasses import dataclass
from . import data_access as da
import random
import os
import tempfile
import pathlib


############
#####
##
# Source data configuration
##
#####
############

# In case local or url method define file names
GRAPH_TYPE = "communities"
TRAVELS_BLACK_LIST_FILE = "travels_blacklist.csv"
LOCATIONS_DATA_FILE = "locations_data.csv"
DATASET_NAME = "diogenes"
# These are the files already processed. Must be created "on the fly"
TRAVEL_EDGES_FILE = "travel_edges_graph.csv"
ALL_PLACES_FILE = "all_places_graph.csv"

DATA_ACCESS_METHOD = "url"
# In case url define BASE_URL
BASE_URL = "https://diogenet.ucsd.edu/data"

# This is referential these structures must be created departing from the root files
#  Create a dataframe from csv
#travel_edges = pd.read_csv("travel_edges_graph.csv", delimiter=",")
travel_edges = da.get_data_entity_local(TRAVEL_EDGES_FILE)
# all_places = pd.read_csv("all_places_graph.csv", delimiter=',')
# all_places = da.get_data_entity(ALL_PLACES_FILE, "local")
# User list comprehension to create a list of lists from Dataframe rows
list_of_rows_travel_edges = [list(row[1:]) for row in travel_edges.values]
# list_of_rows_all_places = [list(row[1:2]) for row in all_places.values]

VIRIDIS_COLORMAP = [
    (68, 1, 84),
    (72, 40, 120),
    (62, 74, 137),
    (49, 104, 142),
    (38, 130, 142),
    (31, 158, 137),
    (53, 183, 121),
    (109, 205, 89),
    (180, 222, 44),
    (253, 231, 37),
]

EDGES_COLORS = [
    "#fff100",
    "#ff8c00",
    "#e81123",
    "#ec008c",
    "#68217a",
    "#00188f",
    "#00bcf2",
    "#00b294",
    "#009e49",
    "#bad80a",
]

arrow_valid_filters = [
    "is teacher of",
    "sent letters to",
    "studied the work of",
]


GRAPHML_SUFFIX = ".graphml"


def get_graphml_temp_file():
    temp_file_name = next(tempfile._get_candidate_names()) + GRAPHML_SUFFIX
    full_filename = os.path.join(
        pathlib.Path().absolute(), "diogenet_py", temp_file_name
    )
    print(full_filename)
    return full_filename


@dataclass
class diogenetGraph:

    graph_type = None

    nodes_file = None
    edges_file = None
    locations_file = None
    blacklist_file = None
    current_edges = None

    nodes_raw_data = None
    edges_raw_data = None
    location_raw_data = None
    blacklist_raw_data = None
    igraph_graph = None
    igraph_subgraph = None
    igraph_localgraph = None

    phylosophers_known_origin = None
    multi_origin_phylosophers = None

    graph_layout = None
    graph_layout_name = "None"
    Xn = []
    Yn = []

    # Estetic's attributes (plot attribs)
    node_min_size = 4
    node_max_size = 6
    label_min_size = 4
    label_max_size = 6
    current_centrality_index = "Degree"
    pyvis_title = ""
    pyvis_height = "95%"
    factor = 50
    pyvis_show_gender = False
    pyvis_show_crossing_ties = False
    node_size_factor = 2
    graph_color_map = VIRIDIS_COLORMAP
    vertex_filter = None

    nodes_graph_data = pd.DataFrame()
    edges_graph_data = pd.DataFrame()
    locations_graph_data = pd.DataFrame()
    travels_graph_data = None
    travels_subgraph_data = pd.DataFrame()

    located_nodes = None

    # This is used only when local graph is plotted
    local_phylosopher = None
    local_order = None

    # This is used only to create the communities
    comm_alg = None
    comm_igraph = None

    # Load the R subsystem
    # r = robjects.r
    # r["source"]("diogenet_py//igraph_libraries.R")
    # # Loading the R functions
    # centralization_degree_r = robjects.globalenv["get_degree"]
    # centralization_closeness_r = robjects.globalenv["get_closeness"]
    # centralization_betweenness_r = robjects.globalenv["get_betweenness"]
    # centralization_eigenvector_r = robjects.globalenv["get_eigenvector"]

    def __init__(
        self,
        graph_type=GRAPH_TYPE,
        nodes_file=DATASET_NAME,
        edges_file=DATASET_NAME,
        locations_file=LOCATIONS_DATA_FILE,
        blacklist_file=TRAVELS_BLACK_LIST_FILE,
    ):
        """Create parameters for the class graph

        :param graph_type: It defines the type of the graph (map, global, local, communities)
        :param nodes_file: File with the full list of nodes name/group (.csv)
        :param edges_file: File with full list of edges (.csv)
        :param locations_file: File with list of nodees/localization (.csv).
        :param blacklist_file: File with list of blacklisted places (.csv)
        :param vertex_filter: Array with the vertex that will  be used to
        create the subgraph

        :param nodes_raw_data: Raw data for nodes
        :param edges_raw_data: Raw data for edges
        :param location_raw_data: Raw data for locations
        :param blacklist_raw_data: Raw data for blacklisted places

        :param igraph_graph: Python igraph graph object
        :param igraph_subgraph: Python igraph sub-graph object
        :param igraph_localgraph: Python igraph local graph object

        :param phylosophers_known_origin: Data for phylosophers and their origin
        :param multi_origin_phylosophers: List of phylosophers with more than one
        origin "is from"

        :param nodes_graph_data: Nodes data processed for graph
        :param edges_graph_data: Edges data processed for graph
        :param locations_graph_data: Locations data processed for graph
        :param travels_graph_data: Full data for graph including edges names,
        phylosopher name and coordinates
        :param travels_subgraph_data: Full data for subgraph including edges names,
        phylosopher name and coordinates

        :param located_nodes: Nodes with an identified location in locations data

        :local_phylosopher: Phylosopher for which the local graph is generated
        :local_order: Order of local graph   

        :comm_alg: Algorith for calculating communities (community_infomap, etc...)   
        """
        # :return:
        # :rtype: :py:class:`pd.DataFrame`

        self.graph_type = graph_type
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.locations_file = locations_file
        self.blacklist_file = blacklist_file

        self.edges_filter = []

        # self.know_locations()

        self.set_locations()
        self.set_nodes()
        self.set_edges()
        self.set_blacklist()

        if self.graph_type == "map":
            self.validate_nodes_locations()
            self.validate_phylosopher_origin()
            self.validate_travels_locations()

        self.create_edges_for_graph()
        self.update_graph()
        self.create_subgraph()
        self.tabulate_subgraph_data()

    def know_locations(self):
        """Create parameters for the class graph

        :param nodes_file: File with the full list of nodes name/group (.csv)
        :param edges_file: File with full list of edges (.csv)
        :param locations_file: File with list of nodees/localization (.csv).
        :param igraph_graph: Python igraph graph object

        """
        print("def know_locations(self): Not implemented")
        return pd.DataFrame()

    # Now all the functions that implement data treatment should be implemented
    def set_locations(self):
        """Retrieve and store locations data in the graph object
        :param method: Declare the name of .csv dataset ('locations_data.csv')

        """
        self.location_raw_data = da.get_data_entity_local(self.locations_file)

    def set_nodes(self):
        """Retrieve and store nodes data in the graph object
        :param method: Declare the input name of nodes dataset ('diogenes', 'iamblichus')

        """
        self.nodes_raw_data = da.get_nodes_dataset(self.nodes_file)

    def set_edges(self):
        """Retrieve and store edges data in the graph object
        :param method: Declare the input name of edges dataset ('diogenes', 'iamblichus')

        """
        self.edges_raw_data = da.get_edges_dataset(self.edges_file)

    def set_blacklist(self):
        """Retrieve and store blacklist of travelers (impossible travelers!)
        :param method: Declare the name of .csv dataset ('travels_blacklist.csv')

        """
        self.blacklist_raw_data = da.get_data_entity_local(self.blacklist_file)

    def validate_nodes_locations(self):
        """Determine if places in nodes have the corresponding location
        and update located nodes

        """
        node_place = self.nodes_raw_data["Groups"] == "Place"
        # print("node_place")
        # print(node_place)
        self.located_nodes = self.nodes_raw_data.loc[
            node_place,
        ]
        located_nodes_bool = self.located_nodes.Name.isin(self.location_raw_data.name)
        self.located_nodes = self.located_nodes.loc[
            located_nodes_bool,
        ]

    def validate_travels_locations(self):
        """Filter edges (travels) where travelers have unidentified origin (no coordinates)

        """
        traveled_to_edges = self.edges_raw_data.Relation == "traveled to"
        names_in_traveled_to = self.edges_raw_data.loc[traveled_to_edges, "Source"]
        destiny_in_traveled_to = self.edges_raw_data.loc[traveled_to_edges, "Target"]

        names_in_traveled_to_blacklisted = names_in_traveled_to.isin(
            self.blacklist_raw_data
        )
        names_in_traveled_to = names_in_traveled_to[-names_in_traveled_to_blacklisted]
        destiny_in_traveled_to = destiny_in_traveled_to[
            -names_in_traveled_to_blacklisted
        ]

        pko = np.array(self.phylosophers_known_origin.name)
        ntt = np.array(names_in_traveled_to)
        located_names_in_traveled_to = names_in_traveled_to.isin(pko)
        names_in_traveled_to = names_in_traveled_to[located_names_in_traveled_to]
        destiny_in_traveled_to = destiny_in_traveled_to[located_names_in_traveled_to]
        located_destiny_in_traveled_to = destiny_in_traveled_to.isin(
            self.located_nodes.Name
        )
        names_in_traveled_to = names_in_traveled_to[located_destiny_in_traveled_to]
        destiny_in_traveled_to = destiny_in_traveled_to[located_destiny_in_traveled_to]
        list_of_tuples = list(zip(names_in_traveled_to, destiny_in_traveled_to))

        self.travels_graph_data = pd.DataFrame(
            list_of_tuples, columns=["Source", "Target"]
        )

    def validate_phylosopher_origin(self):
        """Filter "is from" edges where the target (place) is unidentified (no coordinates)

        """
        is_from_edges = self.edges_raw_data.Relation == "is from"
        names_in_is_from = self.edges_raw_data.loc[is_from_edges, "Source"]
        origin_in_is_from = self.edges_raw_data.loc[is_from_edges, "Target"]
        located_origin_in_is_from = origin_in_is_from.isin(self.located_nodes.Name)
        origin_in_is_from = origin_in_is_from[located_origin_in_is_from]
        names_in_is_from = names_in_is_from[located_origin_in_is_from]

        list_of_tuples = list(zip(names_in_is_from, origin_in_is_from))
        self.phylosophers_known_origin = pd.DataFrame(
            list_of_tuples, columns=["name", "origin"]
        )

    def create_edges_for_graph(self):
        """Create Data Frame with all edge's data for graph construction

        """

        traveler_origin = []
        lat_source = []
        lon_source = []
        lat_target = []
        lon_target = []

        multi_origin = []  # Phylosopher with more than one origin city
        Source = []
        Target = []
        Relation = []

        if self.graph_type == "map":
            for idx, cell in enumerate(self.travels_graph_data.Source):
                current_origin = pd.Series.to_list(
                    self.phylosophers_known_origin.origin[
                        self.phylosophers_known_origin.name == cell
                    ]
                )
                current_destiny = self.travels_graph_data.Target[idx]
                current_origin = current_origin[0]
                if len(current_origin) > 1:
                    # If the traveler shows multiple origins by default the
                    multi_origin.append(cell)
                traveler_origin.append(current_origin)
                current_origin = "".join(current_origin)
                self.multi_origin_phylosophers = multi_origin
                lat_source.append(
                    pd.Series.to_list(
                        self.location_raw_data.lat[
                            self.location_raw_data.name.isin([current_origin])
                        ]
                    )
                )
                lon_source.append(
                    pd.Series.to_list(
                        self.location_raw_data.lon[
                            self.location_raw_data.name.isin([current_origin])
                        ]
                    )
                )
                lat_target.append(
                    pd.Series.to_list(
                        self.location_raw_data.lat[
                            self.location_raw_data.name.isin([current_destiny])
                        ]
                    )
                )
                lon_target.append(
                    pd.Series.to_list(
                        self.location_raw_data.lon[
                            self.location_raw_data.name.isin([current_destiny])
                        ]
                    )
                )

            source = traveler_origin
            target = pd.Series.to_list(self.travels_graph_data.Target)
            name = pd.Series.to_list(self.travels_graph_data.Source)

            list_of_tuples = list(
                zip(
                    source, target, name, lat_source, lon_source, lat_target, lon_target
                )
            )
            list_of_tuples_ = list(list(row[0:]) for row in list_of_tuples)
            self.travels_graph_data = list_of_tuples_

        if (
            self.graph_type == "global"
            or self.graph_type == "local"
            or self.graph_type == "communities"
        ):
            node_list = self.nodes_raw_data[
                (self.nodes_raw_data.Groups == "Male")
                | (self.nodes_raw_data.Groups == "Female")
            ]
            edges = self.edges_raw_data[
                (self.edges_raw_data.Relation == "is teacher of")
                | (self.edges_raw_data.Relation == "is teacher of")
                | (self.edges_raw_data.Relation == "is friend of")
                | (self.edges_raw_data.Relation == "is family of")
                | (self.edges_raw_data.Relation == "studied the work of")
                | (self.edges_raw_data.Relation == "sent letters to")
                | (self.edges_raw_data.Relation == "is benefactor of")
            ]
            source = edges["Source"]
            target = edges["Target"]
            name = edges["Relation"]
            self.current_edges = name.unique()
            list_of_tuples = list(zip(source, target, name))
            list_of_tuples_ = list(list(row[0:]) for row in list_of_tuples)
            self.travels_graph_data = list_of_tuples_

        return self.travels_graph_data

    def update_graph(self):
        """Create graph once defined source data
        """
        if self.travels_graph_data is not None:
            self.igraph_graph = igraph.Graph.TupleList(
                self.travels_graph_data, directed=False, edge_attrs=["edge_name"]
            )

            def search_group_attribute(node_name):
                group_name = ""
                # for node in
                return group_name

            attributes = []
            self.nodes_raw_data.set_index(["Name"])
            for node in self.igraph_graph.vs:
                rows = self.nodes_raw_data.loc[
                    self.nodes_raw_data["Name"] == node["name"]
                ]
                node["group"] = rows.iloc[0]["Groups"]

    def calculate_degree(self):
        """Calculate degree for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            return actual_graph.degree()

    def calculate_closeness(self):
        """Create closeness for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            return actual_graph.closeness()

    def calculate_betweenness(self):
        """Calculate betweenness for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            return actual_graph.betweenness()

    def calculate_eigenvector(self):
        """Create degree for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            return actual_graph.evcent()

    def centralization_degree(self):
        """Calculate unnormalized centralization degree for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            full_filename = get_graphml_temp_file()
            actual_graph.write_graphmlz(full_filename, 1)

            # run R script
            # cent_degree = self.centralization_degree_r(full_filename)
            # Commented out to fix R subsystem crash by elinarezv
            cent_degree = []
            cent_degree.append(1)
            # Delete temp file
            os.remove(full_filename)

            return cent_degree[0]

    def centralization_betweenness(self):
        """Calculate unnormalized centralization betweenness for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            full_filename = get_graphml_temp_file()
            actual_graph.write_graphmlz(full_filename, 1)

            # run R script
            # cent_betweenness = self.centralization_betweenness_r(full_filename)
            # Commented out to fix R subsystem crash by elinarezv
            cent_betweenness = []
            cent_betweenness.append(1)
            # Delete temp file
            os.remove(full_filename)

            return cent_betweenness[0]

    def centralization_closeness(self):
        """Calculate unnormalized centralization closeness for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            full_filename = get_graphml_temp_file()
            actual_graph.write_graphmlz(full_filename, 1)

            # run R script
            # cent_closeness = self.centralization_closeness_r(full_filename)
            # Commented out to fix R subsystem crash by elinarezv
            cent_closeness = []
            cent_closeness.append(1)
            # Delete temp file
            os.remove(full_filename)

            return cent_closeness[0]

    def centralization_eigenvector(self):
        """Calculate unnormalized centralization eigen vector for the graph
        """
        if self.igraph_graph is not None:
            actual_graph = self.create_subgraph()
            full_filename = get_graphml_temp_file()
            actual_graph.write_graphmlz(full_filename, 1)

            # run R script
            # cent_eigenvector = self.centralization_eigenvector_r(full_filename)
            # Commented out to fix R subsystem crash by elinarezv
            cent_eigenvector = []
            cent_eigenvector.append(1)
            # Delete temp file
            os.remove(full_filename)

            return cent_eigenvector[0]

    def get_vertex_names(self):
        """Return names for each vertex of the graph
        """
        if self.igraph_graph is not None:
            # self.create_subgraph()
            vertex_names = []
            for vertex in self.igraph_subgraph.vs:
                vertex_names.append(vertex["name"])
            return vertex_names

    def get_edges_names(self):
        """Return names for each edge of the graph
        """
        if self.igraph_graph is not None:
            # self.create_subgraph()
            edges_names = []
            for edges in self.igraph_subgraph.es:
                print(edges)
                edges_names.append(edges["edge_name"])
            return edges_names

    def get_max_min(self):
        if self.igraph_graph is not None:
            centrality_indexes = []
            ret_val = {}
            if self.current_centrality_index == "Degree":
                centrality_indexes = self.calculate_degree()
            elif self.current_centrality_index == "Betweeness":
                centrality_indexes = self.calculate_betweenness()
            elif self.current_centrality_index == "Closeness":
                centrality_indexes = self.calculate_closeness()
            elif self.current_centrality_index == "Eigenvector":
                centrality_indexes = self.calculate_eigenvector()
            ret_val["min"] = min(centrality_indexes)
            ret_val["max"] = max(centrality_indexes)

            return ret_val

    def get_interpolated_index(self, r1_min, r1_max, r1_value, r2_min=0, r2_max=9):
        """Get an interpolated integer from range [r1_min, r1_max] to [r2_min..r2_max]
        """
        index = 0
        if r1_max > r1_min:
            index = int(
                round(
                    (
                        r2_min
                        + ((r1_value - r1_min) / (r1_max - r1_min)) * (r2_max - r2_min)
                    ),
                    0,
                )
            )
        elif r1_max == r1_min:
            index = r1_min

        return index

    def rgb_to_hex(self, rgb):
        """Converts a triplet (r, g, b) of bytes (0-255) in web color #000000
        :param url: a Triplet of bites with RGB color info
        :returns a string with web color format
        :rtype: :py:str

        """
        return "%02x%02x%02x" % rgb

    def get_graph_centrality_indexes(self):
        centrality_indexes = []
        if self.current_centrality_index == "Degree":
            centrality_indexes = self.calculate_degree()
        if self.current_centrality_index == "Betweeness":
            centrality_indexes = self.calculate_betweenness()
        if self.current_centrality_index == "Closeness":
            centrality_indexes = self.calculate_closeness()
        if self.current_centrality_index == "Eigenvector":
            centrality_indexes = self.calculate_eigenvector()
        if self.current_centrality_index == "communities":
            (modularity, clusters_dict) = self.identify_communities()
            self.pyvis_title = "MODULARITY: {:0.4f}".format(modularity)
            self.pyvis_height = "88%"
            for i in range(len(self.igraph_subgraph.vs)):
                if self.igraph_subgraph.vs[i]["name"] in clusters_dict.keys():
                    centrality_indexes.append(
                        clusters_dict[self.igraph_subgraph.vs[i]["name"]]
                    )

        centrality_indexes_min = min(centrality_indexes)
        centrality_indexes_max = max(centrality_indexes)

        return (centrality_indexes, centrality_indexes_min, centrality_indexes_max)

    def set_graph_layout(self, layout):
        if self.igraph_graph is not None:
            # self.create_subgraph()
            N = len(self.igraph_subgraph.vs)
            if layout == "kk":
                self.graph_layout = self.igraph_subgraph.layout_kamada_kawai()
                self.graph_layout_name = "kk"
                self.factor = 80
            elif layout == "grid_fr":
                self.graph_layout = self.igraph_subgraph.layout_grid()
                self.graph_layout_name = "grid_fr"
                self.factor = 80
            elif layout == "circle":
                self.graph_layout = self.igraph_subgraph.layout_circle()
                self.graph_layout_name = "circle"
                self.factor = 250
                self.node_size_factor = 1
            elif layout == "sphere":
                self.graph_layout = self.igraph_subgraph.layout_sphere()
                self.graph_layout_name = "sphere"
                self.factor = 250
                self.node_size_factor = 1
            else:
                self.graph_layout = self.igraph_subgraph.layout_fruchterman_reingold()
                self.graph_layout_name = "fr"
                self.factor = 50

            self.Xn = [self.graph_layout[k][0] for k in range(N)]
            self.Yn = [self.graph_layout[k][1] for k in range(N)]

    def get_pyvis_options(
        self, min_weight=4, max_weight=6, min_label_size=4, max_label_size=6,
    ):
        pyvis_map_options = {}
        pyvis_map_options["nodes"] = {
            "font": {"size": min_label_size + 8},
            "scaling": {"min": min_label_size, "max": max_label_size},
        }

        show_arrows = True

        for filter in self.edges_filter:
            if filter not in arrow_valid_filters:
                show_arrows = False

        if self.graph_type == "communities" and len(self.edges_filter) > 1:
            show_arrows = False

        pyvis_map_options["edges"] = {
            "arrows": {"to": {"enabled": show_arrows, "scaleFactor": 0.4}},
            "color": {"inherit": True},
            "smooth": True,
            "scaling": {
                "label": {
                    "min": min_weight * 10,
                    "max": max_weight * 10,
                    "maxVisible": 18,
                }
            },
        }
        pyvis_map_options["physics"] = {"enabled": False}
        pyvis_map_options["interaction"] = {
            "dragNodes": True,
            "hideEdgesOnDrag": True,
            "hover": True,
            "navigationButtons": False,
            "selectable": True,
            "multiselect": True,
        }
        pyvis_map_options["manipulation"] = {
            "enabled": False,
            "initiallyActive": True,
        }
        # Allow or remove PYVIS configure options
        # pyvis_map_options["configure"] = {"enabled": True}
        pyvis_map_options["configure"] = {"enabled": False}
        return pyvis_map_options

    def get_pyvis(
        self,
        min_weight=4,
        max_weight=6,
        min_label_size=4,
        max_label_size=6,
        layout="fr",
        avoid_centrality=False,
    ):
        """Create a pyvis object based on current igraph network
        :param int min_weight: Integer with min node size
        :param int max_weight: Integer with max node size
        :param int min_label_size: Integer with min label size
        :param int max_label_size: Integer with max label size
        :param str layout: String with a valid iGraph Layout like
        "fruchterman_reingold", "kamada_kawai" or "circle"
        :return: A PyVis Object filled with the network's data.
        :rtype: :py:class:`pyvis`
        """
        pv_graph = None

        random.seed(1234)

        if self.igraph_graph is not None:
            (
                centrality_indexes,
                centrality_indexes_min,
                centrality_indexes_max,
            ) = self.get_graph_centrality_indexes()

            # if (self.graph_layout is None) or (layout != self.graph_layout_name):
            self.set_graph_layout(layout)

            pv_graph = pyvis.network.Network(
                height=self.pyvis_height, width="100%", heading=self.pyvis_title
            )
            pyvis_map_options = self.get_pyvis_options(
                min_weight, max_weight, min_label_size, max_label_size
            )
            pv_graph.set_options(json.dumps(pyvis_map_options))
            # pv_graph.show_buttons()

            # self.create_subgraph()
            # Add Nodes
            cut_vertex = []
            if self.graph_type == "communities":
                cut_vertex = self.igraph_subgraph.cut_vertices()

            for node in self.igraph_subgraph.vs:
                node_title = node["name"]
                if not avoid_centrality:
                    if self.pyvis_show_crossing_ties:
                        if node.index in cut_vertex:
                            color_index = self.get_interpolated_index(
                                centrality_indexes_min,
                                centrality_indexes_max,
                                centrality_indexes[node.index],
                            )
                            color = "#" + self.rgb_to_hex(VIRIDIS_COLORMAP[color_index])
                        else:
                            color = "#bbbbbb"
                    else:
                        color_index = self.get_interpolated_index(
                            centrality_indexes_min,
                            centrality_indexes_max,
                            centrality_indexes[node.index],
                        )
                        color = "#" + self.rgb_to_hex(VIRIDIS_COLORMAP[color_index])

                    node_title += (
                        " - "
                        + self.current_centrality_index
                        + ": "
                        + "{:.3f}".format(centrality_indexes[node.index])
                    )
                else:
                    color = "#ff6347"

                size = self.get_interpolated_index(
                    centrality_indexes_min,
                    centrality_indexes_max,
                    centrality_indexes[node.index],
                    min_weight,
                    max_weight,
                )
                node_shape = "dot"
                if self.pyvis_show_gender:
                    if node["group"] == "Female":
                        node_shape = "star"
                        node_title += " - Female"
                    elif node["group"] == "God":
                        node_shape = "triangle"
                        node_title += " - God"
                    else:
                        node_title += " - Male"

                pv_graph.add_node(
                    node.index,
                    label=node["name"],
                    color=color,
                    size=int(size * self.node_size_factor),
                    x=int(self.Xn[node.index] * self.factor),
                    y=int(self.Yn[node.index] * self.factor),
                    shape=node_shape,
                    title=node_title,
                )

            edges = {}
            i = 1
            edges_colors_list = {
                "is teacher of": "#411271",
                "is friend of": "#4542B9",
                "is family of": "#1FC3CD",
                "is benefactor of": "#01AA31",
                "studied the work of": "#F5C603",
                "sent letters to": "#D62226",
            }
            for edge in self.igraph_subgraph.es:
                if self.graph_type == "map":
                    title = (
                        edge["edge_name"]
                        + " travels from: "
                        + self.igraph_subgraph.vs[edge.source]["name"]
                        + " to: "
                        + self.igraph_subgraph.vs[edge.target]["name"]
                    )
                    edge_color = "#ff6347"
                else:
                    title = (
                        self.igraph_subgraph.vs[edge.source]["name"]
                        + " "
                        + edge["edge_name"]
                        + " "
                        + self.igraph_subgraph.vs[edge.target]["name"]
                    )
                    edge_color = ""
                    if self.pyvis_show_crossing_ties:
                        if edge.source in cut_vertex and edge.target in cut_vertex:
                            edge_color = edges_colors_list[edge["edge_name"]]
                        else:
                            edge_color = "#888888"
                    else:
                        edge_color = edges_colors_list[edge["edge_name"]]
                pv_graph.add_edge(
                    edge.source, edge.target, title=title, color=edge_color
                )
        return pv_graph

    def get_igraph_plot(
        self,
        min_weight=4,
        max_weight=6,
        min_label_size=4,
        max_label_size=6,
        layout="fr",
        avoid_centrality=False,
    ):
        return True

    def get_map_data(
        self, min_weight=4, max_weight=6, min_label_size=4, max_label_size=6,
    ):
        if self.igraph_graph is not None:
            centrality_indexes = []
            if self.current_centrality_index == "Degree":
                centrality_indexes = self.calculate_degree()
            elif self.current_centrality_index == "Betweeness":
                centrality_indexes = self.calculate_betweenness()
            elif self.current_centrality_index == "Closeness":
                centrality_indexes = self.calculate_closeness()
            else:
                centrality_indexes = self.calculate_eigenvector()

            centrality_indexes_min = min(centrality_indexes)
            centrality_indexes_max = max(centrality_indexes)

            map = []
            # nodes = self.get_edges_names()
            nodes = self.get_vertex_names()
            # print("self.travels_graph_data")
            # print(self.travels_graph_data)
            # print("nodes")
            # print(nodes)
            map_dict_strings = [
                "Source",
                "Destination",
                "Philosopher",
                "SourceLatitude",
                "SourceLongitude",
                "DestLatitude",
                "DestLongitude",
            ]
            if self.igraph_subgraph:

                self.tabulate_subgraph_data()

                for record in self.travels_subgraph_data:
                    index = 0
                    map_record = {}
                    for item in record:
                        tmp_value = item
                        if isinstance(item, list):
                            if len(item) == 1:
                                tmp_value = item[0]
                        map_record[map_dict_strings[index]] = tmp_value
                        if index == 0:
                            i = nodes.index(tmp_value)
                            color_index = self.get_interpolated_index(
                                centrality_indexes_min,
                                centrality_indexes_max,
                                centrality_indexes[i],
                            )
                            color = "#" + self.rgb_to_hex(VIRIDIS_COLORMAP[color_index])
                            map_record["SourceColor"] = color
                            size = self.get_interpolated_index(
                                centrality_indexes_min,
                                centrality_indexes_max,
                                centrality_indexes[i],
                                min_weight,
                                max_weight,
                            )
                            map_record["SourceSize"] = size
                        elif index == 1:
                            i = nodes.index(tmp_value)
                            color_index = self.get_interpolated_index(
                                centrality_indexes_min,
                                centrality_indexes_max,
                                centrality_indexes[i],
                            )
                            color = "#" + self.rgb_to_hex(VIRIDIS_COLORMAP[color_index])
                            map_record["DestinationColor"] = color
                            size = self.get_interpolated_index(
                                centrality_indexes_min,
                                centrality_indexes_max,
                                centrality_indexes[i],
                                min_weight,
                                max_weight,
                            )
                            map_record["DestinationSize"] = size
                        index = index + 1
                    map.append(map_record)
        return map

    def set_edges_filter(self, edges_filter):
        """Create subgraph depending on vertex selected
        """
        # if (edges_filter  not in self.edges_filter):
        # self.edges_filter = []
        self.edges_filter.append(edges_filter)

    def create_subgraph(self):
        """Create subgraph depending on edges selected (i.e travellers in case of)
           the map and type of relations in case of the graph 
        """

        subgraph = None
        if self.igraph_graph is not None:
            edges = self.igraph_graph.es
            edge_names = self.igraph_graph.es["edge_name"]
            # print('edge_names')
            # print(edge_names)
            if not self.edges_filter:
                if self.graph_type == "map":
                    edges_filter = edge_names
                else:
                    edges_filter = "is teacher of"
            else:
                # if not self.edges_filter:
                edges_filter = self.edges_filter
                # print("travellers")
                # print(travellers)
            edge_indexes = [
                j.index for i, j in zip(edge_names, edges) if i in edges_filter
            ]
            subgraph = self.igraph_graph.subgraph_edges(edge_indexes)

            self.igraph_subgraph = subgraph

            """Create local subgraph depending on vertex selected (i.e phylosophers)
            """

            if self.graph_type == "local":
                # If no vertex selected return global graph
                if self.local_phylosopher:
                    neighbour_vertex = subgraph.neighborhood(
                        self.local_phylosopher, self.local_order
                    )
                    # for number in neighbour_vertex:
                    #    print(actual_graph.vs["name"][number])
                    # print(neighbour_vertex)
                    subgraph = subgraph.induced_subgraph(neighbour_vertex)
                self.igraph_subgraph = subgraph
        return subgraph

    def get_subgraph(self):
        subgraph = None
        # if self.edges_filter:
        #     sub_igraph = self.create_subgraph()
        #     self.tabulate_subgraph_data()
        #     sub_travels_map_data = self.travels_subgraph_data
        #     subgraph = copy.deepcopy(self)
        #     subgraph.igraph_graph = sub_igraph
        #     subgraph.travels_graph_data = sub_travels_map_data
        subgraph = self.igraph_subgraph
        return subgraph

    def get_localgraph(self):
        # subgraph = None
        # sub_igraph = self.create_local_graph()
        # self.tabulate_subgraph_data()
        # sub_travels_map_data = self.travels_subgraph_data
        # subgraph = copy.deepcopy(self)
        # subgraph.local_phylosopher = self.local_phylosopher
        # subgraph.local_order = self.local_order
        # subgraph.igraph_graph = sub_igraph
        # subgraph.travels_graph_data = sub_travels_map_data
        subgraph = self.igraph_subgraph
        return subgraph

    def set_colour_scale(self):
        """Create parameters for the class graph

        :param nodes_file: File with the full list of nodes name/group (.csv)
        :param edges_file: File with full list of edges (.csv)
        :param locations_file: File with list of nodees/localization (.csv).
        :param igraph_graph: Python igraph graph object

        """
        return ()

    def tabulate_subgraph_data(self):
        """Create datatable for subgraph
        """
        source = []
        target = []
        name = []
        lat_source = []
        lon_source = []
        lat_target = []
        lon_target = []

        vertex_list = []
        edges_list = []

        if self.igraph_subgraph:
            for vertex in self.igraph_subgraph.vs:
                vertex_list.append(vertex["name"])

        if self.igraph_subgraph:
            for idx, edges in enumerate(self.igraph_subgraph.es):
                source.append(vertex_list[edges.tuple[0]])
                target.append(vertex_list[edges.tuple[1]])
                name.append(edges["edge_name"])
                lat_source.append(
                    pd.Series.to_list(
                        self.location_raw_data.lat[
                            self.location_raw_data.name.isin(
                                [vertex_list[edges.tuple[0]]]
                            )
                        ]
                    )
                )

                lon_source.append(
                    pd.Series.to_list(
                        self.location_raw_data.lon[
                            self.location_raw_data.name.isin(
                                [vertex_list[edges.tuple[0]]]
                            )
                        ]
                    )
                )

                lat_target.append(
                    pd.Series.to_list(
                        self.location_raw_data.lat[
                            self.location_raw_data.name.isin(
                                [vertex_list[edges.tuple[1]]]
                            )
                        ]
                    )
                )

                lon_target.append(
                    pd.Series.to_list(
                        self.location_raw_data.lon[
                            self.location_raw_data.name.isin(
                                [vertex_list[edges.tuple[1]]]
                            )
                        ]
                    )
                )

        list_of_tuples = list(
            zip(source, target, name, lat_source, lon_source, lat_target, lon_target)
        )
        list_of_tuples_ = list(list(row[0:]) for row in list_of_tuples)
        self.travels_subgraph_data = list_of_tuples_

    def get_current_edges(self):
        if self.current_edges is not None:
            return self.current_edges

    def get_global_edges_types(self):
        edges_types = []
        for edge_name in self.igraph_subgraph.es:
            if edge_name["edge_name"] not in edges_types:
                edges_types.append(edge_name["edge_name"])
        return edges_types

    def create_local_graph(self):
        return ()

    def fix_dendrogram(self, graph, cl):
        already_merged = set()
        for merge in cl.merges:
            already_merged.update(merge)

        num_dendrogram_nodes = graph.vcount() + len(cl.merges)
        not_merged_yet = sorted(set(range(num_dendrogram_nodes)) - already_merged)
        if len(not_merged_yet) < 2:
            return

        v1, v2 = not_merged_yet[:2]
        cl._merges.append((v1, v2))
        del not_merged_yet[:2]

        missing_nodes = range(
            num_dendrogram_nodes, num_dendrogram_nodes + len(not_merged_yet)
        )
        cl._merges.extend(zip(not_merged_yet, missing_nodes))
        cl._nmerges = graph.vcount() - 1
        return cl

    def identify_communities(self):
        clusters = []
        actual_graph = self.create_subgraph()
        if self.comm_alg == "community_infomap":
            self.comm = actual_graph.community_infomap()
            # print('community_infomap')
            # membership = comm.membership
            clusters = self.comm.as_cover()
            modularity = self.comm.modularity

        if self.comm_alg == "community_edge_betweenness":
            self.comm = actual_graph.community_edge_betweenness()
            # print("community_edge_betweenness")
            # print(vars(self.comm))
            aux = self.fix_dendrogram(actual_graph, self.comm)
            clusters_ini = aux.as_clustering()
            clusters = clusters_ini.as_cover()
            modularity = clusters_ini.modularity
            # membership = clusters.membership

        if self.comm_alg == "community_spinglass":
            self.comm = actual_graph.community_spinglass()
            clusters = self.comm.as_cover()
            modularity = self.comm.modularity
            # membership = comm.membership

        if self.comm_alg == "community_walktrap":
            self.comm = actual_graph.community_walktrap()
            aux = self.fix_dendrogram(actual_graph, self.comm)
            clusters_ini = aux.as_clustering()
            clusters = clusters_ini.as_cover()
            modularity = clusters_ini.modularity
            # membership = clusters.membership

        if self.comm_alg == "community_leiden":
            self.comm = actual_graph.community_leiden()
            # membership = self.comm.membership
            clusters = self.comm.as_cover()
            modularity = self.comm.modularity

        if self.comm_alg == "community_fastgreedy":
            self.comm = actual_graph.community_fastgreedy()
            aux = self.fix_dendrogram(actual_graph, self.comm)
            clusters_ini = aux.as_clustering()
            clusters = clusters_ini.as_cover()
            modularity = clusters_ini.modularity
            # membership = clusters.membership

        if self.comm_alg == "community_leading_eigenvector":
            self.comm = actual_graph.community_leading_eigenvector()
            clusters = self.comm.as_cover()
            modularity = self.comm.modularity
            # membership = self.comm.membership

        if self.comm_alg == "community_label_propagation":
            self.comm = actual_graph.community_label_propagation()
            # membership = self.comm.membership
            clusters = self.comm.as_cover()
            modularity = self.comm.modularity

        if self.comm_alg == "community_multilevel":
            self.comm = actual_graph.community_multilevel()
            clusters = self.comm.as_cover()
            modularity = self.comm.modularity
            # membership = self.comm.membership

        community_data = []
        community_names = []
        for i in range(len(clusters)):
            for j in range(len(clusters.subgraph(i).vs)):
                community_data.append(i)
                community_names.append(clusters.subgraph(i).vs[j]["name"])
        comm_dataFrame = zip(community_names, community_data)
        comm_Dict = dict(comm_dataFrame)

        return (modularity, comm_Dict)

    def get_cut_vertices(self):
        cutVertices = self.igraph_subgraph.cut_vertices()
        return cutVertices


map_graph = diogenetGraph(
    "map",
    DATASET_NAME,
    DATASET_NAME,
    LOCATIONS_DATA_FILE,
    TRAVELS_BLACK_LIST_FILE,
)

global_graph = diogenetGraph(
    "global",
    DATASET_NAME,
    DATASET_NAME,
    LOCATIONS_DATA_FILE,
    TRAVELS_BLACK_LIST_FILE,
)

local_graph = diogenetGraph(
    "local",
    DATASET_NAME,
    DATASET_NAME,
    LOCATIONS_DATA_FILE,
    TRAVELS_BLACK_LIST_FILE,
)

communities_graph = diogenetGraph(
    "communities",
    DATASET_NAME,
    DATASET_NAME,
    LOCATIONS_DATA_FILE,
    TRAVELS_BLACK_LIST_FILE,
)


def map_graph_change_dataset(dataset):
    global map_graph
    if dataset == "iamblichus":
        map_graph = diogenetGraph(
            "map",
            "new_Nodes_Life_of_Pythagoras_Iamblichus.csv",
            "new_Edges_Life_of_Pythagoras_Iamblichus.csv",
            LOCATIONS_DATA_FILE,
            TRAVELS_BLACK_LIST_FILE,
        )
    else:
        map_graph = diogenetGraph(
            "map",
            NODES_DATA_FILE,
            EDGES_DATA_FILE,
            LOCATIONS_DATA_FILE,
            TRAVELS_BLACK_LIST_FILE,
        )
    return map_graph


# communities_graph.comm_alg = 'community_infomap'                          # OK
# communities_graph.comm_alg = "community_edge_betweenness"  # OK
# communities_graph.comm_alg = 'community_spinglass'                        # Not for unconnected graphs
# communities_graph.comm_alg = 'community_walktrap'	                       # OK
# communities_graph.comm_alg = 'community_leiden'                           # No clusters. No go
# communities_graph.comm_alg = 'community_fastgreedy'                       # OK
# communities_graph.comm_alg = 'communitmodularity, clusters = communities_graph.identify_communities()y_leading_eigenvector'              # OK
# communities_graph.comm_alg = 'community_label_propagation'                # OK
# communities_graph.comm_alg = 'community_multilevel'                       # OK


# communities_graph.set_edges_filter("is teacher of")
# communities_graph.create_subgraph()
# modularity, clusters = communities_graph.identify_communities()
# cut_vertices = communities_graph.get_cut_vertices()
# print(modularity)
# print(clusters["Plato"])
# print(repr(clusters))

# grafo.centralization_degree()
# grafo.centralization_betweenness()
# grafo.centralization_closeness()
# grafo.centralization_eigenvector()

# grafo.set_edges_filter("Aristotle")
# # grafo.set_edges_filter("Pythagoras")
# grafo.create_subgraph()
# # print(grafo.igraph_subgraph)

# grafo.tabulate_subgraph_data()

# datos_sub_grafo =
# print(datos_sub_grafo)

# grafo.set_edges_filter("Aristotle")
# grafo.set_edges_filter("Pythagoras")
# print(grafo.create_subgraph())

# local_graph.local_phylosopher = "Plato"
# local_graph.local_order = 1
# print(local_graph.create_local_graph())
