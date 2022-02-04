"""Module to handle Complex Networks Graph relations in ancient Greece.

.. platform:: Unix, Windows, Mac
.. moduleauthor:: Elio Linarez <elinarezv@gmail.com>
"""
import pandas as pd
import io
import requests
import os

#  Not needed. Data is called from the class. .
#  from . import network_graph

# VARS to fullfill dataframes
# Define data access method: local, url or database

from . import network_graph as ng


def get_data_entity_local(entity_name):
    """Retrieve a dataset with Complex Network's data.

    :param str entity_name: The name of the entity file (csv file name or table name) for local access method.
    :return: A Data Frame filled with the entity's data.
    :rtype: :py:class:`pd.DataFrame`
    """
    df = None
    df = pd.read_csv(f'./data/{entity_name}', delimiter=",", header=0)
    return df

def get_data_entity_database(entity_name, method):
    """Retrieve a dataset with Complex Network's data.

    :param str entity_name: The name of the entity file (csv file name or table name) for database access method.
    :return: A Data Frame filled with the entity's data.
    :rtype: :py:class:`pd.DataFrame`
    """
    df = None

    return df

def get_data_entity_url(entity_name):
    """Retrieve a dataset with Complex Network's data.

    :param str entity_name: The name of the entity file (csv file name or table name) for url access method.
    :return: A Data Frame filled with the entity's data.
    :rtype: :py:class:`pd.DataFrame`
    """
    df = None
    url = ng.BASE_URL + "/" + entity_name
    request = requests.get(url).contents
    df = pd.read_csv(io.StringIO(request.decode("utf-8")), delimiter=",", header=0)

    return df

def get_nodes_dataset(input = 'diogenes'):
    """Retrieve a Network's nodes dataset.

    :param str input: The name of the entity file (ie.: 'iamblichus', 'diogenes',...).
    :return: A Data Frame filled with the entity's data.
    :rtype: :py:class:`pd.DataFrame`
    """

    # lista de datasets disponibles en /data
    dataset_list_df = pd.read_csv("./data/datasetList.csv")

    # condiciones para el path name 
    m1 = dataset_list_df['name'] == str(input)
    m2 = dataset_list_df['type'] == 'nodes'

    # leyendo el archivo de datos de nodos
    nodes_path_name = str(list(dataset_list_df[m1&m2]['path'])[0])
    nodes_path_dir = os.path.abspath(os.path.dirname(nodes_path_name))
    nodes_dataset = pd.read_csv(f'./data/{nodes_path_name}', delimiter=",", header=0)

    return nodes_dataset

def get_edges_dataset(input = 'diogenes'):
    """Retrieve a Network's edges dataset.

    :param str input: The name of the entity file (ie.: 'iamblichus', 'diogenes',...).
    :return: A Data Frame filled with the entity's data.
    :rtype: :py:class:`pd.DataFrame`
    """

    # lista de datasets disponibles en /data
    dataset_list_path = os.path.abspath(os.path.dirname("datasetList.csv"))
    dataset_list = pd.read_csv("./data/datasetList.csv")

    # condiciones para el path name 
    m1 = dataset_list['name'] == str(input)
    m3 = dataset_list['type'] == 'edges'

    # leyendo el archivo de datos de edges
    edges_path_name = str(list(dataset_list[m1&m3]['path'])[0])
    edges_dataset = pd.read_csv(f'./data/{edges_path_name}', delimiter=",", header=0)

    return edges_dataset


#travel_edges = pd.read_csv("travel_edges_graph.csv", delimiter=",")
#all_places = pd.read_csv("all_places_graph.csv", delimiter=",")
