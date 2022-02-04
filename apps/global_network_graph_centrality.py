from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pathlib
import os
import sys
import requests
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import (
    Flask,
    render_template,
    make_response,
    request,
    send_from_directory,
    send_file,
    jsonify,
)

from app import app
from data_analysis_module.network_graph import diogenetGraph

dict_of_datasets = {'Diogenes Laertius': 'diogenes', 'Life of Pythagoras Iamblichus': 'iamblichus'}

STYLE_A_ITEM = {
    'color':'#ffffff',
    'textDecoration': 'none'
}

navbar = dbc.Navbar(
    children=[
        html.Div(
            [
                dbc.NavLink("Horus", style=STYLE_A_ITEM),
                dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("Graph", href="/apps/global_network_graph"), 
                        dbc.DropdownMenuItem("Graph + centrality", href="#")
                    ],
                    label="Global Network",
                    style=STYLE_A_ITEM,
                    color="#1a6ecc"
                )
            ], 
            className="d-flex",

        ),
        dbc.NavLink(
            [
                html.I(className="bi bi-house-fill me-2 text-white")
            ], 
            href="https://diogenet.ucsd.edu/", style=STYLE_A_ITEM,
            target="blank"
        ),
            
    ],
    color="#1a6ecc",
    className="d-flex justify-content-between",
    style={'color':'#ffffff'},
    id='navbar-global-centrality'
)

sidebar_content = [
    html.H5('Dataset selection', className="mt-3 mb-3"),
    dcc.Dropdown(
        id='dataset_selection_global_centrality',
        options=[
            {'label': key, 'value': value}
            for key, value in dict_of_datasets.items()
        ],
        searchable=False,
        placeholder="Select a dataset",
        value='diogenes'
    ),
    html.H5('Network ties', className="mt-5 mb-3"),
    dcc.Checklist( 
        id='graph_filter_global_centrality',
        options=[
            {'label': ' Is teacher of', 'value': 'is teacher of'},
                {'label': ' Is friend of', 'value': 'is friend of'},
                {'label': ' Is family of', 'value': 'is family of'},
                {'label': ' Studied the work of', 'value': 'studied the work of'},
                {'label': ' Sent letters to', 'value': 'sent letters to'},
                {'label': ' Is benefactor of', 'value': 'is benefactor of'},
        ],
        value=['is teacher of'],
        labelStyle={'display': 'flex', 'flexDirection':'row','alingItem':'center'},
        inputStyle={'margin':'0px 5px'},
    ),
    html.H5('Graph layout',className="mt-5 mb-3"),
    dcc.Dropdown(
        id='graph_layout_global_centrality',
        options=[
            {'label': 'Fruchterman-Reingold', 'value': 'fr'},
            {'label': 'Kamada-Kawai', 'value': 'kk'},
            {'label': 'On Sphere', 'value': 'sphere'},
            {'label': 'In Circle', 'value': 'circle'},
            {'label': 'On Grid', 'value': 'grid_fr'},
        ],
        value='fr',
        searchable=False,
    ),
    dcc.Checklist(
        id='show_gender_global_centrality',
        options=[
            {'label': ' Gender and Gods', 'value': "True"}
        ],
        labelStyle={'display': 'flex', 'flexDirection':'row','alingItem':'center'},
        inputStyle={'margin':'0px 5px'},
        className='mt-3'
    ),
    html.H5('Centrality index',className="mt-5 mb-3"),
    dcc.Dropdown(
        id='centrality_index_global_centrality',
        options=[
            {'label': 'Degree', 'value': 'Degree'},
            {'label': 'Betweeness', 'value': 'Betweeness'},
            {'label': 'Closeness', 'value': 'Closeness'},
            {'label': 'Eigenvector', 'value': 'Eigenvector'},
        ],
        value='Degree',
        searchable=False,
    ),
    html.H5('Appearence',className="mt-5 mb-3"),
    html.H6('Label Size',className="mt-1 mb-2"),
    dcc.RangeSlider(
        id='label_size_global_centrality',
        min=0,
        max=10,
        step=1,
        marks={
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '10',
        },
        value=[4, 6]
    ),
    html.H6('Node Size',className="mt-1 mb-2"),
    dcc.RangeSlider(
        id='node_size_global_centrality',
        min=0,
        max=10,
        step=1,
        marks={
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '10',
        },
        value=[4, 6]
    ),
    html.H6('Download current dataset',className="mt-5 mb-3"),
    dbc.Button("Download Data", id="btn_csv_global", color="secondary", className="ml-3"),
    dcc.Download(id="download-dataframe-csv-global"),
]

tabs = dcc.Tabs(
        id='tabs_global_cetrality', 
        value='graph_global_cetrality',
        parent_className='custom-tabs',
        className='custom-tabs-container',
        children=[
            dcc.Tab(
                label='Graph', 
                value='graph_global_cetrality',
                className='custom-tab',
                selected_className='custom-tab--selected'    
            ),
                
            dcc.Tab(
                label='Heatmap', 
                value='heatmap_global_cetrality',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
            dcc.Tab(
                label='Metrics', 
                value='metrics_global_cetrality',
                className='custom-tab',
                selected_className='custom-tab--selected'
            ),
        ])


row = html.Div(
    [
        dbc.Row(navbar),
        dbc.Row(
            [
                dbc.Col(html.Div(sidebar_content), id='sidebar_global_centrality', width=3, style={"backgroundColor": "#2780e31a", "padding":'30px 10px 10px 10px'}),
                dbc.Col(html.Div([tabs, html.Div(id="content", style={'height': '100vh'})]), id='main_global_centrality'),
                dcc.ConfirmDialog(
                        id='confirm-warning-tie-centrality',
                        message='You must select at least one tie',
                    ),
            ],
            className='h-100'
        ),
    ],
    className='h-100',
    style={'padding':'0px 10px'}
)

layout = html.Div([
    row
],  style={"height": "100vh"})

# Update the index

@app.callback(
    Output("content", "children"),[
    Input('tabs_global_cetrality', 'value'),
    Input('dataset_selection_global_centrality', 'value'),
    Input('graph_filter_global_centrality', 'value'),
    Input('graph_layout_global_centrality', 'value'),
    Input('show_gender_global_centrality', 'value'),
    Input('centrality_index_global_centrality', 'value'),
    Input('label_size_global_centrality', 'value'),
    Input('node_size_global_centrality', 'value')
    ]
)
def horus_get_global_graph_centrality(
                                    tab,
                                    dataset_selection_global_centrality,
                                    graph_filter_global_centrality,
                                    graph_layout_global_centrality,
                                    show_gender_global_centrality,
                                    centrality_index_global_centrality,
                                    label_size_global_centrality,
                                    node_size_global_centrality):
    global_graph = diogenetGraph(
            "global",
            dataset_selection_global_centrality,
            dataset_selection_global_centrality,
            'locations_data.csv',
            'travels_blacklist.csv'
        )

    plot_type = "pyvis"
    centrality_index = centrality_index_global_centrality
    node_min_size = int(node_size_global_centrality[0])
    node_max_size = int(node_size_global_centrality[1])
    label_min_size = int(label_size_global_centrality[0])
    label_max_size = int(label_size_global_centrality[1])
    global_graph.current_centrality_index = "Degree"
    not_centrality = False
    graph_layout = graph_layout_global_centrality
    graph_filter = graph_filter_global_centrality
    
    if tab == "graph_global_cetrality":
        if centrality_index:
            if centrality_index in [None, "", "None"]:
                global_graph.current_centrality_index = "Degree"
                not_centrality = True
            else:
                global_graph.current_centrality_index = centrality_index


        if not graph_filter:
            global_graph.set_edges_filter("is teacher of")
        else:
            global_graph.edges_filter = []
            filters = graph_filter
            for m_filter in filters:
                global_graph.set_edges_filter(m_filter)

        if not graph_layout_global_centrality:
            graph_layout_global_centrality = "fr"

        if show_gender_global_centrality:
            global_graph.pyvis_show_gender = True
        else:
            global_graph.pyvis_show_gender = False

        global_graph.create_subgraph()
        pvis_graph = None

        if plot_type == "pyvis":
            pvis_graph = global_graph.get_pyvis(
                min_weight=node_min_size,
                max_weight=node_max_size,
                min_label_size=label_min_size,
                max_label_size=label_max_size,
                layout=graph_layout_global_centrality,
                avoid_centrality=not_centrality,
            )

        if pvis_graph:
            suffix = ".html"
            temp_file_name = next(tempfile._get_candidate_names()) + suffix
            full_filename = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'assets',temp_file_name))
            pvis_graph.write_html(full_filename)
            return html.Iframe(src=f"/assets/{temp_file_name}",style={"height": "100%", "width": "100%"})

    elif tab == "heatmap_global_cetrality":
        plotly_graph = None

        if not graph_filter:
            global_graph.set_edges_filter("is teacher of")
        else:
            global_graph.edges_filter = []
            filters = graph_filter
            for m_filter in filters:
                global_graph.set_edges_filter(m_filter)

        def round_list_values(list_in):
            return [round(value, 4) for value in list_in]

        calculated_degree = [round(value) for value in global_graph.calculate_degree()]
        calculated_betweenness = round_list_values(global_graph.calculate_betweenness())
        calculated_closeness = round_list_values(global_graph.calculate_closeness())
        calculated_eigenvector = round_list_values(global_graph.calculate_eigenvector())

        dict_global_data_tables ={
            "Phylosopher": global_graph.get_vertex_names(),
            "Degree": calculated_degree,
            "Betweeness": calculated_betweenness,
            "Closeness": calculated_betweenness,
            "Eigenvector": calculated_eigenvector 
        }

        interpolated_data = {
            "Phylosopher": dict_global_data_tables["Phylosopher"],
            "Degree": np.interp(
                dict_global_data_tables["Degree"], (min(dict_global_data_tables["Degree"]), max(dict_global_data_tables["Degree"])), (0, +1)
            ),
            "Betweeness": np.interp(
                dict_global_data_tables["Betweeness"],
                (min(dict_global_data_tables["Betweeness"]), max(dict_global_data_tables["Betweeness"])),
                (0, +1),
            ),
            "Closeness": np.interp(
                dict_global_data_tables["Closeness"],
                (min(dict_global_data_tables["Closeness"]), max(dict_global_data_tables["Closeness"])),
                (0, +1),
            ),
            "Eigenvector": np.interp(
                dict_global_data_tables["Eigenvector"],
                (min(dict_global_data_tables["Eigenvector"]), max(dict_global_data_tables["Eigenvector"])),
                (0, +1),
            ),
        }

        df_global_heatmap = pd.DataFrame(interpolated_data).sort_values(["Degree", "Betweeness", "Closeness"])
                            
        plotly_graph = go.Figure(
            data=go.Heatmap(
                z=df_global_heatmap[["Degree", "Betweeness", "Closeness", "Eigenvector"]],
                y=df_global_heatmap["Phylosopher"],
                x=["Degree", "Betweeness", "Closeness", "Eigenvector"],
                hoverongaps=False,
                type="heatmap",
                colorscale="Viridis",
            )
        )
        plotly_graph.update_layout(
            legend_font_size=12, legend_title_font_size=12, font_size=8
        )
        
        return html.Div([dcc.Graph(figure=plotly_graph, style={"height": "100%", "width": "100%"})], style={"height": "100%", "width": "100%"})

    elif tab == "metrics_global_cetrality":
        if not graph_filter:
            global_graph.set_edges_filter("is teacher of")
        else:
            global_graph.edges_filter = []
            filters = graph_filter
            for m_filter in filters:
                global_graph.set_edges_filter(m_filter)
        
        def round_list_values(list_in):
            return [round(value, 4) for value in list_in]

        calculated_degree = [round(value) for value in global_graph.calculate_degree()]
        calculated_betweenness = round_list_values(global_graph.calculate_betweenness())
        calculated_closeness = round_list_values(global_graph.calculate_closeness())
        calculated_eigenvector = round_list_values(global_graph.calculate_eigenvector())

        dict_global_data_tables ={
            "Phylosopher": global_graph.get_vertex_names(),
            "Degree": calculated_degree,
            "Betweeness": calculated_betweenness,
            "Closeness": calculated_betweenness,
            "Eigenvector": calculated_eigenvector 
        }

        df_global_data_tables = pd.DataFrame(dict_global_data_tables)

        dt = dash_table.DataTable( 
            id='table-global-graph', 
            columns=[{"name": i, "id": i, 'deletable': True} for i in df_global_data_tables.columns],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            style_cell={'textAlign': 'center'}, 
            page_current=0,
            page_size=20,
            page_action='custom',
            sort_mode='single',
            sort_by=[{'column_id': 'Degree', 'direction': 'asc'}]
        )
        
        return [html.H6('Centrality Scores',className="mt-1 mb-2"), html.Hr(className='py-0'), dt]

@app.callback(
    Output('table-global-graph', 'data'),
    Input('table-global-graph', "page_current"),
    Input('table-global-graph', "page_size"),
    Input('table-global-graph', 'sort_by'),
    Input('dataset_selection_global_centrality', 'value'),
    Input('graph_filter_global_centrality', 'value'),
    Input('graph_layout_global_centrality', 'value'),
    Input('show_gender_global_centrality', 'value'),
    Input('centrality_index_global_centrality', 'value'),
    Input('label_size_global_centrality', 'value'),
    Input('node_size_global_centrality', 'value')
)
def update_table(
                page_current, 
                page_size, 
                sort_by,
                dataset_selection_global_centrality,
                graph_filter_global_centrality,
                graph_layout_global_centrality,
                show_gender_global_centrality,
                centrality_index_global_centrality,
                label_size_global_centrality,
                node_size_global_centrality
):
    global_graph = diogenetGraph(
                "global",
                dataset_selection_global_centrality,
                dataset_selection_global_centrality,
                'locations_data.csv',
                'travels_blacklist.csv'
            )

    plot_type = "pyvis"
    centrality_index = centrality_index_global_centrality
    node_min_size = int(label_size_global_centrality[0])
    node_max_size = int(label_size_global_centrality[1])
    label_min_size = int(node_size_global_centrality[0])
    label_max_size = int(node_size_global_centrality[1])
    global_graph.current_centrality_index = "Degree"
    not_centrality = False
    graph_layout = graph_layout_global_centrality
    graph_filter = graph_filter_global_centrality


    if not graph_filter:
            global_graph.set_edges_filter("is teacher of")
    else:
        global_graph.edges_filter = []
        filters = graph_filter
        for m_filter in filters:
            global_graph.set_edges_filter(m_filter)
        
    def round_list_values(list_in):
        return [round(value, 4) for value in list_in]

    calculated_degree = [round(value) for value in global_graph.calculate_degree()]
    calculated_betweenness = round_list_values(global_graph.calculate_betweenness())
    calculated_closeness = round_list_values(global_graph.calculate_closeness())
    calculated_eigenvector = round_list_values(global_graph.calculate_eigenvector())

    dict_global_data_tables ={
        "Phylosopher": global_graph.get_vertex_names(),
        "Degree": calculated_degree,
        "Betweeness": calculated_betweenness,
        "Closeness": calculated_betweenness,
        "Eigenvector": calculated_eigenvector 
    }

    df_global_data_tables = pd.DataFrame(dict_global_data_tables)
    
    print(sort_by)
    if len(sort_by):
        dff = df_global_data_tables.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'desc',
            inplace=False
        )
    else:
        # No sort is applied
        dff = df_global_data_tables

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')

@app.callback(
    Output("download-dataframe-csv-global", "data"),
    Input('dataset_selection_global_centrality', 'value'),
    Input("btn_csv_global", "n_clicks"),
    prevent_initial_call=True,
)
def func(dataset_selection, n_clicks):
    if dataset_selection == 'diogenes':
        full_filename_csv = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data','new_Edges.csv'))
    elif dataset_selection == 'iamblichus':
        full_filename_csv = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data','new_Edges_Life_of_Pythagoras_Iamblichus.csv'))
    
    if n_clicks is None:
        raise PreventUpdate
    else:
        print(full_filename_csv)
        df = pd.read_csv(full_filename_csv)
        print(df.head())
        return dcc.send_data_frame(df.to_csv, 'edges.csv')


@app.callback(Output('confirm-warning-tie-centrality', 'displayed'),
              Input('graph_filter_global_centrality', 'value'))
def display_confirm(graph_filter_global_centrality):
    if len(graph_filter_global_centrality) == 0:
        return True
    return False
