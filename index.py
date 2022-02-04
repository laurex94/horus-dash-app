from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import global_network_graph, global_network_graph_centrality

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="/apps/global_network_graph"), id="network-graph-link"),
        dbc.NavItem(dbc.NavLink("Page 2", href="/apps/global_network_graph_centrality"), id="global-network-graph-centrality-link")
    ],
    brand="Home",
    brand_href="/",
    color="primary",
    dark=True,
)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

main_layout = html.Div([
    navbar,
    html.H3('Home'),
    dcc.Link('Go to Global Network', href='/apps/global_network_graph'),
    dcc.Link('Go to App 2', href='/apps/global_network_graph_centrality'),
])

# Update the index
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/global_network_graph':
        return global_network_graph.layout
    elif pathname == '/apps/global_network_graph_centrality':
        return global_network_graph_centrality.layout
    elif pathname == '/':
        return global_network_graph.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)