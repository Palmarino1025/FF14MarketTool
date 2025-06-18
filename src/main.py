import json
import os
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from joblib import load
import pandas as pd
import plotly.graph_objs as go
from DataAquisition import fetch_and_save_item_data, fetch_top_sales_data, train_and_save_model
from prediction_util import predict_next_price_from_model



DC_WORLDS = {
    "Aether": ["Adamantoise", "Cactuar", "Faerie", "Gilgamesh", "Jenova", "Midgardsormr", "Sargatanas", "Siren"],
    "Crystal": ["Balmung", "Brynhildr", "Coeurl", "Diabolos", "Goblin", "Malboro", "Mateus", "Zalera"],
    "Primal": ["Behemoth", "Excalibur", "Exodus", "Famfrit", "Hyperion", "Lamia", "Leviathan", "Ultros"],
    "Dynamis": ["Halicarnassus", "Maduin", "Marilith", "Seraph"]
}
item_data = {}
model = load("linear_regression_model.joblib")

def load_item_data():
    global item_data
    file_path = os.path.join(os.path.dirname(__file__), "items.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            item_data = json.load(f)
        print(f"Loaded {len(item_data)} items from {file_path}")
    else:
        print(f"items.json not found at: {file_path}")
        item_data = {}

def get_item_id_from_name(name):
    global item_data
    load_item_data()
    for item_name, item_id in item_data.items():
        if item_name.lower() == name.lower():
            return int(item_id)
    return None

def run_dash_app():
    global item_data
    load_item_data()


    app = dash.Dash(__name__)
    app.title = "FFXIV Market Tool"

    # Prepare dropdown options (first 100 items)
    def get_dropdown_options(filter_value=""):
        filtered = list(item_data.keys())
        return [{"label": name, "value": name} for name in filtered]

    app.layout = html.Div([
        html.H1("FFXIV Item Lookup"),

        html.Div([
            # Data Center dropdown (left)
            html.Div([
                html.Label("Choose your Data Center:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="dc-dropdown",
                    options=[{"label": dc, "value": dc} for dc in DC_WORLDS.keys()],
                    value="Aether",
                    placeholder="Select Data Center",
                    style={"width": "200px"}
                )
            ], style={"flex": "1", "minWidth": "220px", "marginRight": "20px"}),

            # Item name dropdown + lookup button (right)
            html.Div([
                html.Label("Select Item:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                html.Div([
                    dcc.Dropdown(
                        id="item-name-dropdown",
                        options=get_dropdown_options(),
                        placeholder="Start typing item name...",
                        searchable=True,
                        clearable=True,
                        style={"width": "300px"}
                    ),
                    html.Button("Lookup Item ID", id="lookup-btn", n_clicks=0, style={"marginLeft": "10px", "height": "38px"})
                ], style={"display": "flex", "alignItems": "center"})
            ], style={"flex": "2", "minWidth": "350px", "marginLeft": "40px"})
        ],
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "flex-start",
            "maxWidth": "900px",
            "margin": "auto",
            "paddingBottom": "20px"
        }),

        html.Div(id="item-id-output", style={"margin-right": "20px", "font-weight": "bold"}),

        html.Div(id="summary-container", style={"marginTop": "20px"}),

        # Sales graph container
        html.Div(id="sales-output", style={"margin-top": "20px"}),

        html.Hr(),

        html.Button("Update Item List", id="update-btn", n_clicks=0),
        html.Div(id="update-status", style={"margin-top": "10px", "color": "green"})
    ])

    def get_sales_by_worlds(worlds, item_id):
        rows = []
        current_row = []
        top_servers = []

        # Puts top three servers with the lowest price to by the item
        for world in worlds:
            try:
                item_df = fetch_top_sales_data(world, item_id, sales_limit=1000)
                if not item_df.empty:
                    current_price = item_df['Price'].iloc[-1]  # most recent
                    top_servers.append((world, current_price))
            except Exception as e:
                print(f"[Error] Fetching for {world}: {e}")

        # Sort by price (ascending) and get top 3
        top_servers = sorted(top_servers, key=lambda x: x[1])[:3]

        # Create summary HTML block
        summary_block = html.Div(
            [
                html.H3("Top 3 Servers (Lowest Current Price)", style={"textAlign": "center"}),
                html.Ul([
                    html.Li(f"{server}: {price:,} gil") for server, price in top_servers
                ])
            ],
            style={
                "padding": "10px",
                "marginBottom": "20px",
                "backgroundColor": "#e6f7ff",
                "borderRadius": "8px",
                "textAlign": "center",
                "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)"
            }
        )


        for i, world in enumerate(worlds):
            try:
                item_df = fetch_top_sales_data(world, item_id, sales_limit=1000)
                print(f"{world} - Fetched sales: {len(item_df)}")
                train_and_save_model(world, item_id)
                if item_df.empty:
                    graph = html.Div(f"No sales found for {world}")
                    stats_div = html.Div()
                else:
                    # Create price trend graph
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(item_df['Timestamp'], unit='s'),
                        y=item_df['Price'],
                        mode='lines+markers',
                        name=world
                    ))
                    fig.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Price (gil)",
                        height=300,
                        margin=dict(l=20, r=20, t=10, b=20)
                    )
                    graph = dcc.Graph(figure=fig)

                    # Stats
                    min_price = item_df['Price'].min()
                    max_price = item_df['Price'].max()
                    current_price = item_df['Price'].iloc[-1]

                    # Predict next price using the pre-trained model
                    try:
                        predicted_price = predict_next_price_from_model(item_df)
                        predicted_text = f" | Predicted Next: {predicted_price:,.2f}"
                        print(predicted_price)
                    except Exception as e:
                        predicted_text = " | Predicted Next: (error)"

                    stats_text = (
                        f"High: {max_price:,} | Low: {min_price:,} | "
                        f"Current: {current_price:,}{predicted_text}"
                    )

                    stats_div = html.Div(
                        stats_text,
                        style={
                            "textAlign": "center",
                            "fontWeight": "bold",
                            "marginTop": "6px"
                        }
                    )

            except Exception as e:
                graph = html.Div(f"Error for {world}: {str(e)}", style={"color": "red"})
                stats_div = html.Div()

            block = html.Div(
                [
                    html.H4(world, style={
                        "textAlign": "center",
                        "fontWeight": "bold",
                        "fontSize": "20px",
                        "marginBottom": "10px",
                        "color": "#2a2a2a"
                    }),
                    graph,
                    stats_div
                ],
                style={
                    "flex": "1",
                    "minWidth": "300px",
                    "maxWidth": "33%",
                    "padding": "10px",
                    "boxSizing": "border-box",
                    "border": "1px solid #ccc",
                    "borderRadius": "8px",
                    "backgroundColor": "#f9f9f9"
                }
            )

            current_row.append(block)

            if (i + 1) % 3 == 0 or i == len(worlds) - 1:
                rows.append(html.Div(current_row, style={
                    "display": "flex",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "20px"
                }))
                current_row = []

        return rows

    @app.callback(
        Output("item-id-output", "children"),
        Output("summary-container", "children"),
        Output("sales-output", "children"),
        Input("lookup-btn", "n_clicks"),
        State("dc-dropdown", "value"),
        State("item-name-dropdown", "value")
    )
    def update_all_outputs(n_clicks, selected_dc, selected_item_name):
        if n_clicks == 0:
            return "", "", ""

        if not selected_item_name:
            return "Please select an item name.", "", ""

        if not selected_dc or selected_dc not in DC_WORLDS:
            return "Please select a valid data center.", "", ""

        item_id = get_item_id_from_name(selected_item_name)
        if not item_id:
            return "Item not found. Try updating the list or check spelling.", "", ""

        # Show item ID output
        item_id_output = f"Item ID for '{selected_item_name}': {item_id}"

        # Get world list
        worlds = DC_WORLDS[selected_dc]

        # Sales Graphs
        graph_blocks = get_sales_by_worlds(worlds, item_id)

        # Build summary block
        current_prices = []
        for world in worlds:
            df = fetch_top_sales_data(world, item_id)
            if not df.empty:
                current_price = df["Price"].iloc[-1]
                current_prices.append((world, current_price))

        top_servers = sorted(current_prices, key=lambda x: x[1])[:3]

        summary_block = html.Div(
            [
                html.H3("Top 3 Servers (Lowest Current Price)", style={"textAlign": "center"}),
                html.Ul([
                    html.Li(f"{server}: {price:,} gil") for server, price in top_servers
                ])
            ],
            style={
                "padding": "10px",
                "marginBottom": "20px",
                "backgroundColor": "#e6f7ff",
                "borderRadius": "8px",
                "textAlign": "center",
                "boxShadow": "0px 2px 6px rgba(0,0,0,0.1)"
            }
        )

        return item_id_output, summary_block, html.Div(graph_blocks)

    
    @app.callback(
        Output("update-status", "children"),
        Input("update-btn", "n_clicks")
    )
    def update_item_list(n_clicks):
        global item_data
        if n_clicks > 0:
            # fetch_and_save_item_data()
            load_item_data()
            return "Item list updated successfully."
        return ""

    #app.run_server(debug=True)
    app.run(debug=True)

if __name__ == "__main__":
    run_dash_app()