import json
import os
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from joblib import load
import pandas as pd
import plotly.graph_objs as go
from DataAquisition import fetch_and_save_item_data, fetch_top_sales_data
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
            ], style={"flex": "1", "minWidth": "220px", "marginRight": "20px"}),  # Added marginRight for spacing

            # World dropdown (next to Data Center)
            html.Div([
                html.Label("Choose your World:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                dcc.Dropdown(
                    id="world-dropdown",
                    options=[{"label": w, "value": w} for w in DC_WORLDS["Aether"]],  # Default to Aether worlds initially
                    value=DC_WORLDS["Aether"][0],
                    placeholder="Select World",
                    style={"width": "200px"}
                )
            ], style={"flex": "1", "minWidth": "220px"}),

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
            "maxWidth": "900px",  # Increased to accommodate 3 dropdowns
            "margin": "auto",
            "paddingBottom": "20px"
        }),

        html.Div(id="item-id-output", style={"margin-right": "20px", "font-weight": "bold"}),

        html.Div(id="sales-output", style={"margin-top": "20px"}),

        html.Hr(),

        html.Button("Update Item List", id="update-btn", n_clicks=0),
        html.Div(id="update-status", style={"margin-top": "10px", "color": "green"})
    ])

    def get_sales_by_worlds(worlds, item_id):
        rows = []
        current_row = []

        for i, world in enumerate(worlds):
            try:
                item_df = fetch_top_sales_data(world, item_id, sales_limit=300)

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
                        predicted_price = predict_next_price_from_model(item_df, model)
                        predicted_text = f" | Predicted Next: {predicted_price:,.2f}"
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


    
    #Callback for item lookup and fetching sales
    @app.callback(
        [Output("item-id-output", "children"),
         Output("sales-output", "children")],
        Input("lookup-btn", "n_clicks"),
        State("item-name-dropdown", "value"),
        State("dc-dropdown", "value")

    )
    def lookup_item_id(n_clicks, item_name, selected_dc):
        if n_clicks == 0:
            return "", ""

        if not item_name:
            return "Please select an item name.", ""

        if not selected_dc or selected_dc not in DC_WORLDS:
            return "Please select a valid data center.", ""

        item_id = item_data.get(item_name)
        if not item_id:
            return "Item not found. Try updating the list or check spelling.", ""

        worlds_to_query = DC_WORLDS[selected_dc]
        graphs = get_sales_by_worlds(worlds_to_query, item_id)

        return f"Item ID for '{item_name}': {item_id}", html.Div(graphs)
    
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

    @app.callback(
        Output("world-dropdown", "options"),
        Output("world-dropdown", "value"),
        Input("dc-dropdown", "value")
    )
    def update_worlds(selected_dc):
        worlds = DC_WORLDS.get(selected_dc, [])
        options = [{"label": w, "value": w} for w in worlds]
        value = worlds[0] if worlds else None
        return options, value

    app.run_server(debug=True)

# For Model Training/Data gathering purposes
def main():
    print("Welcome to the FFXIV Market Tool")

    while True:
        print("\nOptions:")
        print("1. Update item list from XIVAPI")
        print("2. Train or update model")
        print("3. Launch app server")
        print("4. Exit")

        choice = input("Enter your choice: ")

        match choice:
            case "1":
                fetch_and_save_item_data()
            case "2":
                # Load items.json
                items_path = os.path.join(os.path.dirname(__file__), "items.json")
                with open(items_path, "r", encoding="utf-8") as f:
                    items = json.load(f)
                item_ids = list(items.values())
                # Choose a server/world
                server = "Leviathan"  # or prompt user for input
                # Fetch prices for all items
                df = fetch_top_sales_data(world, top_n=100, sales_limit=50)
                if not df or len(df) < 10:
                    print("Not enough price data to train.")
                else:
                    train_linear_model(df)
                    print("Model training/updating complete.")
            case "3":
                run_dash_app()
            case "4":
                print("Goodbye!")
                break
            case _:
                print("Invalid choice. Try again.")

if __name__ == "__main__":
    run_dash_app()