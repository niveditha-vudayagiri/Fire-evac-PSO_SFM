# simulation/dashboard.py

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
import dash_bootstrap_components as dbc
# --- Statistics and Progress Graph update callback ---
from dash.dependencies import Output as DashOutput, Input as DashInput
import plotly.graph_objs as go
from param_config import N_PEDESTRIANS, MAX_ITERATIONS

# Shared data — this will be updated live from simulation
ALGORITHM_KEYS = [
    'Random',
    'PSO', 
    'SFM',
    'PSO SFM(Staff all evacuate)',
    'PSO SFM(Staff all assist)',
    'PSO SFM(Half Assist)',
    'PSO SFM(Assist Mobile)',
    'PSO SFM(Assist Elderly)',
    'PSO SFM(Top Evac)',
    'PSO SFM(Avoid Top)',
    'PSO SFM(Zone Sweep)',
    'MFO SFM',
    'ACO SFM'
]

live_data = {
    algo: {
        'evacuated': [],
        'deaths': [],
        'deaths_per_floor': [],
        'evacuated_by_staff_type': [],  # List to track staff vs non-staff evacuations
        'staff_type_totals': {'staff': 0, 'civilian': 0},  # Total counts for each type
        # Add more per-algorithm stats here as needed
    } for algo in ALGORITHM_KEYS
}
live_data.update({
    "timestamps": [],
    "total_people": N_PEDESTRIANS,
    "current_time": 0
})

def create_dashboard(simulations):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Crowd Evacuation Simulation Dashboard"

    app.layout = dbc.Container([
        html.H2("Crowd Evacuation Simulation Dashboard", className="text-center my-4"),
        dbc.Row([
            dbc.Col(html.Div(id='simulation-status'), width=12),
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("Current Statistics"),
                html.Div(id="live-stats"),
                dcc.Graph(id="evacuation-progress-graph"),
                dcc.Graph(id="evacuation-progress-graph-other"),
                html.Button("Download CSV", id="download-csv-btn", n_clicks=0, className="btn btn-success my-2"),
                dcc.Download(id="download-csv"),
                dcc.Interval(id='update-interval', interval=5000, n_intervals=0)
            ], width=12)
        ])
    ], fluid=True)
    import io
    import csv
    import base64

    @app.callback(
        Output("download-csv", "data"),
        [Input("download-csv-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def download_csv(n_clicks):
        import io
        import csv
        from datetime import datetime, timedelta
        output = io.StringIO()
        # Use the export_live_data_csv logic but write to StringIO
        # Determine all unique floors, age groups, and staff types present in the data
        all_floors = set()
        all_age_groups = set()
        all_staff_types = set()
        for algo in live_data:
            if algo in ("timestamps", "total_people", "current_time"): continue
            for deaths_per_floor in live_data[algo].get("deaths_per_floor", []):
                all_floors.update(deaths_per_floor.keys())
            for evac_by_age in live_data[algo].get("evacuated_by_age", []):
                all_age_groups.update(evac_by_age.keys())
            for evac_by_staff in live_data[algo].get("evacuated_by_staff_type", []):
                all_staff_types.update(evac_by_staff.keys())
        all_floors = sorted(all_floors)
        all_age_groups = sorted(all_age_groups)
        all_staff_types = sorted(all_staff_types)

        # Prepare header
        fieldnames = ["timestamp", "algorithm", "evacuated", "deaths"]
        fieldnames += [f"deaths_floor_{f}" for f in all_floors]
        fieldnames += [f"evacuated_age_{a}" for a in all_age_groups]
        fieldnames += [f"evacuated_stafftype_{s}" for s in all_staff_types]

        # Find max length
        max_len = max(
            max(len(live_data[algo]["evacuated"]) for algo in live_data if algo not in ("timestamps", "total_people", "current_time")),
            len(live_data["timestamps"])
        )
        timestamps = list(live_data["timestamps"])
        while len(timestamps) < max_len:
            timestamps.append(len(timestamps))

        # Assume simulation starts at a fixed datetime (e.g., 2025-01-01 00:00:00)
        sim_start = datetime(2025, 1, 1, 0, 0, 0)
        writer = csv.DictWriter(output, fieldnames=fieldnames + ["datetime"])
        writer.writeheader()
        for i in range(max_len):
            t = timestamps[i] if i < len(timestamps) else i
            dt_str = (sim_start + timedelta(seconds=t)).isoformat()
            for algo in live_data:
                if algo in ("timestamps", "total_people", "current_time"): continue
                row = {"timestamp": t, "algorithm": algo, "datetime": dt_str}
                # Evacuated and deaths
                evac_list = live_data[algo]["evacuated"] if algo in live_data else []
                deaths_list = live_data[algo]["deaths"] if algo in live_data else []
                row["evacuated"] = evac_list[i] if i < len(evac_list) else 0
                row["deaths"] = deaths_list[i] if i < len(deaths_list) else 0
                # Deaths per floor
                deaths_per_floor = live_data[algo].get("deaths_per_floor", [])
                deaths_floor = deaths_per_floor[i] if i < len(deaths_per_floor) else {}
                for f in all_floors:
                    row[f"deaths_floor_{f}"] = deaths_floor.get(f, 0)
                # Evacuated by age
                evac_by_age = live_data[algo].get("evacuated_by_age", [])
                evac_age = evac_by_age[i] if i < len(evac_by_age) else {}
                for a in all_age_groups:
                    row[f"evacuated_age_{a}"] = evac_age.get(a, 0)
                # Evacuated by staff type
                evac_by_staff = live_data[algo].get("evacuated_by_staff_type", [])
                evac_staff = evac_by_staff[i] if i < len(evac_by_staff) else {}
                for s in all_staff_types:
                    row[f"evacuated_stafftype_{s}"] = evac_staff.get(s, 0)
                writer.writerow(row)
        return dict(content=output.getvalue(), filename="evacuation_stats_detailed.csv")

    def sim_layout(idx, title):
        sim = simulations[idx]
        return html.Div([
            html.H2(title),
            html.P(f"Visualization available at: {sim.visualization_url}"),
            html.A(
                "Open Visualization",
                href=sim.visualization_url,
                target="_blank",
                style={
                    "padding": "10px",
                    "backgroundColor": "#4CAF50",
                    "color": "white",
                    "textDecoration": "none",
                    "display": "inline-block",
                    "margin": "10px"
                }
            )
        ])


    @app.callback(
        [DashOutput("live-stats", "children"), 
         DashOutput("evacuation-progress-graph", "figure"), 
         DashOutput("evacuation-progress-graph-other", "figure")],
        [DashInput("update-interval", "n_intervals")]
    )
    def update_stats_and_graph(n):
        # Synchronous update: if data is not ready, do not update any output
        if not live_data:
            import dash
            return dash.no_update, dash.no_update, dash.no_update
        # Compute max evacuated for each algorithm
        evacuated = {algo: max(live_data[algo]["evacuated"]) if live_data[algo]["evacuated"] else 0 for algo in ALGORITHM_KEYS}
        #Advanced analytics
        # Show deaths only when simulation is finished (i.e., last timestamp has all people accounted for)
        show_deaths = len(live_data["timestamps"]) == MAX_ITERATIONS

        stats = [
            html.Div(
                " | ".join([
                    f"{algo.replace('_', '+').upper() if '+' in algo else algo.upper()}: {evacuated[algo]} / {live_data['total_people']}" for algo in ALGORITHM_KEYS
                ]) + f" | Time: {live_data['current_time']}s"
            )
        ]
        # Always prepare table data for DataTable
        table_data = []
        for algo in ALGORITHM_KEYS:
            algo_name = algo.replace('_', '+').upper() if '+' in algo else algo.upper()
            evac = evacuated[algo]
            deaths = live_data[algo]['deaths'][-1] if live_data[algo]['deaths'] else 0
            if live_data[algo]['deaths_per_floor']:
                last = live_data[algo]['deaths_per_floor'][-1]
                floor_str = ", ".join([f"Floor {floor}: {count}" for floor, count in sorted(last.items())])
            else:
                floor_str = "No data"
            if 'evacuated_by_age' in live_data[algo] and live_data[algo]['evacuated_by_age']:
                last_age = live_data[algo]['evacuated_by_age'][-1]
                age_totals = live_data[algo].get('age_group_totals', {})
                age_str = ", ".join([
                    f"{age}: {last_age.get(age, 0)}/{age_totals.get(age, 0)}" for age in sorted(set(list(last_age.keys()) + list(age_totals.keys())))
                ])
            else:
                age_str = "No data"
                
            # Add staff vs non-staff evacuation data
            if 'evacuated_by_staff_type' in live_data[algo] and live_data[algo]['evacuated_by_staff_type']:
                last_staff = live_data[algo]['evacuated_by_staff_type'][-1]
                staff_totals = live_data[algo].get('staff_type_totals', {})
                staff_str = f"Staff: {last_staff.get('staff', 0)}/{staff_totals.get('staff', 0)}, "
                staff_str += f"Civilian: {last_staff.get('civilian', 0)}/{staff_totals.get('civilian', 0)}"
            else:
                staff_str = "No data"
                
            table_data.append({
                "Algorithm": algo_name,
                "Evacuated": f"{evac} / {live_data['total_people']}",
                "Deaths": str(deaths),
                "Deaths by Floor": floor_str,
                "Evacuated by Age": age_str,
                "Staff vs Civilian": staff_str
            })
        stats.append(dash_table.DataTable(
            columns=[
                {"name": "Algorithm", "id": "Algorithm", "type": "text"},
                {"name": "Evacuated", "id": "Evacuated", "type": "text"},
                {"name": "Deaths", "id": "Deaths", "type": "numeric"},
                {"name": "Deaths by Floor", "id": "Deaths by Floor", "type": "text"},
                {"name": "Evacuated by Age", "id": "Evacuated by Age", "type": "text"},
                {"name": "Staff vs Civilian", "id": "Staff vs Civilian", "type": "text"}
            ],
            data=table_data,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto", "marginTop": "20px"},
            style_cell={"textAlign": "center", "minWidth": "120px", "maxWidth": "300px", "whiteSpace": "normal"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
            style_data={"backgroundColor": "#fdfdfd"},
            page_size=12
        ))
        
        # Prepare evacuation progress graphs
        # Split algorithms into PSO SFM variants and others
        pso_sfm_algos = [algo for algo in ALGORITHM_KEYS if algo.startswith('PSO SFM')]
        other_algos = [algo for algo in ALGORITHM_KEYS if algo not in pso_sfm_algos]

        # PSO SFM chart
        traces_pso_sfm = []
        for algo in pso_sfm_algos:
            evac_list = live_data[algo]["evacuated"] if algo in live_data else []
            if evac_list:
                traces_pso_sfm.append(go.Scatter(
                    x=live_data["timestamps"][:len(evac_list)],
                    y=evac_list,
                    mode='lines+markers',
                    name=algo
                ))
        fig_pso_sfm = go.Figure(data=traces_pso_sfm)
        fig_pso_sfm.update_layout(title='Evacuation Progress (PSO SFM Variants)', xaxis_title='Time (s)', yaxis_title='Evacuated', legend_title='Algorithm', height=400)

        # Other algorithms chart
        traces_other = []
        for algo in other_algos:
            evac_list = live_data[algo]["evacuated"] if algo in live_data else []
            if evac_list:
                traces_other.append(go.Scatter(
                    x=live_data["timestamps"][:len(evac_list)],
                    y=evac_list,
                    mode='lines+markers',
                    name=algo
                ))
        fig_other = go.Figure(data=traces_other)
        fig_other.update_layout(title='Evacuation Progress (Other Algorithms)', xaxis_title='Time (s)', yaxis_title='Evacuated', legend_title='Algorithm', height=400)

        # --- New: Time series bar/line charts for selected columns ---
        algo_charts = []
        for selected_algo in ALGORITHM_KEYS:
            # 1. Deaths by Floor over time (stacked bar)
            deaths_by_floor_traces = []
            floors = set()
            for deaths_per_floor in live_data[selected_algo].get('deaths_per_floor', []):
                floors.update(deaths_per_floor.keys())
            floors = sorted(floors)
            for floor in floors:
                y = []
                deaths_list = live_data[selected_algo].get('deaths_per_floor', [])
                for t in range(len(live_data['timestamps'])):
                    if t < len(deaths_list):
                        y.append(deaths_list[t].get(floor, 0))
                    else:
                        y.append(0)
                deaths_by_floor_traces.append(go.Bar(
                    x=live_data['timestamps'][:len(y)],
                    y=y,
                    name=f'Floor {floor}'
                ))
            fig_deaths_by_floor = go.Figure(data=deaths_by_floor_traces)
            fig_deaths_by_floor.update_layout(barmode='stack', title=f'Deaths by Floor Over Time ({selected_algo})', xaxis_title='Time (s)', yaxis_title='Deaths', height=250)

            # 2. Evacuated by Age over time (stacked bar)
            age_groups = set()
            for evac_by_age in live_data[selected_algo].get('evacuated_by_age', []):
                age_groups.update(evac_by_age.keys())
            age_groups = sorted(age_groups)
            evac_by_age_traces = []
            for age in age_groups:
                y = []
                evac_list = live_data[selected_algo].get('evacuated_by_age', [])
                for t in range(len(live_data['timestamps'])):
                    if t < len(evac_list):
                        y.append(evac_list[t].get(age, 0))
                    else:
                        y.append(0)
                evac_by_age_traces.append(go.Bar(
                    x=live_data['timestamps'][:len(y)],
                    y=y,
                    name=f'Age {age}'
                ))
            fig_evac_by_age = go.Figure(data=evac_by_age_traces)
            fig_evac_by_age.update_layout(barmode='stack', title=f'Evacuated by Age Over Time ({selected_algo})', xaxis_title='Time (s)', yaxis_title='Evacuated', height=250)

            # 3. Staff vs Civilian evacuation over time (grouped bar)
            staff_types = ['staff', 'civilian']
            staff_evac_traces = []
            evac_list = live_data[selected_algo].get('evacuated_by_staff_type', [])
            for stype in staff_types:
                y = []
                for t in range(len(live_data['timestamps'])):
                    if t < len(evac_list):
                        y.append(evac_list[t].get(stype, 0))
                    else:
                        y.append(0)
                staff_evac_traces.append(go.Bar(
                    x=live_data['timestamps'][:len(y)],
                    y=y,
                    name=stype.capitalize()
                ))
            fig_staff_evac = go.Figure(data=staff_evac_traces)
            fig_staff_evac.update_layout(barmode='group', title=f'Staff vs Civilian Evacuation Over Time ({selected_algo})', xaxis_title='Time (s)', yaxis_title='Evacuated', height=250)

            algo_charts.extend([
                html.Hr(),
                html.H4(f"{selected_algo}"),
                dcc.Graph(figure=fig_deaths_by_floor),
                dcc.Graph(figure=fig_evac_by_age),
                dcc.Graph(figure=fig_staff_evac)
            ])
        # Return all charts in the dashboard
        return stats + algo_charts, fig_pso_sfm, fig_other

    import csv

    def export_live_data_csv(live_data, csv_path):
        # Determine all unique floors, age groups, and staff types present in the data
        all_floors = set()
        all_age_groups = set()
        all_staff_types = set()
        for algo in live_data:
            if algo in ("timestamps", "total_people", "current_time"): continue
            for deaths_per_floor in live_data[algo].get("deaths_per_floor", []):
                all_floors.update(deaths_per_floor.keys())
            for evac_by_age in live_data[algo].get("evacuated_by_age", []):
                all_age_groups.update(evac_by_age.keys())
            for evac_by_staff in live_data[algo].get("evacuated_by_staff_type", []):
                all_staff_types.update(evac_by_staff.keys())
        all_floors = sorted(all_floors)
        all_age_groups = sorted(all_age_groups)
        all_staff_types = sorted(all_staff_types)

        # Prepare header
        fieldnames = ["timestamp", "algorithm", "evacuated", "deaths"]
        fieldnames += [f"deaths_floor_{f}" for f in all_floors]
        fieldnames += [f"evacuated_age_{a}" for a in all_age_groups]
        fieldnames += [f"evacuated_stafftype_{s}" for s in all_staff_types]

        # Find max length
        max_len = max(
            max(len(live_data[algo]["evacuated"]) for algo in live_data if algo not in ("timestamps", "total_people", "current_time")),
            len(live_data["timestamps"])
        )
        timestamps = list(live_data["timestamps"])
        while len(timestamps) < max_len:
            timestamps.append(len(timestamps))

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(max_len):
                t = timestamps[i] if i < len(timestamps) else i
                for algo in live_data:
                    if algo in ("timestamps", "total_people", "current_time"): continue
                    row = {"timestamp": t, "algorithm": algo}
                    # Evacuated and deaths
                    evac_list = live_data[algo]["evacuated"] if algo in live_data else []
                    deaths_list = live_data[algo]["deaths"] if algo in live_data else []
                    row["evacuated"] = evac_list[i] if i < len(evac_list) else 0
                    row["deaths"] = deaths_list[i] if i < len(deaths_list) else 0
                    # Deaths per floor
                    deaths_per_floor = live_data[algo].get("deaths_per_floor", [])
                    deaths_floor = deaths_per_floor[i] if i < len(deaths_per_floor) else {}
                    for f in all_floors:
                        row[f"deaths_floor_{f}"] = deaths_floor.get(f, 0)
                    # Evacuated by age
                    evac_by_age = live_data[algo].get("evacuated_by_age", [])
                    evac_age = evac_by_age[i] if i < len(evac_by_age) else {}
                    for a in all_age_groups:
                        row[f"evacuated_age_{a}"] = evac_age.get(a, 0)
                    # Evacuated by staff type
                    evac_by_staff = live_data[algo].get("evacuated_by_staff_type", [])
                    evac_staff = evac_by_staff[i] if i < len(evac_by_staff) else {}
                    for s in all_staff_types:
                        row[f"evacuated_stafftype_{s}"] = evac_staff.get(s, 0)
                    writer.writerow(row)
    return app

