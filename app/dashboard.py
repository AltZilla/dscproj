"""
Plotly Dash Dashboard — Interactive Energy Optimizer
=====================================================
Users configure occupancy, electricity rates, and appliance schedule,
then run live DE optimization to compare their schedule vs optimized.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APPLIANCE_NAMES = [
    "hvac", "washer", "dryer", "dishwasher",
    "ev_charger", "lighting", "water_heater"
]
APPLIANCE_LABELS = {
    "hvac": "❄ HVAC", "washer": "🫧 Washer", "dryer": "🌀 Dryer",
    "dishwasher": "🍽 Dishwasher", "ev_charger": "🔌 EV Charger",
    "lighting": "💡 Lighting", "water_heater": "🔥 Water Heater",
}
RATED_POWERS = {
    "hvac": 3.5, "washer": 0.5, "dryer": 2.0,
    "dishwasher": 1.8, "ev_charger": 7.0,
    "lighting": 0.3, "water_heater": 4.5,
}

ROOMS = {
    "Living Room":  (2.0, 3.0, 3.5, 2.5),
    "Kitchen":      (5.5, 3.0, 2.5, 2.5),
    "Bedroom":      (2.0, 0.75, 3.5, 1.5),
    "Bathroom":     (5.5, 0.75, 2.5, 1.0),
    "Garage":       (5.5, 1.75, 2.5, 1.0),
}
APPLIANCE_LOCATIONS = {
    "hvac":         (1.0, 3.8, "❄ HVAC"),
    "lighting":     (3.0, 3.8, "💡 Lights"),
    "washer":       (5.0, 0.9, "🫧 Wash"),
    "dryer":        (6.0, 0.9, "🌀 Dry"),
    "dishwasher":   (5.0, 3.0, "🍽 Dish"),
    "water_heater": (6.5, 3.0, "🔥 Heat"),
    "ev_charger":   (5.5, 1.8, "🔌 EV"),
}

HOUR_LABELS = [f"{h%12 or 12}{'A' if h<12 else 'P'}" for h in range(24)]

# Default schedule: peak-heavy
DEFAULT_SCHEDULE = {
    "hvac":         [0]*6 + [0]*4 + [0]*4 + [0]*4 + [1]*4 + [1]*2,
    "washer":       [0]*6 + [0]*4 + [0]*4 + [0]*4 + [1]*4 + [0]*2,
    "dryer":        [0]*6 + [0]*4 + [0]*4 + [0]*4 + [1]*4 + [0]*2,
    "dishwasher":   [0]*6 + [0]*4 + [0]*4 + [0]*4 + [1]*4 + [0]*2,
    "ev_charger":   [0]*6 + [0]*4 + [0]*4 + [0]*4 + [1]*4 + [1]*2,
    "lighting":     [0]*6 + [0]*4 + [0]*4 + [0]*4 + [1]*4 + [1]*2,
    "water_heater": [0]*6 + [1]*2 + [0]*6 + [0]*4 + [1]*4 + [1]*2,
}

# Default occupancy
DEFAULT_OCCUPANCY = [0]*6 + [2]*2 + [0]*4 + [0]*4 + [1]*2 + [3]*4 + [2]*2

# Default rates
DEFAULT_RATES = {"off_peak": 2.0, "shoulder": 5.0, "peak": 12.0}
DEFAULT_PERIODS = {
    "off_peak": list(range(0, 6)) + list(range(22, 24)),
    "shoulder": list(range(6, 10)) + list(range(14, 18)),
    "peak": list(range(10, 14)) + list(range(18, 22)),
}


def hour_to_price(rates, periods):
    """Convert rate/period config to 24-length price array."""
    prices = np.zeros(24)
    for period_name, hours in periods.items():
        for h in hours:
            prices[h] = rates[period_name]
    return prices


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
def _card(content, **kwargs):
    return html.Div(content, style={
        "background": "rgba(108,99,255,0.06)",
        "border": "1px solid rgba(108,99,255,0.2)",
        "borderRadius": "12px", "padding": "15px",
        "marginBottom": "12px", **kwargs,
    })


def _section_title(text):
    return html.H6(text, style={
        "color": "#6c63ff", "fontWeight": "700",
        "marginBottom": "8px", "fontSize": "13px",
        "textTransform": "uppercase", "letterSpacing": "1px",
    })


def _metric_card(label, value, color, mid="metric"):
    return html.Div([
        html.Div(value, id=f"{mid}-val", style={
            "fontSize": "24px", "fontWeight": "700", "color": color,
        }),
        html.Div(label, style={
            "fontSize": "10px", "color": "#aaa",
            "textTransform": "uppercase", "letterSpacing": "1px",
        }),
    ], style={
        "background": "rgba(108,99,255,0.08)",
        "border": f"1px solid {color}33",
        "borderRadius": "12px", "padding": "12px", "textAlign": "center",
    })


# ---------------------------------------------------------------------------
# Floorplan
# ---------------------------------------------------------------------------
def draw_house(appliance_states, title=""):
    fig = go.Figure()
    for name, (cx, cy, w, h) in ROOMS.items():
        fig.add_shape(type="rect", x0=cx-w/2, y0=cy-h/2, x1=cx+w/2, y1=cy+h/2,
                      line=dict(color="#6c63ff", width=2),
                      fillcolor="rgba(108,99,255,0.05)")
        fig.add_annotation(x=cx, y=cy+h/2-0.15, text=f"<b>{name}</b>",
                           showarrow=False, font=dict(size=10, color="#666"))
    fig.add_shape(type="rect", x0=0, y0=0, x1=7, y1=4.5,
                  line=dict(color="#6c63ff", width=3))
    fig.add_shape(type="line", x0=3.5, y0=0, x1=4.2, y1=0,
                  line=dict(color="#ffd700", width=4))
    total_power = 0
    for app, (ax, ay, label) in APPLIANCE_LOCATIONS.items():
        power = appliance_states.get(app, 0)
        rated = RATED_POWERS.get(app, 1.0)
        if power > 0:
            r = min(power / max(rated, 0.1), 1.0)
            color = "#4caf50" if r < 0.3 else ("#ffc107" if r < 0.7 else "#f44336")
        else:
            color = "rgba(80,80,100,0.3)"
        total_power += power
        fig.add_shape(type="circle", x0=ax-0.28, y0=ay-0.28, x1=ax+0.28, y1=ay+0.28,
                      fillcolor=color, line=dict(color="white", width=1))
        txt = label + (f"<br><b>{power:.1f}kW</b>" if power > 0 else "<br>OFF")
        fig.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                           font=dict(size=8, color="white"))
    fig.update_layout(
        xaxis=dict(range=[-0.5,7.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5,5.2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        plot_bgcolor="rgba(15,12,41,0.95)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5,r=5,t=35,b=5), height=340,
        title=dict(text=title, font=dict(size=13, color="#6c63ff"), x=0.5),
    )
    return fig, total_power


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app():
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder=str(Path(__file__).parent / "assets"),
        suppress_callback_exceptions=True,
    )

    # -- Schedule grid --
    def make_schedule_grid():
        rows = []
        for app_name in APPLIANCE_NAMES:
            cells = [html.Td(APPLIANCE_LABELS[app_name], style={
                "fontSize": "11px", "color": "#ccc", "padding": "2px 6px",
                "whiteSpace": "nowrap", "fontWeight": "600",
            })]
            default = DEFAULT_SCHEDULE[app_name]
            for h in range(24):
                is_on = default[h]
                cells.append(html.Td(
                    html.Div(style={
                        "width": "18px", "height": "18px", "borderRadius": "4px",
                        "cursor": "pointer",
                        "background": "#4caf50" if is_on else "rgba(80,80,100,0.3)",
                        "border": "1px solid rgba(255,255,255,0.1)",
                        "transition": "all 0.15s",
                    }),
                    id={"type": "cell", "app": app_name, "hour": h},
                    n_clicks=0,
                    style={"padding": "1px", "textAlign": "center"},
                ))
            rows.append(html.Tr(cells))

        # Hour labels
        header = [html.Th("", style={"width": "95px"})]
        for h in range(24):
            header.append(html.Th(HOUR_LABELS[h], style={
                "fontSize": "8px", "color": "#888", "textAlign": "center",
                "padding": "1px", "fontWeight": "400",
            }))

        return html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)],
                         style={"borderCollapse": "collapse", "width": "100%"})

    # -- Occupancy bar --
    def make_occupancy_input():
        items = []
        for h in range(24):
            items.append(html.Div([
                html.Div(HOUR_LABELS[h], style={"fontSize": "7px", "color": "#888", "textAlign": "center"}),
                dcc.Input(
                    id={"type": "occ", "hour": h},
                    type="number", min=0, max=5,
                    value=DEFAULT_OCCUPANCY[h],
                    style={
                        "width": "28px", "height": "24px", "textAlign": "center",
                        "fontSize": "11px", "background": "rgba(108,99,255,0.1)",
                        "border": "1px solid rgba(108,99,255,0.3)",
                        "borderRadius": "4px", "color": "#fff", "padding": "0",
                    },
                ),
            ], style={"display": "inline-block", "margin": "0 1px"}))
        return html.Div(items, style={"overflowX": "auto", "whiteSpace": "nowrap"})

    # -- Price bar display (updates dynamically) --
    def make_price_period_bar():
        return html.Div(id="price-bar")

    # -- Layout --
    app.layout = dbc.Container([
        # Header
        html.H3("🏠 Smart Home Energy Optimizer",
                 className="text-center mt-3 mb-2",
                 style={"color": "#6c63ff", "fontWeight": "700"}),
        html.P("Configure your schedule, set rates, and optimize!",
               className="text-center mb-3", style={"color": "#888", "fontSize": "13px"}),

        # Metric cards
        dbc.Row([
            dbc.Col(_metric_card("💰 Your Cost", "—", "#f44336", "unopt"), md=3),
            dbc.Col(_metric_card("💰 Optimized", "—", "#4caf50", "opt"), md=3),
            dbc.Col(_metric_card("📉 Savings", "—", "#6c63ff", "savings"), md=3),
            dbc.Col(_metric_card("⚡ Peak Cut", "—", "#ffc107", "peak"), md=3),
        ], id="metrics-row", className="mb-3"),

        dbc.Row([
            # LEFT: Settings
            dbc.Col([
                # Rates
                _card([
                    _section_title("⚡ Electricity Rates (₹/kWh)"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Off-Peak", style={"fontSize": "10px", "color": "#4caf50"}),
                            dcc.Input(id="rate-offpeak", type="number", value=DEFAULT_RATES["off_peak"],
                                      min=0, step=0.5, style=_input_style("#4caf50")),
                        ], width=4),
                        dbc.Col([
                            html.Label("Shoulder", style={"fontSize": "10px", "color": "#ffc107"}),
                            dcc.Input(id="rate-shoulder", type="number", value=DEFAULT_RATES["shoulder"],
                                      min=0, step=0.5, style=_input_style("#ffc107")),
                        ], width=4),
                        dbc.Col([
                            html.Label("Peak", style={"fontSize": "10px", "color": "#f44336"}),
                            dcc.Input(id="rate-peak", type="number", value=DEFAULT_RATES["peak"],
                                      min=0, step=0.5, style=_input_style("#f44336")),
                        ], width=4),
                    ]),
                    html.Div([html.Span("Off-peak: 10pm-6am  |  Shoulder: 6-10am, 2-6pm  |  Peak: 10am-2pm, 6-10pm",
                             style={"fontSize": "9px", "color": "#666"})], className="mt-1"),
                ]),

                # Occupancy
                _card([
                    _section_title("👤 Occupancy (people per hour)"),
                    make_occupancy_input(),
                ]),

                # Schedule
                _card([
                    _section_title("📅 Your Appliance Schedule (click to toggle)"),
                    make_schedule_grid(),
                    html.Div([
                        html.Span("■ ", style={"color": "#4caf50"}), "ON  ",
                        html.Span("■ ", style={"color": "rgba(80,80,100,0.5)"}), "OFF",
                    ], style={"fontSize": "10px", "color": "#888", "marginTop": "4px"}),
                ]),

                # Presets + Optimize
                dbc.Row([
                    dbc.Col(dbc.Button("All Peak", id="preset-peak", size="sm",
                                       color="danger", outline=True, className="w-100"), width=3),
                    dbc.Col(dbc.Button("All Off-Peak", id="preset-offpeak", size="sm",
                                       color="success", outline=True, className="w-100"), width=3),
                    dbc.Col(dbc.Button("Spread", id="preset-spread", size="sm",
                                       color="warning", outline=True, className="w-100"), width=3),
                    dbc.Col(dbc.Button("Reset", id="preset-reset", size="sm",
                                       color="secondary", outline=True, className="w-100"), width=3),
                ], className="mb-2"),

                dbc.Button("🚀 Optimize Schedule", id="btn-optimize", color="primary",
                          size="lg", className="w-100",
                          style={"fontWeight": "700", "fontSize": "16px",
                                 "background": "linear-gradient(135deg, #6c63ff, #3f37c9)",
                                 "border": "none", "borderRadius": "10px"}),

                dcc.Loading(html.Div(id="optimize-status", className="text-center mt-2",
                                     style={"color": "#aaa", "fontSize": "12px"}),
                           type="circle", color="#6c63ff"),

                # Hidden store for schedule state
                dcc.Store(id="schedule-store",
                         data={app: DEFAULT_SCHEDULE[app] for app in APPLIANCE_NAMES}),
                dcc.Store(id="results-store", data=None),
            ], md=5, style={"maxHeight": "90vh", "overflowY": "auto"}),

            # RIGHT: Results
            dbc.Col([
                # Time slider
                html.Div([
                    html.Div(id="time-label", style={
                        "textAlign": "center", "color": "#6c63ff",
                        "fontSize": "16px", "fontWeight": "700", "marginBottom": "5px",
                    }),
                    dcc.Slider(id="time-slider", min=0, max=23, value=18, step=1,
                              marks={h: {"label": HOUR_LABELS[h],
                                         "style": {"fontSize": "9px", "color": "#aaa"}}
                                     for h in range(0, 24, 2)}),
                ], style={
                    "background": "rgba(108,99,255,0.08)",
                    "borderRadius": "12px", "padding": "12px 16px",
                    "border": "1px solid rgba(108,99,255,0.2)",
                }),

                # Floorplans
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id="user-house", config={"displayModeBar": False}),
                        html.Div(id="user-stats", className="text-center"),
                    ], md=6),
                    dbc.Col([
                        dcc.Graph(id="opt-house", config={"displayModeBar": False}),
                        html.Div(id="opt-stats", className="text-center"),
                    ], md=6),
                ], className="mt-2"),

                # Heatmap
                dcc.Graph(id="heatmap", config={"displayModeBar": False}),

            ], md=7),
        ]),

    ], fluid=True, style={"backgroundColor": "transparent", "minHeight": "100vh"})

    # ===================================================================
    # Callbacks
    # ===================================================================

    # -- Toggle cell clicks → update store --
    @app.callback(
        Output("schedule-store", "data"),
        [Input({"type": "cell", "app": ALL, "hour": ALL}, "n_clicks"),
         Input("preset-peak", "n_clicks"),
         Input("preset-offpeak", "n_clicks"),
         Input("preset-spread", "n_clicks"),
         Input("preset-reset", "n_clicks")],
        [State("schedule-store", "data")],
        prevent_initial_call=True,
    )
    def update_schedule(cell_clicks, peak_n, offpeak_n, spread_n, reset_n, store):
        triggered = ctx.triggered_id

        if triggered == "preset-peak":
            return {app: [0]*6+[0]*4+[0]*4+[0]*4+[1]*4+[1]*2 for app in APPLIANCE_NAMES}
        elif triggered == "preset-offpeak":
            return {app: [1]*6+[0]*12+[0]*4+[0]*2 for app in APPLIANCE_NAMES}
        elif triggered == "preset-spread":
            return {app: [int(h%3==0) for h in range(24)] for app in APPLIANCE_NAMES}
        elif triggered == "preset-reset":
            return {app: DEFAULT_SCHEDULE[app] for app in APPLIANCE_NAMES}

        # Cell toggle
        if isinstance(triggered, dict) and triggered.get("type") == "cell":
            app_name = triggered["app"]
            hour = triggered["hour"]
            store[app_name][hour] = 1 - store[app_name][hour]

        return store

    # -- Style cells based on store --
    @app.callback(
        [Output({"type": "cell", "app": app, "hour": h}, "children")
         for app in APPLIANCE_NAMES for h in range(24)],
        Input("schedule-store", "data"),
    )
    def style_cells(store):
        outputs = []
        for app in APPLIANCE_NAMES:
            for h in range(24):
                is_on = store[app][h]
                outputs.append(html.Div(style={
                    "width": "18px", "height": "18px", "borderRadius": "4px",
                    "cursor": "pointer",
                    "background": "#4caf50" if is_on else "rgba(80,80,100,0.3)",
                    "border": "1px solid rgba(255,255,255,0.1)",
                    "transition": "all 0.15s",
                }))
        return outputs

    # -- Optimize button --
    @app.callback(
        [Output("results-store", "data"),
         Output("optimize-status", "children"),
         Output("unopt-val", "children"),
         Output("opt-val", "children"),
         Output("savings-val", "children"),
         Output("peak-val", "children")],
        Input("btn-optimize", "n_clicks"),
        [State("schedule-store", "data"),
         State("rate-offpeak", "value"),
         State("rate-shoulder", "value"),
         State("rate-peak", "value")]
        + [State({"type": "occ", "hour": h}, "value") for h in range(24)],
        prevent_initial_call=True,
    )
    def run_optimize(n_clicks, schedule_store, rate_off, rate_sh, rate_pk, *occ_values):
        from src.differential_evolution import quick_optimize

        # Build inputs
        user_schedule = np.array([schedule_store[app] for app in APPLIANCE_NAMES], dtype=float)

        rates = {"off_peak": rate_off or 2.0, "shoulder": rate_sh or 5.0, "peak": rate_pk or 12.0}
        prices = hour_to_price(rates, DEFAULT_PERIODS)

        occupancy = np.array([v or 0 for v in occ_values], dtype=float)

        # Run optimization
        results = quick_optimize(user_schedule, prices, occupancy)

        # Serialize for store
        store_data = {
            "optimized_hourly": results["optimized_hourly"].tolist(),
            "user_hourly": user_schedule.tolist(),
            "prices_hourly": prices.tolist(),
            "unoptimized_cost": results["unoptimized_cost"],
            "optimized_cost": results["optimized_cost"],
            "savings_pct": results["savings_pct"],
            "peak_reduction_pct": results["peak_reduction_pct"],
            "unopt_peak": results["unopt_peak"],
            "opt_peak": results["opt_peak"],
        }

        status = f"✅ Optimized in {len(results['history']['best_fitness'])} generations"

        return (
            store_data, status,
            f"₹{results['unoptimized_cost']:.0f}",
            f"₹{results['optimized_cost']:.0f}",
            f"{results['savings_pct']:.1f}%",
            f"{results['peak_reduction_pct']:.1f}%",
        )

    # -- Update visualizations on time slider or results change --
    @app.callback(
        [Output("time-label", "children"),
         Output("user-house", "figure"),
         Output("opt-house", "figure"),
         Output("user-stats", "children"),
         Output("opt-stats", "children"),
         Output("heatmap", "figure")],
        [Input("time-slider", "value"),
         Input("results-store", "data"),
         Input("schedule-store", "data")],
        [State("rate-offpeak", "value"),
         State("rate-shoulder", "value"),
         State("rate-peak", "value")],
    )
    def update_visuals(hour, results_data, schedule_store, rate_off, rate_sh, rate_pk):
        hour = int(hour) if hour is not None else 18

        if not schedule_store:
            schedule_store = {app: DEFAULT_SCHEDULE[app] for app in APPLIANCE_NAMES}

        rates = {"off_peak": float(rate_off or 2.0),
                 "shoulder": float(rate_sh or 5.0),
                 "peak": float(rate_pk or 12.0)}
        prices = hour_to_price(rates, DEFAULT_PERIODS)
        price = float(prices[hour])

        if price <= rates["off_peak"]:
            period, pcol = "Off-Peak", "#4caf50"
        elif price <= rates["shoulder"]:
            period, pcol = "Shoulder", "#ffc107"
        else:
            period, pcol = "Peak", "#f44336"

        label = html.Span([
            f"🕐 {HOUR_LABELS[hour]} ({hour}:00)  |  ",
            html.Span(f"₹{price:.0f}/kWh ({period})", style={"color": pcol}),
        ])

        # User house
        user_states = {}
        for app in APPLIANCE_NAMES:
            sched = schedule_store.get(app, [0]*24)
            on = sched[hour] if hour < len(sched) else 0
            user_states[app] = RATED_POWERS[app] if on else 0
        user_fig, user_power = draw_house(user_states, "📋 YOUR SCHEDULE")
        user_cost = user_power * price
        user_stat = html.Div([
            html.Span(f"⚡ {user_power:.1f}kW | ", style={"color": "#aaa", "fontSize": "12px"}),
            html.Span(f"₹{user_cost:.1f}/hr", style={"color": "#f44336", "fontSize": "12px", "fontWeight": "700"}),
        ])

        # Optimized house
        if results_data and isinstance(results_data, dict) and "optimized_hourly" in results_data:
            opt_hourly = np.array(results_data["optimized_hourly"])
            opt_states = {}
            for i, app in enumerate(APPLIANCE_NAMES):
                opt_states[app] = RATED_POWERS[app] if opt_hourly[i, hour] > 0.5 else 0
            opt_fig, opt_power = draw_house(opt_states, "✅ OPTIMIZED")
            opt_cost = opt_power * price
            opt_stat = html.Div([
                html.Span(f"⚡ {opt_power:.1f}kW | ", style={"color": "#aaa", "fontSize": "12px"}),
                html.Span(f"₹{opt_cost:.1f}/hr", style={"color": "#4caf50", "fontSize": "12px", "fontWeight": "700"}),
            ])
        else:
            opt_fig, _ = draw_house({app: 0 for app in APPLIANCE_NAMES},
                                    "✅ OPTIMIZED (click Optimize)")
            opt_stat = html.Div("Click 🚀 Optimize to see results",
                               style={"color": "#666", "fontSize": "11px"})

        # Heatmap
        heatmap_fig = build_heatmap(schedule_store, results_data, hour, prices)

        return label, user_fig, opt_fig, user_stat, opt_stat, heatmap_fig

    return app


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def build_heatmap(schedule_store, results_data, current_hour, prices):
    labels_before = [f"{APPLIANCE_LABELS[a]} (Yours)" for a in APPLIANCE_NAMES]

    user_arr = np.array([schedule_store[a] for a in APPLIANCE_NAMES], dtype=float)

    if results_data:
        opt_arr = np.array(results_data["optimized_hourly"])
        labels_after = [f"{APPLIANCE_LABELS[a]} (Optimized)" for a in APPLIANCE_NAMES]
        combined = np.vstack([user_arr, opt_arr])
        labels = labels_before + labels_after
    else:
        combined = user_arr
        labels = labels_before

    # Color by price when ON
    display = np.zeros_like(combined)
    for h in range(24):
        for row in range(combined.shape[0]):
            if combined[row, h] > 0.5:
                display[row, h] = prices[h]

    fig = go.Figure(go.Heatmap(
        z=display, x=list(range(24)), y=labels,
        colorscale=[
            [0, "rgba(40,40,60,0.8)"], [0.01, "rgba(40,40,60,0.8)"],
            [0.15, "#4caf50"], [0.4, "#ffc107"], [1.0, "#f44336"],
        ],
        colorbar=dict(title="₹/kWh", tickfont=dict(color="#aaa"),
                      title_font=dict(color="#aaa")),
        hovertemplate="Hour %{x}:00<br>%{y}<br>₹%{z:.1f}/kWh<extra></extra>",
    ))

    fig.add_vline(x=current_hour, line=dict(color="#6c63ff", width=2, dash="dash"))

    if results_data:
        fig.add_hline(y=len(APPLIANCE_NAMES) - 0.5, line=dict(color="white", width=2))

    fig.update_layout(
        title=dict(text="📊 Schedule Comparison (colored by price when ON)",
                   font=dict(size=13, color="#6c63ff"), x=0.5),
        xaxis=dict(tickvals=list(range(24)), ticktext=HOUR_LABELS,
                   tickfont=dict(color="#aaa", size=9)),
        yaxis=dict(tickfont=dict(color="#aaa", size=9), autorange="reversed"),
        plot_bgcolor="rgba(15,12,41,0.95)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=140, r=20, t=40, b=30), height=320,
    )
    return fig


def _input_style(color):
    return {
        "width": "100%", "background": "rgba(108,99,255,0.1)",
        "border": f"1px solid {color}55", "borderRadius": "6px",
        "color": "#fff", "padding": "4px 8px", "fontSize": "14px",
    }


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=8050)
