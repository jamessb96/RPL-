import base64
import json
import pathlib
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- GLOBAL CSS ----------------
st.markdown(
    """
<style>
/* Container for the Add Graph button */
#add-graph-container {
    width: 100%;
    display: flex;
    justify-content: flex-end;
    margin-top: 15px;
    margin-bottom: 40px;
}

/* Style the Streamlit button inside our container */
#add-graph-container button {
    width: 40px !important;
    height: 40px !important;
    background-color: #3a3a3a !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.35) !important;
    padding: 0 !important;
}

/* Center and size the "+" text */
#add-graph-container button p {
    font-size: 26px !important;
    font-weight: bold !important;
    line-height: 40px !important;
    text-align: center !important;
}

/* Hover state */
#add-graph-container button:hover {
    background-color: #505050 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- PATHS ----------------

BASE_DIR = pathlib.Path(__file__).parent
DATA_FILE = BASE_DIR / "stats_data.csv"
COLOR_FILE = BASE_DIR / "stats_colors.json"
LOGO_FILE = BASE_DIR / "red_panda_logo.png"

DEFAULT_PALETTE = [
    "#FF4B4B",
    "#FF9F1C",
    "#FFEA00",
    "#4ECDC4",
    "#1E90FF",
    "#2ECC71",
    "#9B59B6",
    "#E67E22",
    "#F1C40F",
    "#16A085",
]

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Red Panda Leads – Stats Dashboard",
    layout="wide",
)

# ---------------- LOAD/SAVE UTILS ----------------

def load_data() -> pd.DataFrame:
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame({"Date": pd.to_datetime([], utc=False)})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def save_data(df: pd.DataFrame) -> None:
    df_out = df.copy()
    if "Date" in df_out.columns:
        df_out["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(DATA_FILE, index=False)


def load_colors(columns: List[str]) -> Dict[str, str]:
    if COLOR_FILE.exists():
        try:
            colors = json.loads(COLOR_FILE.read_text())
        except Exception:
            colors = {}
    else:
        colors = {}

    palette_idx = 0
    for col in columns:
        if col == "Date":
            continue
        if col not in colors:
            colors[col] = DEFAULT_PALETTE[palette_idx % len(DEFAULT_PALETTE)]
            palette_idx += 1
    return colors


def save_colors(colors: Dict[str, str]) -> None:
    COLOR_FILE.write_text(json.dumps(colors, indent=2))


# ---------------- DATE FILTERING ----------------

def filter_by_date(df: pd.DataFrame, date_range_label: str, custom_range):
    if df.empty or "Date" not in df:
        return df

    df = df.sort_values("Date")
    min_d, max_d = df["Date"].min(), df["Date"].max()

    if date_range_label == "All time":
        return df

    if date_range_label == "Custom":
        if custom_range:
            start, end = custom_range
            if start:
                df = df[df["Date"] >= pd.to_datetime(start)]
            if end:
                df = df[df["Date"] <= pd.to_datetime(end)]
        return df

    days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
    days = days_map.get(date_range_label)
    if days:
        start_date = max(max_d - timedelta(days=days - 1), min_d)
        return df[df["Date"].between(start_date, max_d)]

    return df


def resample_df(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    if df.empty or "Date" not in df:
        return df
    if granularity == "Daily":
        return df

    df = df.set_index("Date")
    numeric_cols = df.select_dtypes(include="number").columns
    if numeric_cols.empty:
        return df.reset_index()

    weekly = df[numeric_cols].resample("W").sum()
    return weekly.reset_index()


# ---------------- SESSION INIT ----------------

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "colors" not in st.session_state:
    st.session_state.colors = load_colors(st.session_state.df.columns)
if "graphs" not in st.session_state:
    st.session_state.graphs = [{"id": 1, "metrics": [], "overrides": {}}]


# ---------------- LOGO HELPERS ----------------

def get_logo_base64():
    if LOGO_FILE.exists():
        with open(LOGO_FILE, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


LOGO_B64 = get_logo_base64()


def centered_logo_and_title():
    # Logo centered with HTML, sized smaller, with crisp rendering
    if LOGO_B64:
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:5px;">
                <img src="data:image/png;base64,{LOGO_B64}"
                     style="height:90px; image-rendering:-webkit-optimize-contrast; image-rendering:crisp-edges;" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        "<h1 style='text-align:center; margin-top:0px;'>Red Panda Leads – Stats Dashboard</h1>",
        unsafe_allow_html=True,
    )
    # Slightly reduced bottom margin so Graph 1 is higher on the screen
    st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Page", ["Data table", "Graphs"], index=1)

    st.header("Filters")
    granularity = st.radio("Granularity", ["Daily", "Weekly"], index=0)

    date_range_label = st.selectbox(
        "Date range",
        ["All time", "Last 7 days", "Last 30 days", "Last 90 days", "Custom"],
    )

    df_dates = st.session_state.df
    if "Date" in df_dates and not df_dates.empty:
        dmin = df_dates["Date"].min().date()
        dmax = df_dates["Date"].max().date()
    else:
        dmin = dmax = datetime.today().date()

    custom_range = None
    if date_range_label == "Custom":
        custom_range = st.date_input("Custom range", (dmin, dmax))


# ---------------- DATA PAGE ----------------

def page_data_table():
    centered_logo_and_title()
    st.subheader("Data Table")

    df = st.session_state.df

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        uploaded = st.file_uploader("Import CSV", type="csv")
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
                if "Date" in new_df:
                    new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce")
                st.session_state.df = new_df
                st.session_state.colors = load_colors(new_df.columns)
                save_data(new_df)
                save_colors(st.session_state.colors)
                st.success("Imported.")
            except Exception as e:
                st.error(str(e))

    with col2:
        if not df.empty:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "stats_export.csv",
                "text/csv",
            )

    with col3:
        if st.button("Save table"):
            save_data(df)
            st.success("Saved.")

    if "Date" not in df:
        df["Date"] = pd.NaT

    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    if "Date" in edited:
        edited["Date"] = pd.to_datetime(edited["Date"], errors="coerce")

    if st.button("Apply changes", type="primary"):
        st.session_state.df = edited
        st.session_state.colors = load_colors(edited.columns)
        save_data(edited)
        save_colors(st.session_state.colors)
        st.success("Updated.")


# ---------------- GRAPH ADD HELPER ----------------

def add_graph():
    graphs = st.session_state.graphs
    new_id = max(g["id"] for g in graphs) + 1
    graphs.append({"id": new_id, "metrics": [], "overrides": {}})
    st.session_state.graphs = graphs


# ---------------- GRAPHS PAGE ----------------

def page_graphs():
    centered_logo_and_title()

    df = st.session_state.df.copy()
    if df.empty or "Date" not in df:
        st.warning("No data available.")
        return

    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df = filter_by_date(df, date_range_label, custom_range)
    df = resample_df(df, granularity)

    if df.empty:
        st.info("No data in this range.")
        return

    numeric_cols = [
        c for c in df.columns
        if c != "Date" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        st.info("No numeric columns.")
        return

    graphs = st.session_state.graphs

    # Render each graph
    for idx, graph in enumerate(graphs):
        st.markdown("---")
        st.markdown(f"### Graph {idx + 1}")

        selected = st.multiselect(
            "Choose options",
            numeric_cols,
            default=[m for m in graph["metrics"] if m in numeric_cols],
            key=f"metrics_{graph['id']}",
        )
        graph["metrics"] = selected

        if selected:
            fig = go.Figure()
            for metric in selected:
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df[metric],
                        mode="lines",
                        name=metric,
                        line=dict(color=st.session_state.colors.get(metric)),
                    )
                )

            fig.update_layout(
                height=700,  # bigger graph area
                margin=dict(l=40, r=40, t=10, b=40),
                dragmode="pan",
                xaxis=dict(
                    type="date",
                    rangeslider=dict(visible=True),
                ),
                yaxis=dict(title="Value"),
                legend=dict(
                    orientation="v",
                    xanchor="right",
                    x=1.02,
                    y=0.95,
                ),
                modebar=dict(orientation="h"),
            )

            config = {
                "scrollZoom": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "modeBarButtonsToAdd": [
                    "zoom2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "resetScale2d",
                    "zoomX",
                    "zoomY",
                ],
            }

            st.plotly_chart(fig, use_container_width=True, config=config)

    # ---- Single Add Graph FAB at bottom-right of last graph ----
    st.markdown('<div id="add-graph-container">', unsafe_allow_html=True)
    if st.button("+", key="add_graph"):
        add_graph()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- ROUTER ----------------

if page == "Data table":
    page_data_table()
else:
    page_graphs()
