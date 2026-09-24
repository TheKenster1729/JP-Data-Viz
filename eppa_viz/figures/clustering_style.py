"""Publication-style layout for time-series clustering figures."""

import numpy as np

PAPER_CLUSTER_FONT = "Times New Roman, Times, Liberation Serif, serif"
PAPER_CLUSTER_COLORS = ["#4E79B4", "#C46BA2", "#E8A317", "#00539a", "#0e6027", "#565151"]

PAPER_X_TICKS = [2020, 2040, 2060, 2080, 2100]

PAPER_TITLE_SIZE = 26
PAPER_FONT_SIZE = 20
PAPER_LEGEND_SIZE = 20
PAPER_AXIS_TITLE_SIZE = 24
PAPER_TICK_SIZE = 20
PAPER_AXIS_TITLE_STANDOFF = 12
PAPER_PARCOORDS_LABEL_SIZE = 11
PAPER_PARCOORDS_RANGE_FONT_SIZE = 11
PAPER_PARCOORDS_TICK_FONT_SIZE = 11
PAPER_PARCOORDS_TITLE_SIZE = 19
# Parcoords axes use the left band; title and cluster legend sit in the right margin.
PAPER_PARCOORDS_DOMAIN = dict(x=[0.025, 0.79], y=[0.08, 0.90])
PAPER_PARCOORDS_WIDTH = int(1200 * 1.15 * 1.20)  # wide canvas for axis title clearance on PNG
PAPER_PARCOORDS_HEIGHT = int(720 * 0.8)  # −20% vertical vs baseline 720


def cluster_legend_labels(labels, n_clusters):
    """Legend text for centroid traces, with share of runs per cluster."""
    labels = list(labels)
    n_runs = len(labels)
    if n_runs == 0:
        return ["Cluster {}".format(i + 1) for i in range(n_clusters)]
    out = []
    for k in range(n_clusters):
        pct = 100.0 * sum(1 for lab in labels if lab == k) / n_runs
        out.append("Cluster {} ({:.0f}%)".format(k + 1, pct))
    return out


def paper_cluster_title(output, region, scenario):
    from styling import Options, Readability

    opts = Options()
    read = Readability()
    name = read.display_title_for_output(output)
    if scenario in opts.publication_scenario_display_names:
        scenario_phrase = "Reference" if scenario == "Ref" else opts.publication_scenario_display_names[scenario]
    else:
        scenario_phrase = opts.scenario_display_names.get(scenario, scenario)
    return "Time Series Clustering, {} Under {}".format(name, scenario_phrase)


def default_y_axis_label(output):
    display = __import__("styling", fromlist=["Readability"]).Readability().display_name_for_output(
        output
    )
    if "share" in display.lower():
        return "Fraction of Energy from Renewables"
    return display


def cart_y_axis_label(output):
    display = __import__("styling", fromlist=["Readability"]).Readability().display_name_for_output(
        output
    )
    if "share" in display.lower() or "renewable" in display.lower():
        return "Fraction of Renewables"
    return default_y_axis_label(output)


def paper_cart_title(output, region, scenario, cluster_one_based):
    from styling import Options, Readability

    opts = Options()
    read = Readability()
    name = read.display_title_for_output(output)
    if region == "GLB" and not name.lower().startswith("global"):
        name = "Global " + name
    if scenario in opts.publication_scenario_display_names:
        scenario_phrase = "Reference" if scenario == "Ref" else opts.publication_scenario_display_names[scenario]
    else:
        scenario_phrase = opts.scenario_display_names.get(scenario, scenario)
    return "Most Important Features to Predict Cluster {}, {} Under {}".format(
        cluster_one_based, name, scenario_phrase,
    )


PAPER_CLUSTER_GRAY = "#C5C5C5"


def _axis_style_kwargs():
    axis_title = dict(
        font=dict(size=PAPER_AXIS_TITLE_SIZE, family=PAPER_CLUSTER_FONT),
        standoff=PAPER_AXIS_TITLE_STANDOFF,
    )
    tickfont = dict(size=PAPER_TICK_SIZE, family=PAPER_CLUSTER_FONT)
    return axis_title, tickfont


def apply_publication_cluster_cart_style(fig, title, y_axis_title, x_years):
    axis_title, tickfont = _axis_style_kwargs()
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=PAPER_TITLE_SIZE, family=PAPER_CLUSTER_FONT),
        ),
        font=dict(family=PAPER_CLUSTER_FONT, size=PAPER_FONT_SIZE, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=88, r=40, t=88, b=64),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="Four Most Important Features", **axis_title),
        tickfont=tickfont,
        showgrid=False,
        zeroline=False,
        mirror=True,
        linecolor="black",
        linewidth=1,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title=dict(text="Avg. Feature Importance", **axis_title),
        tickfont=tickfont,
        showgrid=False,
        zeroline=False,
        mirror=True,
        linecolor="black",
        linewidth=1,
        row=1,
        col=1,
    )
    tickvals = [y for y in PAPER_X_TICKS if y in x_years] or x_years
    fig.update_xaxes(
        title=dict(text="Year", **axis_title),
        tickfont=tickfont,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickmode="array",
        tickvals=tickvals,
        mirror=True,
        linecolor="black",
        linewidth=1,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title=dict(text=y_axis_title, **axis_title),
        tickfont=tickfont,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        mirror=True,
        linecolor="black",
        linewidth=1,
        row=1,
        col=2,
    )


def cluster_parcoords_colorbar(n_clusters):
    return dict(
        tickvals=[
            (n_clusters - 1) / (2 * n_clusters) * (2 * i + 1) for i in range(n_clusters)
        ],
        ticktext=["Cluster {}".format(i + 1) for i in range(n_clusters)],
        len=0.22,
        thickness=12,
        x=0.805,
        xref="paper",
        xanchor="left",
        y=0.54,
        yref="paper",
        yanchor="top",
        tickfont=dict(size=PAPER_PARCOORDS_TITLE_SIZE, family=PAPER_CLUSTER_FONT),
    )


def cluster_parcoords_colorscale(n_clusters):
    scale = []
    for i in range(n_clusters):
        color = PAPER_CLUSTER_COLORS[i % len(PAPER_CLUSTER_COLORS)]
        scale.append((i / n_clusters, color))
        scale.append(((i + 1) / n_clusters, color))
    return scale


def paper_cluster_outputs_title(region, scenario, year):
    from styling import Options

    opts = Options()
    if scenario == "Ref":
        scenario_phrase = "Reference"
    elif scenario in opts.publication_scenario_display_names:
        scenario_phrase = opts.publication_scenario_display_names[scenario]
    else:
        scenario_phrase = opts.scenario_display_names.get(scenario, scenario)
    geo = "Global" if region == "GLB" else region
    return "{} {} Outputs with Clusters, {}".format(scenario_phrase, geo, year)


def parcoords_axis_label(name, unit=None):
    """Single-line "Name (unit)"; a second line would push the axes down."""
    if unit:
        return "{} {}".format(name, unit)
    return name


def parcoords_dimension(name, unit, values):
    """
    One parcoords axis spanning exactly the data range, so the min/max text
    plotly draws at the ends sits flush with the axis and the auto ticks in
    between get labeled.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi <= lo:
        # Constant output (e.g. Ref carbon price is 0 everywhere): give the axis a
        # readable span instead of a picoscale one.
        hi = lo + (abs(lo) * 0.01 or 1.0)
    return dict(
        label=parcoords_axis_label(name, unit),
        values=values,
        range=[lo, hi],
    )


def _wrap_title(title, max_chars=26):
    """Break the title onto short lines so it fits the column right of the axes."""
    lines, current = [], ""
    for word in title.split():
        candidate = word if not current else current + " " + word
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "<br>".join(lines)


def cluster_parcoords_title_annotation(title):
    """Title sits in the right margin above the cluster legend (see example figure)."""
    return dict(
        text=_wrap_title(title),
        x=0.80,
        xref="paper",
        xanchor="left",
        y=0.6,
        yref="paper",
        yanchor="bottom",
        showarrow=False,
        align="left",
        font=dict(size=PAPER_PARCOORDS_TITLE_SIZE, family=PAPER_CLUSTER_FONT),
    )


def apply_publication_cluster_parcoords_style(fig, title):
    fig.update_layout(
        title=None,
        annotations=[cluster_parcoords_title_annotation(title)],
        font=dict(family=PAPER_CLUSTER_FONT, size=PAPER_FONT_SIZE, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
    )


def plot_years_from_pivot(pivot_df):
    cols = list(pivot_df.columns)
    if not cols:
        from styling import Options
        return list(Options().years)

    if hasattr(cols[0], "__iter__") and not isinstance(cols[0], (str, bytes)):
        cols = [c[-1] if isinstance(c, tuple) else c for c in cols]
    return [int(c) for c in cols]


def apply_publication_cluster_style(fig, title, y_axis_title, x_years):
    axis_title = dict(
        font=dict(size=PAPER_AXIS_TITLE_SIZE, family=PAPER_CLUSTER_FONT),
        standoff=PAPER_AXIS_TITLE_STANDOFF,
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=PAPER_TITLE_SIZE, family=PAPER_CLUSTER_FONT),
        ),
        font=dict(family=PAPER_CLUSTER_FONT, size=PAPER_FONT_SIZE, color="black"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=88, r=30, t=88, b=64),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=PAPER_LEGEND_SIZE, family=PAPER_CLUSTER_FONT),
        ),
        xaxis=dict(
            title=dict(text="Year", **axis_title),
            tickfont=dict(size=PAPER_TICK_SIZE, family=PAPER_CLUSTER_FONT),
            showgrid=False,
            zeroline=False,
            ticks="outside",
            tickmode="array",
            tickvals=[y for y in PAPER_X_TICKS if y in x_years] or x_years,
            mirror=True,
            linecolor="black",
            linewidth=1,
        ),
        yaxis=dict(
            title=dict(text=y_axis_title, **axis_title),
            tickfont=dict(size=PAPER_TICK_SIZE, family=PAPER_CLUSTER_FONT),
            showgrid=False,
            zeroline=False,
            ticks="outside",
            mirror=True,
            linecolor="black",
            linewidth=1,
        ),
    )
