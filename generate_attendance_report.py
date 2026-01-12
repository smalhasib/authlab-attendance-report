#!/usr/bin/env python3
"""
Generate graphical attendance report from authlab_attendences.json

Outputs:
- report/*.png charts
- report/attendance_report.html summary + embedded images

Usage:
  python3 generate_attendance_report.py
"""
from __future__ import annotations
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import json
import webbrowser
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering


def load_attendance(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Expecting keys: total, per_page, ... , data: [ {...}, ... ]
    data = payload.get("data", [])
    df = pd.DataFrame(data)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce types
    df = df.copy()
    df["enter_dt"] = pd.to_datetime(df.get("enter_date_time"), errors="coerce")
    df["exit_dt"] = pd.to_datetime(df.get("exit_date_time"), errors="coerce")
    df["duration_min"] = pd.to_numeric(
        df.get("duration_minutes"), errors="coerce")

    # Where duration is missing but we have both datetimes, compute it
    missing_mask = df["duration_min"].isna(
    ) & df["enter_dt"].notna() & df["exit_dt"].notna()
    df.loc[missing_mask, "duration_min"] = (
        (df.loc[missing_mask, "exit_dt"] -
         df.loc[missing_mask, "enter_dt"]).dt.total_seconds() / 60.0
    )

    df["duration_hours"] = df["duration_min"] / 60.0
    df["date"] = df["enter_dt"].dt.date
    df["weekday"] = df["enter_dt"].dt.day_name()
    df["arrival_hour"] = df["enter_dt"].dt.hour + \
        df["enter_dt"].dt.minute / 60.0
    df["exit_hour"] = df["exit_dt"].dt.hour + df["exit_dt"].dt.minute / 60.0

    return df


def compute_daily(df_completed: pd.DataFrame) -> pd.DataFrame:
    # Sum duration per day in hours
    daily = df_completed.groupby("date")["duration_hours"].sum().reset_index()
    # convert back to datetime for plotting
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["meets_9h"] = daily["duration_hours"] >= 9.0
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def ensure_outdir(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_plot(fig, out_path: Path):
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_daily_hours(daily: pd.DataFrame, df_completed: pd.DataFrame, outdir: Path, year_suffix: str = "") -> Path:
    # Get first arrival and last exit for each day
    daily_times = df_completed.groupby("date").agg(
        first_arrival=("arrival_hour", "min"),
        last_exit=("exit_hour", "max")
    ).reset_index()
    daily_times["date"] = pd.to_datetime(daily_times["date"])

    # Merge with daily data
    daily_with_times = daily.merge(daily_times, on="date", how="left")

    # Function to format decimal hours to 12-hour format
    def format_time(hour_decimal):
        if pd.isna(hour_decimal):
            return "N/A"
        hours = int(hour_decimal)
        minutes = int(round((hour_decimal - hours) * 60))
        period = "AM" if hours < 12 else "PM"
        display_hours = hours % 12
        if display_hours == 0:
            display_hours = 12
        return f"{display_hours}:{minutes:02d} {period}"

    # Add formatted times
    daily_with_times["start_time"] = daily_with_times["first_arrival"].apply(
        format_time)
    daily_with_times["end_time"] = daily_with_times["last_exit"].apply(
        format_time)

    fig = go.Figure()

    # Plot days that met 9h requirement
    daily_met = daily_with_times[daily_with_times["meets_9h"]].copy()
    daily_not_met = daily_with_times[~daily_with_times["meets_9h"]].copy()

    if not daily_met.empty:
        fig.add_trace(go.Scatter(
            x=daily_met["date"],
            y=daily_met["duration_hours"],
            mode='lines+markers',
            name='≥9 hours',
            line=dict(color='#2a9d8f', width=2),
            marker=dict(size=8, symbol='circle'),
            customdata=daily_met[["start_time", "end_time"]],
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Hours:</b> %{y:.2f}h<br><b>Start:</b> %{customdata[0]}<br><b>End:</b> %{customdata[1]}<extra></extra>'
        ))

    if not daily_not_met.empty:
        fig.add_trace(go.Scatter(
            x=daily_not_met["date"],
            y=daily_not_met["duration_hours"],
            mode='markers',
            name='<9 hours',
            marker=dict(size=10, symbol='x',
                        color='#e76f51', line=dict(width=2)),
            customdata=daily_not_met[["start_time", "end_time"]],
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Hours:</b> %{y:.2f}h<br><b>Start:</b> %{customdata[0]}<br><b>End:</b> %{customdata[1]}<br><b>⚠️ Below target</b><extra></extra>'
        ))

    # Add 9-hour target line
    if not daily.empty:
        fig.add_hline(
            y=9,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            annotation_text="9h target",
            annotation_position="right"
        )

    fig.update_layout(
        title='Daily Worked Hours (Hover for details)',
        xaxis_title='Date',
        yaxis_title='Hours',
        hovermode='closest',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=50, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(tickangle=-45)

    out_path = outdir / f"attendance_daily_hours{year_suffix}.html"
    fig.write_html(out_path, include_plotlyjs='cdn')
    return out_path


def plot_cumulative_hours(daily: pd.DataFrame, outdir: Path, year_suffix: str = "") -> Path:
    # Aggregate total hours by month and plot as bar chart
    tmp = daily.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("month")["duration_hours"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=monthly, x="month",
                y="duration_hours", ax=ax, color="#2a9d8f")
    ax.set_title("Hours per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hours")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    # Add value labels on bars (hours)
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt='%.1f', padding=3)
        except Exception:
            pass
    out_path = outdir / f"attendance_monthly_hours{year_suffix}.png"
    save_plot(fig, out_path)
    return out_path


def plot_yearly_comparison(daily: pd.DataFrame, outdir: Path) -> Path:
    yearly = daily.groupby("year").agg(
        total_hours=("duration_hours", "sum"),
        days_worked=("duration_hours", "count"),
        days_met_9h=("meets_9h", "sum"),
    ).reset_index()
    yearly["days_not_met_9h"] = yearly["days_worked"] - yearly["days_met_9h"]
    yearly["avg_daily_hours"] = yearly["total_hours"] / yearly["days_worked"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total hours by year
    sns.barplot(data=yearly, x="year", y="total_hours",
                ax=axes[0], color="#457b9d")
    axes[0].set_title("Total Hours by Year")
    axes[0].set_ylabel("Hours")
    for container in axes[0].containers:
        try:
            axes[0].bar_label(container, fmt='%.1f', padding=3)
        except Exception:
            pass

    # Average daily hours by year
    sns.barplot(data=yearly, x="year", y="avg_daily_hours",
                ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Average Daily Hours by Year")
    axes[1].set_ylabel("Avg Hours")
    axes[1].axhline(y=9, color="red", linestyle="--", alpha=0.5)
    for container in axes[1].containers:
        try:
            axes[1].bar_label(container, fmt='%.2f', padding=3)
        except Exception:
            pass

    # Days not meeting 9h by year
    sns.barplot(data=yearly, x="year", y="days_not_met_9h",
                ax=axes[2], color="#e76f51")
    axes[2].set_title("Days Not Meeting 9h by Year")
    axes[2].set_ylabel("Days")
    for container in axes[2].containers:
        try:
            axes[2].bar_label(container, fmt='%d', padding=3)
        except Exception:
            pass

    plt.tight_layout()
    out_path = outdir / "attendance_yearly_comparison.png"
    save_plot(fig, out_path)
    return out_path


def plot_arrival_hist(df_completed: pd.DataFrame, outdir: Path, year_suffix: str = "") -> Path:
    # Only valid arrival hours
    series = df_completed["arrival_hour"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    if series.empty:
        min_h, max_h = 0, 24
    else:
        min_h = int(math.floor(series.min()))
        max_h = int(math.ceil(series.max()))
        if min_h == max_h:
            min_h = max(0, min_h - 1)
            max_h = min(24, max_h + 1)
    edges = list(range(min_h, max_h + 1))
    sns.histplot(series, bins=edges, ax=ax, color="#264653")
    ax.set_title("Distribution of Arrival Times")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count")
    ax.set_xlim(min_h, max_h)
    ax.set_xticks(range(min_h, max_h + 1))
    # Add value labels on histogram bars (counts)
    for container in ax.containers:
        try:
            heights = [patch.get_height() for patch in container]
            labels = [str(int(h)) if h > 0 else "" for h in heights]
            ax.bar_label(container, labels=labels, padding=2)
        except Exception:
            pass
    ax.grid(True, axis="y", alpha=0.3)
    out_path = outdir / f"attendance_arrival_hist{year_suffix}.png"
    save_plot(fig, out_path)
    return out_path


def plot_exit_hist(df_completed: pd.DataFrame, outdir: Path, year_suffix: str = "") -> Path:
    series = df_completed["exit_hour"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    if series.empty:
        min_h, max_h = 0, 24
    else:
        min_h = int(math.floor(series.min()))
        max_h = int(math.ceil(series.max()))
        if min_h == max_h:
            min_h = max(0, min_h - 1)
            max_h = min(24, max_h + 1)
    edges = list(range(min_h, max_h + 1))
    sns.histplot(series, bins=edges, ax=ax, color="#e76f51")
    ax.set_title("Distribution of Exit Times")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count")
    ax.set_xlim(min_h, max_h)
    ax.set_xticks(range(min_h, max_h + 1))
    # Add value labels on histogram bars (counts)
    for container in ax.containers:
        try:
            heights = [patch.get_height() for patch in container]
            labels = [str(int(h)) if h > 0 else "" for h in heights]
            ax.bar_label(container, labels=labels, padding=2)
        except Exception:
            pass
    ax.grid(True, axis="y", alpha=0.3)
    out_path = outdir / f"attendance_exit_hist{year_suffix}.png"
    save_plot(fig, out_path)
    return out_path


def plot_dow_avg(daily: pd.DataFrame, outdir: Path, year_suffix: str = "") -> Path:
    dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    tmp = daily.copy()
    tmp["weekday"] = tmp["date"].dt.day_name()
    tmp = tmp.groupby("weekday")["duration_hours"].mean().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=tmp.index, y=tmp.values, ax=ax, color="#457b9d")
    ax.set_title("Average Hours by Day of Week")
    ax.set_xlabel("")
    ax.set_ylabel("Avg Hours")
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")
    # Add value labels on bars (avg hours)
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt='%.2f', padding=3)
        except Exception:
            pass
    out_path = outdir / f"attendance_dow_avg{year_suffix}.png"
    save_plot(fig, out_path)
    return out_path


def compute_duration_categories(daily: pd.DataFrame, df_completed: pd.DataFrame) -> Dict[str, Any]:
    """Compute duration categories with day details using vectorized operations."""
    if daily.empty:
        return {"less_than_9h": [], "9h_plus": [], "10h_plus": [], "11h_plus": [], "12h_plus": [], "weekend_work": []}

    # Get first arrival and last exit for each day
    daily_times = df_completed.groupby("date").agg(
        first_arrival=("arrival_hour", "min"),
        last_exit=("exit_hour", "max")
    ).reset_index()

    # Convert date columns to the same type for merging
    daily_copy = daily.copy()
    daily_copy["date_key"] = daily_copy["date"].dt.date
    daily_copy["weekday"] = daily_copy["date"].dt.day_name()
    daily_times["date_key"] = pd.to_datetime(daily_times["date"]).dt.date

    # Merge with daily data
    merged = daily_copy.merge(
        daily_times[["date_key", "first_arrival", "last_exit"]], on="date_key", how="left")

    # Vectorized time formatting
    def format_time_12h_vectorized(hour_series: pd.Series) -> pd.Series:
        hours = hour_series.fillna(-1).astype(int)
        minutes = ((hour_series - hours) * 60).round().fillna(0).astype(int)
        period = hours.apply(lambda h: "AM" if h < 12 else "PM")
        display_hours = hours % 12
        display_hours = display_hours.replace(0, 12)
        result = display_hours.astype(
            str) + ":" + minutes.astype(str).str.zfill(2) + " " + period
        result[hour_series.isna()] = "N/A"
        return result

    # Pre-compute all formatted values
    merged["start_time"] = format_time_12h_vectorized(merged["first_arrival"])
    merged["end_time"] = format_time_12h_vectorized(merged["last_exit"])
    merged["date_str"] = merged["date"].dt.date.astype(str)
    merged["total_hours"] = merged["duration_hours"].round(2)
    merged["year_int"] = merged["year"].astype(int)

    # Convert to records for fast iteration
    records = merged[["date_str", "start_time", "end_time", "total_hours",
                      "year_int", "weekday", "duration_hours"]].to_dict("records")

    categories = {
        "less_than_9h": [],
        "9h_plus": [],
        "10h_plus": [],
        "11h_plus": [],
        "12h_plus": [],
        "weekend_work": []
    }

    for row in records:
        day_info = {
            "date": row["date_str"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "total_hours": row["total_hours"],
            "year": row["year_int"],
            "weekday": row["weekday"]
        }
        hours = row["duration_hours"]

        # Duration categories
        if hours >= 12:
            categories["12h_plus"].append(day_info)
        if hours >= 11:
            categories["11h_plus"].append(day_info)
        if hours >= 10:
            categories["10h_plus"].append(day_info)
        if hours >= 9:
            categories["9h_plus"].append(day_info)
        else:
            categories["less_than_9h"].append(day_info)

        # Weekend work (Saturday and Sunday)
        if row["weekday"] in ("Saturday", "Sunday"):
            categories["weekend_work"].append(day_info)

    return categories


def compute_summary(df_completed: pd.DataFrame, daily: pd.DataFrame) -> Dict[str, Any]:
    if not daily.empty:
        start_date = daily["date"].min().date()
        end_date = daily["date"].max().date()
        total_hours = float(daily["duration_hours"].sum())
        avg_daily_hours = float(daily["duration_hours"].mean())
        median_daily_hours = float(daily["duration_hours"].median())
        days_not_met_9h = int((~daily["meets_9h"]).sum())
        days_met_9h = int(daily["meets_9h"].sum())
    else:
        start_date = end_date = None
        total_hours = avg_daily_hours = median_daily_hours = 0.0
        days_not_met_9h = days_met_9h = 0

    # Compute per-day first arrival and last exit, then their averages
    per_day = (
        df_completed.groupby("date").agg(
            first_arrival=("arrival_hour", "min"),
            last_exit=("exit_hour", "max"),
        )
    )
    avg_arrival = float(per_day["first_arrival"].mean()
                        ) if not per_day.empty else None
    avg_exit = float(per_day["last_exit"].mean()
                     ) if not per_day.empty else None

    # Compute duration categories
    duration_categories = compute_duration_categories(daily, df_completed)

    # Yearly breakdown
    yearly_stats = []
    if not daily.empty:
        for year in sorted(daily["year"].unique()):
            year_data = daily[daily["year"] == year]
            year_completed = df_completed[df_completed["enter_dt"].dt.year == year]
            year_categories = compute_duration_categories(
                year_data, year_completed)
            yearly_stats.append({
                "year": int(year),
                "days_worked": int(year_data.shape[0]),
                "total_hours": float(year_data["duration_hours"].sum()),
                "avg_daily_hours": float(year_data["duration_hours"].mean()),
                "days_met_9h": int(year_data["meets_9h"].sum()),
                "days_not_met_9h": int((~year_data["meets_9h"]).sum()),
                "duration_categories": year_categories,
            })

    return {
        "start_date": start_date,
        "end_date": end_date,
        "days_count": int(daily.shape[0]),
        "total_hours": total_hours,
        "avg_daily_hours": avg_daily_hours,
        "median_daily_hours": median_daily_hours,
        "days_met_9h": days_met_9h,
        "days_not_met_9h": days_not_met_9h,
        "avg_arrival": avg_arrival,
        "avg_exit": avg_exit,
        "yearly_stats": yearly_stats,
        "duration_categories": duration_categories,
    }


def hms_from_hour_decimal(h: float | None) -> str:
    if h is None or pd.isna(h):
        return "N/A"
    hours = int(h)
    minutes = int(round((h - hours) * 60))
    period = "AM" if hours < 12 else "PM"
    display_hours = hours % 12
    if display_hours == 0:
        display_hours = 12
    return f"{display_hours}:{minutes:02d} {period}"


def render_html(summary: Dict[str, Any], year_images: Dict[str, List[str]], out_html: Path):
    def fmt_hours(val: float) -> str:
        return f"{val:.2f} h" if val is not None else "N/A"

    gen_time = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")

    # Get years sorted in descending order (most recent first)
    years = sorted([ys['year']
                   for ys in summary.get('yearly_stats', [])], reverse=True)
    default_year = years[0] if years else None

    # Build year options
    year_options = '<option value="all">All Years</option>'
    for year in years:
        selected = 'selected' if year == default_year else ''
        year_options += f'<option value="{year}" {selected}>{year}</option>'

    # Build summary data as JSON for JavaScript
    # Convert date objects to strings for JSON serialization
    summary_for_json = summary.copy()
    if summary_for_json.get('start_date'):
        summary_for_json['start_date'] = str(summary_for_json['start_date'])
    if summary_for_json.get('end_date'):
        summary_for_json['end_date'] = str(summary_for_json['end_date'])

    import json
    summary_json = json.dumps(summary_for_json)
    year_images_json = json.dumps(year_images)

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Authlab Attendance Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 0; display: inline-block; }}
    .header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }}
    .year-selector {{ display: flex; align-items: center; gap: 8px; }}
    .year-selector label {{ font-weight: 600; color: #374151; }}
    .year-selector select {{ padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; background: white; cursor: pointer; }}
    .muted {{ color: #6b7280; margin-top: 4px; margin-bottom: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }}
    .card h3 {{ margin: 0 0 6px; font-size: 14px; color: #374151; }}
    .card p {{ margin: 0; font-size: 18px; font-weight: 600; color: #111827; }}
    .card.highlight {{ background: #fef3c7; border-color: #fcd34d; }}
    h2 {{ margin-top: 32px; margin-bottom: 12px; color: #111827; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0 28px; background: white; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
    th {{ background: #f9fafb; font-weight: 600; color: #374151; }}
    tr:last-child td {{ border-bottom: none; }}
    figure {{ margin: 0 0 28px; }}
    figcaption {{ color: #6b7280; font-size: 13px; margin-top: 6px; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e5e7eb; }}
    iframe {{ width: 100%; height: 450px; border: 1px solid #e5e7eb; border-radius: 8px; }}
    .year-content {{ display: none; }}
    .year-content.active {{ display: block; }}
    
    /* Collapsible styles */
    .collapsible-section {{ margin: 16px 0; }}
    .collapsible-header {{ background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 16px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; transition: background 0.2s; }}
    .collapsible-header:hover {{ background: #e5e7eb; }}
    .collapsible-header h3 {{ margin: 0; font-size: 15px; color: #374151; }}
    .collapsible-header .count {{ background: #3b82f6; color: white; padding: 2px 10px; border-radius: 12px; font-size: 13px; font-weight: 600; }}
    .collapsible-header .count.warning {{ background: #ef4444; }}
    .collapsible-header .count.success {{ background: #10b981; }}
    .collapsible-header .arrow {{ transition: transform 0.2s; font-size: 12px; color: #6b7280; }}
    .collapsible-header.open .arrow {{ transform: rotate(180deg); }}
    .collapsible-content {{ display: none; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px; max-height: 400px; overflow-y: auto; }}
    .collapsible-content.open {{ display: block; }}
    .collapsible-content table {{ margin: 0; border-radius: 0; }}
    .collapsible-content th {{ position: sticky; top: 0; background: #f9fafb; }}
    .duration-categories {{ display: flex; flex-direction: column; gap: 8px; }}
  </style>
</head>
<body>
  <div class=\"header\">
    <h1>Authlab Attendance Report</h1>
    <div class=\"year-selector\">
      <label for=\"yearSelect\">View Year:</label>
      <select id=\"yearSelect\" onchange=\"switchYear()\">
        {year_options}
      </select>
    </div>
  </div>
  <div class=\"muted\">Generated at {gen_time}</div>

  <div id=\"summaryCards\"></div>
  <div id=\"yearlyTable\"></div>
  <div id=\"visualizations1\"></div>
  <div id=\"durationCategories\"></div>
  <div id=\"weekendWork\"></div>
  <div id=\"visualizations2\"></div>

  <script>
    const summaryData = {summary_json};
    const yearImages = {year_images_json};
    const defaultYear = {default_year if default_year else 'null'};

    function formatHours(val) {{
      return val != null ? val.toFixed(2) + ' h' : 'N/A';
    }}

    function formatTime(hourDecimal) {{
      if (hourDecimal == null || isNaN(hourDecimal)) return 'N/A';
      const hours = Math.floor(hourDecimal);
      const minutes = Math.round((hourDecimal - hours) * 60);
      const period = hours < 12 ? 'AM' : 'PM';
      let displayHours = hours % 12;
      if (displayHours === 0) displayHours = 12;
      return `${{displayHours}}:${{minutes.toString().padStart(2, '0')}} ${{period}}`;
    }}

    function toggleCollapsible(id) {{
      const header = document.querySelector(`[data-collapse="${{id}}"]`);
      const content = document.getElementById(id);
      header.classList.toggle('open');
      content.classList.toggle('open');
    }}

    function renderDurationCategories(categories, containerId) {{
      const container = document.getElementById(containerId);
      
      const categoryConfigs = [
        {{ key: 'less_than_9h', title: 'Less than 9 hours', countClass: 'warning' }},
        {{ key: '9h_plus', title: '9+ hours', countClass: 'success' }},
        {{ key: '10h_plus', title: '10+ hours', countClass: 'success' }},
        {{ key: '11h_plus', title: '11+ hours', countClass: 'success' }},
        {{ key: '12h_plus', title: '12+ hours', countClass: 'success' }}
      ];

      let html = '<h2>Hours by Duration Category</h2><div class="duration-categories">';
      
      categoryConfigs.forEach((config, index) => {{
        const days = categories[config.key] || [];
        const collapseId = `collapse-${{config.key}}-${{Date.now()}}-${{index}}`;
        
        let tableRows = '';
        days.forEach(day => {{
          tableRows += `
            <tr>
              <td>${{day.date}}</td>
              <td>${{day.start_time}}</td>
              <td>${{day.end_time}}</td>
              <td>${{day.total_hours.toFixed(2)}} h</td>
            </tr>
          `;
        }});

        html += `
          <div class="collapsible-section">
            <div class="collapsible-header" data-collapse="${{collapseId}}" onclick="toggleCollapsible('${{collapseId}}')">
              <h3>${{config.title}}</h3>
              <div style="display: flex; align-items: center; gap: 10px;">
                <span class="count ${{config.countClass}}">${{days.length}} days</span>
                <span class="arrow">▼</span>
              </div>
            </div>
            <div class="collapsible-content" id="${{collapseId}}">
              ${{days.length > 0 ? `
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Start Time</th>
                      <th>End Time</th>
                      <th>Total Hours</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${{tableRows}}
                  </tbody>
                </table>
              ` : '<p style="padding: 16px; color: #6b7280; text-align: center;">No days in this category</p>'}}
            </div>
          </div>
        `;
      }});

      html += '</div>';
      container.innerHTML = html;
    }}

    function renderWeekendWork(categories, containerId) {{
      const container = document.getElementById(containerId);
      const weekendDays = categories.weekend_work || [];
      const collapseId = `collapse-weekend-${{Date.now()}}`;
      
      let tableRows = '';
      weekendDays.forEach(day => {{
        tableRows += `
          <tr>
            <td>${{day.date}}</td>
            <td>${{day.weekday}}</td>
            <td>${{day.start_time}}</td>
            <td>${{day.end_time}}</td>
            <td>${{day.total_hours.toFixed(2)}} h</td>
          </tr>
        `;
      }});

      let html = `
        <h2>Weekend Work</h2>
        <div class="duration-categories">
          <div class="collapsible-section">
            <div class="collapsible-header" data-collapse="${{collapseId}}" onclick="toggleCollapsible('${{collapseId}}')">
              <h3>Days Worked on Weekends (Saturday & Sunday)</h3>
              <div style="display: flex; align-items: center; gap: 10px;">
                <span class="count" style="background: #8b5cf6;">${{weekendDays.length}} days</span>
                <span class="arrow">▼</span>
              </div>
            </div>
            <div class="collapsible-content" id="${{collapseId}}">
              ${{weekendDays.length > 0 ? `
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Day</th>
                      <th>Start Time</th>
                      <th>End Time</th>
                      <th>Total Hours</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${{tableRows}}
                  </tbody>
                </table>
              ` : '<p style="padding: 16px; color: #6b7280; text-align: center;">No weekend work recorded</p>'}}
            </div>
          </div>
        </div>
      `;
      container.innerHTML = html;
    }}

    function switchYear() {{
      const selectedYear = document.getElementById('yearSelect').value;
      
      if (selectedYear === 'all') {{
        showAllYears();
      }} else {{
        showYearData(parseInt(selectedYear));
      }}
    }}

    function showAllYears() {{
      const cards = document.getElementById('summaryCards');
      cards.innerHTML = `
        <h2>Overall Summary</h2>
        <div class=\"cards\">
          <div class=\"card\"><h3>Range</h3><p>${{summaryData.start_date || 'N/A'}} → ${{summaryData.end_date || 'N/A'}}</p></div>
          <div class=\"card\"><h3>Days Worked</h3><p>${{summaryData.days_count || 0}}</p></div>
          <div class=\"card\"><h3>Total Hours</h3><p>${{formatHours(summaryData.total_hours)}}</p></div>
          <div class=\"card\"><h3>Avg Daily</h3><p>${{formatHours(summaryData.avg_daily_hours)}}</p></div>
          <div class=\"card\"><h3>Median Daily</h3><p>${{formatHours(summaryData.median_daily_hours)}}</p></div>
          <div class=\"card highlight\"><h3>Days Met 9h</h3><p style=\"color: #059669;\">${{summaryData.days_met_9h || 0}}</p></div>
          <div class=\"card highlight\"><h3>Days Not Met 9h</h3><p style=\"color: #dc2626;\">${{summaryData.days_not_met_9h || 0}}</p></div>
          <div class=\"card\"><h3>Avg Arrival</h3><p>${{formatTime(summaryData.avg_arrival)}}</p></div>
          <div class=\"card\"><h3>Avg Exit</h3><p>${{formatTime(summaryData.avg_exit)}}</p></div>
        </div>
      `;

      const table = document.getElementById('yearlyTable');
      let tableRows = '';
      summaryData.yearly_stats.forEach(ys => {{
        tableRows += `
          <tr>
            <td>${{ys.year}}</td>
            <td>${{ys.days_worked}}</td>
            <td>${{formatHours(ys.total_hours)}}</td>
            <td>${{formatHours(ys.avg_daily_hours)}}</td>
            <td style=\"color: #059669; font-weight: 600;\">${{ys.days_met_9h}}</td>
            <td style=\"color: #dc2626; font-weight: 600;\">${{ys.days_not_met_9h}}</td>
          </tr>
        `;
      }});
      
      table.innerHTML = `
        <h2>Yearly Breakdown</h2>
        <table>
          <thead>
            <tr>
              <th>Year</th>
              <th>Days Worked</th>
              <th>Total Hours</th>
              <th>Avg Daily Hours</th>
              <th>Days Met 9h</th>
              <th>Days Not Met 9h</th>
            </tr>
          </thead>
          <tbody>
            ${{tableRows}}
          </tbody>
        </table>
      `;

      // Split visualizations - first one (daily hours)
      const viz1 = document.getElementById('visualizations1');
      const viz2 = document.getElementById('visualizations2');
      const allImages = yearImages.all || [];
      
      // First visualization (daily hours)
      viz1.innerHTML = '<h2>Daily Performance</h2>';
      if (allImages.length > 0) {{
        const name = allImages[0];
        if (name.endsWith('.html')) {{
          viz1.innerHTML += `<figure><iframe src="${{name}}" frameborder="0"></iframe><figcaption>${{name}}</figcaption></figure>`;
        }} else {{
          viz1.innerHTML += `<figure><img src="${{name}}" alt="chart" /><figcaption>${{name}}</figcaption></figure>`;
        }}
      }}

      // Render duration categories and weekend work (after daily hours)
      renderDurationCategories(summaryData.duration_categories || {{}}, 'durationCategories');
      renderWeekendWork(summaryData.duration_categories || {{}}, 'weekendWork');

      // Remaining visualizations (monthly hours, yearly comparison, histograms, etc.)
      viz2.innerHTML = '<h2>Additional Visualizations</h2>';
      allImages.slice(1).forEach(name => {{
        if (name.endsWith('.html')) {{
          viz2.innerHTML += `<figure><iframe src="${{name}}" frameborder="0"></iframe><figcaption>${{name}}</figcaption></figure>`;
        }} else {{
          viz2.innerHTML += `<figure><img src="${{name}}" alt="chart" /><figcaption>${{name}}</figcaption></figure>`;
        }}
      }});
    }}

    function showYearData(year) {{
      const yearData = summaryData.yearly_stats.find(ys => ys.year === year);
      if (!yearData) return;

      const cards = document.getElementById('summaryCards');
      cards.innerHTML = `
        <h2>${{year}} Summary</h2>
        <div class=\"cards\">
          <div class=\"card\"><h3>Year</h3><p>${{year}}</p></div>
          <div class=\"card\"><h3>Days Worked</h3><p>${{yearData.days_worked}}</p></div>
          <div class=\"card\"><h3>Total Hours</h3><p>${{formatHours(yearData.total_hours)}}</p></div>
          <div class=\"card\"><h3>Avg Daily</h3><p>${{formatHours(yearData.avg_daily_hours)}}</p></div>
          <div class=\"card highlight\"><h3>Days Met 9h</h3><p style=\"color: #059669;\">${{yearData.days_met_9h}}</p></div>
          <div class=\"card highlight\"><h3>Days Not Met 9h</h3><p style=\"color: #dc2626;\">${{yearData.days_not_met_9h}}</p></div>
        </div>
      `;

      document.getElementById('yearlyTable').innerHTML = '';

      // Split visualizations - first one (daily hours)
      const viz1 = document.getElementById('visualizations1');
      const viz2 = document.getElementById('visualizations2');
      const images = yearImages[year.toString()] || [];
      
      // First visualization (daily hours)
      viz1.innerHTML = '<h2>Daily Performance</h2>';
      if (images.length > 0) {{
        const name = images[0];
        if (name.endsWith('.html')) {{
          viz1.innerHTML += `<figure><iframe src="${{name}}" frameborder="0"></iframe><figcaption>${{name}}</figcaption></figure>`;
        }} else {{
          viz1.innerHTML += `<figure><img src="${{name}}" alt="chart" /><figcaption>${{name}}</figcaption></figure>`;
        }}
      }}

      // Render duration categories and weekend work (after daily hours)
      renderDurationCategories(yearData.duration_categories || {{}}, 'durationCategories');
      renderWeekendWork(yearData.duration_categories || {{}}, 'weekendWork');

      // Remaining visualizations (monthly hours, histograms, etc.)
      viz2.innerHTML = '<h2>Additional Visualizations</h2>';
      images.slice(1).forEach(name => {{
        if (name.endsWith('.html')) {{
          viz2.innerHTML += `<figure><iframe src="${{name}}" frameborder="0"></iframe><figcaption>${{name}}</figcaption></figure>`;
        }} else {{
          viz2.innerHTML += `<figure><img src="${{name}}" alt="chart" /><figcaption>${{name}}</figcaption></figure>`;
        }}
      }});
    }}

    // Initialize with default year on load
    window.onload = function() {{
      if (defaultYear) {{
        showYearData(defaultYear);
      }} else {{
        showAllYears();
      }}
    }};
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def build_report(input_path: Path, outdir: Path) -> Path:
    sns.set_theme(style="whitegrid")

    df_raw = load_attendance(input_path)
    if df_raw.empty:
        raise SystemExit("No data found in JSON.")

    df = prepare_dataframe(df_raw)

    # Keep completed sessions with valid durations
    df_completed = df[(df["status"] == "completed") &
                      df["duration_hours"].notna()].copy()

    daily = compute_daily(df_completed)

    # Output directory
    report_dir = ensure_outdir(outdir)

    # Get all unique years
    years = sorted(daily["year"].unique())

    # Pre-filter data for each year to avoid repeated filtering
    year_data = {}
    for year in years:
        year_data[year] = {
            "daily": daily[daily["year"] == year].copy(),
            "completed": df_completed[df_completed["enter_dt"].dt.year == year].copy()
        }

    # Define all plot tasks
    plot_tasks = []

    # All years combined plots
    plot_tasks.append(("all", 0, plot_daily_hours,
                      (daily, df_completed, report_dir, "_all")))
    plot_tasks.append(("all", 1, plot_yearly_comparison, (daily, report_dir)))
    plot_tasks.append(("all", 2, plot_cumulative_hours,
                      (daily, report_dir, "_all")))
    plot_tasks.append(("all", 3, plot_arrival_hist,
                      (df_completed, report_dir, "_all")))
    plot_tasks.append(
        ("all", 4, plot_exit_hist, (df_completed, report_dir, "_all")))
    plot_tasks.append(("all", 5, plot_dow_avg, (daily, report_dir, "_all")))

    # Individual year plots
    for year in years:
        year_suffix = f"_{year}"
        yd = year_data[year]
        plot_tasks.append((str(year), 0, plot_daily_hours,
                          (yd["daily"], yd["completed"], report_dir, year_suffix)))
        plot_tasks.append((str(year), 1, plot_cumulative_hours,
                          (yd["daily"], report_dir, year_suffix)))
        plot_tasks.append((str(year), 2, plot_arrival_hist,
                          (yd["completed"], report_dir, year_suffix)))
        plot_tasks.append((str(year), 3, plot_exit_hist,
                          (yd["completed"], report_dir, year_suffix)))
        plot_tasks.append((str(year), 4, plot_dow_avg,
                          (yd["daily"], report_dir, year_suffix)))

    # Execute plots in parallel using ThreadPoolExecutor
    year_images = {"all": [None] * 6}
    for year in years:
        year_images[str(year)] = [None] * 5

    def execute_plot(task):
        key, idx, func, args = task
        return key, idx, func(*args)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(execute_plot, task)                   : task for task in plot_tasks}
        for future in as_completed(futures):
            key, idx, path = future.result()
            year_images[key][idx] = path.name

    # Summary (computed in parallel with plots conceptually, but depends on data)
    summary = compute_summary(df_completed, daily)

    # HTML
    report_html = report_dir / "attendance_report.html"
    render_html(summary, year_images, report_html)

    return report_html


def main():
    input_path = "authlab_attendences.json"
    outdir = Path("report").expanduser().resolve()

    if not Path(input_path).exists():
        raise SystemExit(f"Input file not found: {input_path}")

    report_html = build_report(input_path, outdir)
    print(f"Report generated: {report_html}")

    # Open in system default browser
    webbrowser.open(f"file://{report_html}")
    print("Opened report in browser.")


if __name__ == "__main__":
    main()
