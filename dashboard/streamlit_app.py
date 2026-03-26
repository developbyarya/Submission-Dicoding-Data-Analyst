import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_FOLDER = "dataset"
RFM_WEIGHTS = {"Monetary": 5, "Frequency": 5, "Recency": 2}
SULTAN_QUANTILE = 0.90  # top 10% by weighted RFM score

MONTH_NAME_ID = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


@st.cache_data(show_spinner=False)
def load_data(dataset_folder: str = DATASET_FOLDER) -> dict:
    orders_df = pd.read_csv(f"{dataset_folder}/orders_dataset.csv")
    order_items_df = pd.read_csv(f"{dataset_folder}/order_items_dataset.csv")
    customers_df = pd.read_csv(f"{dataset_folder}/customers_dataset.csv")
    geo_df = pd.read_csv(f"{dataset_folder}/geolocation_dataset.csv")
    products_df = pd.read_csv(f"{dataset_folder}/products_dataset.csv")
    product_category_name_translation_df = pd.read_csv(
        f"{dataset_folder}/product_category_name_translation.csv"
    )

    return {
        "orders_df": orders_df,
        "order_items_df": order_items_df,
        "customers_df": customers_df,
        "geo_df": geo_df,
        "products_df": products_df,
        "product_category_name_translation_df": product_category_name_translation_df,
    }


@st.cache_data(show_spinner=False)
def build_order_merge_and_prepare(dfs: dict) -> pd.DataFrame:
    orders_df = dfs["orders_df"].copy()
    order_items_df = dfs["order_items_df"].copy()

    # 1) Outlier handling for item price (same rule as notebook)
    q1 = order_items_df["price"].quantile(0.25)
    q3 = order_items_df["price"].quantile(0.75)
    iqr = q3 - q1
    price_upper = 1.5 * iqr
    order_items_df = order_items_df[order_items_df["price"] <= price_upper].copy()

    # 2) Parse datetimes
    orders_df["order_purchase_timestamp"] = pd.to_datetime(
        orders_df["order_purchase_timestamp"], errors="coerce"
    )
    orders_df["order_approved_at"] = pd.to_datetime(
        orders_df["order_approved_at"], errors="coerce"
    )
    orders_df["order_delivered_carrier_date"] = pd.to_datetime(
        orders_df["order_delivered_carrier_date"], errors="coerce"
    )
    orders_df["order_delivered_customer_date"] = pd.to_datetime(
        orders_df["order_delivered_customer_date"], errors="coerce"
    )
    orders_df["order_estimated_delivery_date"] = pd.to_datetime(
        orders_df["order_estimated_delivery_date"], errors="coerce"
    )

    # 3) Keep orders with valid approval timestamp (needed for time series + RFM)
    orders_df = orders_df.dropna(subset=["order_approved_at"]).copy()

    # 4) Build merged fact table
    order_merge = order_items_df.merge(orders_df, on="order_id", how="inner")
    order_merge = order_merge.dropna(subset=["order_approved_at", "price", "customer_id"]).copy()

    # 5) Convenience time fields
    order_merge["year"] = order_merge["order_approved_at"].dt.year
    order_merge["month"] = order_merge["order_approved_at"].dt.month
    order_merge["year_month"] = order_merge["order_approved_at"].dt.to_period("M").astype(str)

    return order_merge


@st.cache_data(show_spinner=False)
def build_order_merge_enriched(dfs: dict, order_merge: pd.DataFrame) -> pd.DataFrame:
    """Enrich `order_merge` with `product_category_name_english` for drilldowns."""
    products_enriched = dfs["products_df"].merge(
        dfs["product_category_name_translation_df"],
        on="product_category_name",
        how="left",
    )

    products_enriched["product_category_name_english"] = products_enriched[
        "product_category_name_english"
    ].fillna(products_enriched["product_category_name"])

    return order_merge.merge(
        products_enriched[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )


@st.cache_data(show_spinner=False)
def compute_q1_q2_q3(order_merge_enriched: pd.DataFrame) -> dict:
    # Q1: Yearly revenue + YoY
    yearly_revenue = (
        order_merge_enriched.groupby("year", as_index=False)
        .agg(revenue=("price", "sum"), orders=("order_id", "nunique"))
        .sort_values("year")
    )
    yearly_revenue["yoy_growth_%"] = yearly_revenue["revenue"].pct_change() * 100
    best_yoy_row = yearly_revenue.loc[yearly_revenue["yoy_growth_%"].idxmax()]

    # Q2: Month increasing trend based on MoM growth
    monthly_revenue = (
        order_merge_enriched.groupby(
            ["year_month", "year", "month"], as_index=False
        )
        .agg(revenue=("price", "sum"))
        .sort_values("year_month")
    )
    monthly_revenue["mom_growth_%"] = monthly_revenue["revenue"].pct_change() * 100

    month_trend = (
        monthly_revenue.dropna(subset=["mom_growth_%"])
        .groupby("month", as_index=False)
        .agg(
            avg_mom_growth_pct=("mom_growth_%", "mean"),
            positive_mom_rate=("mom_growth_%", lambda s: (s > 0).mean()),
            n_months=("mom_growth_%", "size"),
        )
    )

    month_trend = month_trend.sort_values(
        ["positive_mom_rate", "avg_mom_growth_pct"], ascending=False
    )
    best_month_row = month_trend.iloc[0]
    best_month_num = int(best_month_row["month"])

    # Q3: Top 3 product categories per calendar month by revenue (rank by month only)
    cat_month_revenue = (
        order_merge_enriched.groupby(
            ["month", "product_category_name_english"], as_index=False
        )
        .agg(revenue=("price", "sum"))
    )
    cat_month_revenue["rank"] = (
        cat_month_revenue.groupby("month")["revenue"]
        .rank(ascending=False, method="first")
    )

    top3_rows = cat_month_revenue[cat_month_revenue["rank"] <= 3].copy()
    top3_plot = top3_rows.copy()
    top3_plot["rank"] = top3_plot["rank"].astype(int)
    top3_plot["month_name"] = top3_plot["month"].map(MONTH_NAME_ID)
    top3_plot = top3_plot.sort_values(["month", "rank"])

    winners = top3_plot[top3_plot["rank"] == 1]
    winner_counts = winners["product_category_name_english"].value_counts().head(10)

    return {
        "yearly_revenue": yearly_revenue,
        "best_yoy_row": best_yoy_row,
        "monthly_revenue": monthly_revenue,
        "month_trend": month_trend,
        "best_month_num": best_month_num,
        "best_month_name": MONTH_NAME_ID[best_month_num],
        "top3_plot": top3_plot,
        "winner_counts": winner_counts,
        "month_name_id": MONTH_NAME_ID,
    }


@st.cache_data(show_spinner=False)
def get_revenue_by_month_for_year(monthly_revenue: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return revenue aggregated by calendar month for the selected year."""
    df = monthly_revenue.loc[monthly_revenue["year"] == year, ["month", "revenue"]].copy()
    df = (
        df.groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"))
        .sort_values("month")
    )
    df = (
        df.set_index("month")
        .reindex(range(1, 13), fill_value=0)
        .reset_index()
        .rename(columns={"index": "month"})
    )
    df["month_name"] = df["month"].map(MONTH_NAME_ID)
    return df


@st.cache_data(show_spinner=False)
def compute_mom_month_trend(
    monthly_revenue: pd.DataFrame, year_start: int, year_end: int
) -> dict:
    """Compute MoM% series and the 'most increasing month calendar' summary for a year range."""
    df = monthly_revenue.loc[
        (monthly_revenue["year"] >= year_start) & (monthly_revenue["year"] <= year_end)
    ].copy()
    df = df.sort_values("year_month")
    df["mom_growth_%"] = df["revenue"].pct_change() * 100

    month_trend = (
        df.dropna(subset=["mom_growth_%"])
        .groupby("month", as_index=False)
        .agg(
            avg_mom_growth_pct=("mom_growth_%", "mean"),
            positive_mom_rate=("mom_growth_%", lambda s: (s > 0).mean()),
            n_months=("mom_growth_%", "size"),
        )
    )

    month_trend = month_trend.sort_values(
        ["positive_mom_rate", "avg_mom_growth_pct"], ascending=False
    )
    if month_trend.empty:
        best_month_num = None
        best_month_name = "N/A"
    else:
        best_month_num = int(month_trend.iloc[0]["month"])
        best_month_name = MONTH_NAME_ID[best_month_num]

    return {
        "monthly_revenue_range": df.dropna(subset=["revenue"]).copy(),
        "month_trend": month_trend,
        "best_month_num": best_month_num,
        "best_month_name": best_month_name,
    }


@st.cache_data(show_spinner=False)
def get_top_contributors(
    order_merge_enriched: pd.DataFrame,
    year_month: str,
    top_n: int,
    level: str,
) -> dict:
    """Rank top revenue contributors for a selected month and aggregation level."""
    if level not in {"category", "product_id"}:
        raise ValueError("level must be one of: {'category', 'product_id'}")

    base_col = (
        "product_category_name_english" if level == "category" else "product_id"
    )
    out_col = "category" if level == "category" else "product_id"

    df = order_merge_enriched.loc[
        order_merge_enriched["year_month"] == year_month, ["price", base_col]
    ].copy()
    df = (
        df.groupby(base_col, as_index=False)
        .agg(revenue=("price", "sum"))
        .sort_values("revenue", ascending=False)
    )
    df = df.head(top_n).copy()
    df["rank"] = np.arange(1, len(df) + 1)
    df = df.rename(columns={base_col: out_col})

    total_revenue = (
        order_merge_enriched.loc[order_merge_enriched["year_month"] == year_month, "price"].sum()
    )

    return {"contributors": df, "total_revenue": float(total_revenue)}


@st.cache_data(show_spinner=False)
def get_top_categories_for_calendar_month(
    order_merge_enriched: pd.DataFrame,
    year_start: int,
    year_end: int,
    calendar_month: int,
    top_n: int,
) -> dict:
    """Top categories by revenue for a selected calendar month and year range."""
    if calendar_month < 1 or calendar_month > 12:
        raise ValueError("calendar_month must be between 1 and 12")

    df = order_merge_enriched.loc[
        (order_merge_enriched["year"] >= year_start)
        & (order_merge_enriched["year"] <= year_end)
        & (order_merge_enriched["month"] == calendar_month),
        ["product_category_name_english", "price"],
    ].copy()

    total_revenue = float(df["price"].sum())

    df = (
        df.groupby("product_category_name_english", as_index=False)
        .agg(revenue=("price", "sum"))
        .sort_values("revenue", ascending=False)
        .head(top_n)
        .copy()
    )
    df["rank"] = np.arange(1, len(df) + 1)
    df = df.rename(columns={"product_category_name_english": "category"})
    return {
        "contributors": df,
        "total_revenue": total_revenue,
        "month_name": MONTH_NAME_ID[calendar_month],
    }


def compute_rfm(
    order_merge_cust: pd.DataFrame,
    monetary_weight: float,
    frequency_weight: float,
    recency_weight: float,
    sultan_quantile: float,
    medium_quantile: float,
) -> dict:
    """Compute per-customer RFM scores + Sultan flag + RFM segments."""
    customer_last_order = order_merge_cust.groupby("customer_unique_id")["order_approved_at"].max()
    customer_recency_days = (
        order_merge_cust["order_approved_at"].max() - customer_last_order
    ).dt.days

    customer_frequency = order_merge_cust.groupby("customer_unique_id")["order_id"].nunique()
    customer_monetary = order_merge_cust.groupby("customer_unique_id")["price"].sum()

    rfm_df = pd.DataFrame(
        {
            "recency_days": customer_recency_days,
            "frequency": customer_frequency,
            "monetary": customer_monetary,
        }
    ).reset_index()

    # Robust scoring 1..5 (avoid qcut errors when bins collapse)
    recency_pct_rank = rfm_df["recency_days"].rank(pct=True, method="first")
    rfm_df["Recency_score"] = (6 - np.ceil(recency_pct_rank * 5).astype(int)).clip(1, 5)

    frequency_pct_rank = rfm_df["frequency"].rank(pct=True, method="first")
    rfm_df["Frequency_score"] = np.ceil(frequency_pct_rank * 5).astype(int).clip(1, 5)

    monetary_pct_rank = rfm_df["monetary"].rank(pct=True, method="first")
    rfm_df["Monetary_score"] = np.ceil(monetary_pct_rank * 5).astype(int).clip(1, 5)

    rfm_df["weighted_score"] = (
        monetary_weight * rfm_df["Monetary_score"]
        + frequency_weight * rfm_df["Frequency_score"]
        + recency_weight * rfm_df["Recency_score"]
    )

    sultan_threshold = rfm_df["weighted_score"].quantile(sultan_quantile)
    rfm_df["sultan_flag"] = (rfm_df["weighted_score"] >= sultan_threshold).astype(int)

    sultan_count = int(rfm_df["sultan_flag"].sum())
    sultan_percent = sultan_count / len(rfm_df) * 100

    medium_threshold = rfm_df["weighted_score"].quantile(medium_quantile)
    rfm_df["rfm_segment"] = np.where(
        rfm_df["sultan_flag"] == 1,
        "High (Sultan)",
        np.where(rfm_df["weighted_score"] >= medium_threshold, "Medium", "Low"),
    )

    segment_order = ["Low", "Medium", "High (Sultan)"]
    segment_df = (
        rfm_df["rfm_segment"]
        .value_counts()
        .reindex(segment_order)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    segment_df.columns = ["rfm_segment", "n_customers"]
    segment_df["pct_customers"] = segment_df["n_customers"] / len(rfm_df) * 100

    top_sultan_customers = (
        rfm_df.loc[rfm_df["sultan_flag"] == 1]
        .sort_values("weighted_score", ascending=False)
        .head(20)
        .copy()
    )

    return {
        "rfm_df": rfm_df,
        "sultan_count": sultan_count,
        "sultan_percent": sultan_percent,
        "segment_df": segment_df,
        "top_sultan_customers": top_sultan_customers,
    }


@st.cache_data(show_spinner=False)
def compute_q4_q5(
    dfs: dict,
    order_merge: pd.DataFrame,
    monetary_weight: float,
    frequency_weight: float,
    recency_weight: float,
    sultan_quantile: float,
    medium_quantile: float,
) -> dict:
    customers_df = dfs["customers_df"].copy()
    geo_df = dfs["geo_df"].copy()

    # Q4: RFM + Sultan
    order_merge_cust = order_merge.merge(
        customers_df[
            ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]
        ],
        on="customer_id",
        how="left",
    )
    order_merge_cust = order_merge_cust.dropna(subset=["customer_unique_id"]).copy()

    rfm_results = compute_rfm(
        order_merge_cust=order_merge_cust,
        monetary_weight=monetary_weight,
        frequency_weight=frequency_weight,
        recency_weight=recency_weight,
        sultan_quantile=sultan_quantile,
        medium_quantile=medium_quantile,
    )
    rfm_df = rfm_results["rfm_df"]
    sultan_count = rfm_results["sultan_count"]
    sultan_percent = rfm_results["sultan_percent"]
    segment_df = rfm_results["segment_df"]
    top_sultan_customers = rfm_results["top_sultan_customers"]

    # Q5: Geospatial + State potential (All customers RFM) — mirror notebook
    state_profile = rfm_df.merge(
        customers_df[["customer_unique_id", "customer_zip_code_prefix", "customer_state"]],
        on="customer_unique_id",
        how="left",
    )

    geo_enriched = state_profile.merge(
        geo_df[
            ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
        ].drop_duplicates(subset=["geolocation_zip_code_prefix"]),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    geo_enriched = geo_enriched.dropna(
        subset=["geolocation_lat", "geolocation_lng", "customer_state"]
    ).copy()

    if "weighted_score" in geo_enriched.columns:
        geo_enriched["rfm_value"] = geo_enriched["weighted_score"]
    elif "RFM_Score" in geo_enriched.columns:
        geo_enriched["rfm_value"] = geo_enriched["RFM_Score"]
    elif "rfm_score" in geo_enriched.columns:
        geo_enriched["rfm_value"] = geo_enriched["rfm_score"]
    else:
        component_cols = [
            c
            for c in [
                "Recency_score",
                "Frequency_score",
                "Monetary_score",
                "R_score",
                "F_score",
                "M_score",
                "R",
                "F",
                "M",
            ]
            if c in geo_enriched.columns
        ]
        if len(component_cols) >= 3:
            geo_enriched["rfm_value"] = geo_enriched[component_cols[:3]].mean(axis=1)
        else:
            raise ValueError(
                "Kolom RFM tidak ditemukan. Pastikan rfm_df memiliki weighted_score "
                "atau skor komponen R/F/M sebelum menjalankan Q5."
            )

    state_stats = (
        geo_enriched.groupby("customer_state", as_index=False)
        .agg(
            total_customers_state=("customer_unique_id", "nunique"),
            avg_rfm_state=("rfm_value", "mean"),
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
        )
        .sort_values("avg_rfm_state", ascending=False)
    )

    top_states = state_stats.head(10).copy()
    best_state = str(top_states.iloc[0]["customer_state"])
    best_rfm = float(top_states.iloc[0]["avg_rfm_state"])
    best_count = int(top_states.iloc[0]["total_customers_state"])

    q5_fig = px.scatter_geo(
        state_stats,
        lat="lat",
        lon="lng",
        size="total_customers_state",
        color="avg_rfm_state",
        hover_name="customer_state",
        hover_data={"total_customers_state": True, "avg_rfm_state": ":.4f"},
        projection="natural earth",
        scope="south america",
        title="Q5 - State potential based on Average RFM (All Customers)",
        color_continuous_scale="YlOrRd",
    )

    return {
        "rfm_df": rfm_df,
        "sultan_count": sultan_count,
        "sultan_percent": sultan_percent,
        "segment_counts": segment_df,
        "top_sultan_customers": top_sultan_customers,
        "top_states": top_states,
        "best_state": best_state,
        "best_rfm": best_rfm,
        "best_count": best_count,
        "q5_fig": q5_fig,
    }


@st.cache_data(show_spinner=False)
def compute_state_potential(
    rfm_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    segment_choice: str,
    top_states_n: int,
) -> dict:
    """Compute state-level potential map based on current RFM table."""
    if top_states_n < 1:
        raise ValueError("top_states_n must be >= 1")

    if segment_choice == "Sultan only":
        rfm_filtered = rfm_df.loc[rfm_df["sultan_flag"] == 1].copy()
    else:
        rfm_filtered = rfm_df.copy()

    state_profile = rfm_filtered.merge(
        customers_df[
            ["customer_unique_id", "customer_zip_code_prefix", "customer_state"]
        ],
        on="customer_unique_id",
        how="left",
    )

    geo_enriched = state_profile.merge(
        geo_df[["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]]
        .drop_duplicates(subset=["geolocation_zip_code_prefix"]),
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    geo_enriched = geo_enriched.dropna(
        subset=["geolocation_lat", "geolocation_lng", "customer_state"]
    ).copy()

    geo_enriched["rfm_value"] = geo_enriched["weighted_score"]

    state_stats = (
        geo_enriched.groupby("customer_state", as_index=False)
        .agg(
            total_customers_state=("customer_unique_id", "nunique"),
            avg_rfm_state=("rfm_value", "mean"),
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
        )
        .sort_values("avg_rfm_state", ascending=False)
    )

    top_states = state_stats.head(top_states_n).copy()
    best_state = str(top_states.iloc[0]["customer_state"]) if len(top_states) else "N/A"
    best_rfm = float(top_states.iloc[0]["avg_rfm_state"]) if len(top_states) else float("nan")
    best_count = int(top_states.iloc[0]["total_customers_state"]) if len(top_states) else 0

    q5_fig = px.scatter_geo(
        state_stats,
        lat="lat",
        lon="lng",
        size="total_customers_state",
        color="avg_rfm_state",
        hover_name="customer_state",
        hover_data={"total_customers_state": True, "avg_rfm_state": ":.4f"},
        projection="natural earth",
        scope="south america",
        title=f"Q5 - State potential based on Average RFM ({segment_choice})",
        color_continuous_scale="YlOrRd",
    )

    return {
        "state_stats": state_stats,
        "top_states": top_states,
        "best_state": best_state,
        "best_rfm": best_rfm,
        "best_count": best_count,
        "q5_fig": q5_fig,
    }


def main():
    st.set_page_config(page_title="E-Commerce RFM & Revenue Insights", layout="wide")
    st.title("E-Commerce Insights: Revenue, Top Products, RFM Sultan, dan Geospatial City Potential")

    with st.spinner("Computing metrics..."):
        dfs = load_data(DATASET_FOLDER)
        order_merge = build_order_merge_and_prepare(dfs)
        order_merge_enriched = build_order_merge_enriched(dfs, order_merge)
        q1_q2_q3 = compute_q1_q2_q3(order_merge_enriched)

    q4_q5 = None
    tab_q1, tab_q2, tab_q3, tab_q4, tab_q5 = st.tabs(
        ["Q1 Revenue", "Q2 MoM Trend", "Q3 Top Categories", "Q4 RFM Sultan", "Q5 Geospatial"]
    )

    with tab_q1:
        # Q1
        st.subheader("Q1. Kenaikan Revenue dari tahun ke tahun")
        yearly_revenue = q1_q2_q3["yearly_revenue"].copy()

        # Interactive: pilih tahun, lihat total revenue dan breakdown per bulan.
        year_options = sorted(yearly_revenue["year"].unique().tolist())
        selected_year = st.selectbox(
            "Pilih tahun untuk breakdown revenue per bulan",
            year_options,
            index=len(year_options) - 1,
            key="q1_selected_year",
        )
        yearly_row = yearly_revenue.loc[yearly_revenue["year"] == selected_year].iloc[0]

        st.metric(
            label=f"Total Revenue {selected_year}", value=f"{yearly_row['revenue']:,.2f}"
        )
        st.metric(
            label=f"Total Orders {selected_year}", value=f"{int(yearly_row['orders'])}"
        )

        monthly_for_year = get_revenue_by_month_for_year(
            q1_q2_q3["monthly_revenue"], selected_year
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=monthly_for_year, x="month_name", y="revenue", ax=ax)
        ax.set_title(f"Revenue per Bulan (Selected Year: {selected_year})")
        ax.set_xlabel("Bulan Kalender")
        ax.set_ylabel("Revenue (sum price)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Context: garis revenue per tahun (original notebook view)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.lineplot(data=yearly_revenue, x="year", y="revenue", marker="o", ax=ax)
        ax.set_title("Revenue per Tahun (berdasarkan sum(price))")
        ax.set_xticks(yearly_revenue["year"].unique().tolist())
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab_q2:
        # Q2
        st.subheader("Q2. Dibulan apa penjualan cenderung meningkat")
        monthly_revenue_all = q1_q2_q3["monthly_revenue"].copy()
        year_min = int(q1_q2_q3["yearly_revenue"]["year"].min())
        year_max = int(q1_q2_q3["yearly_revenue"]["year"].max())

        year_range = st.slider(
            "Pilih rentang tahun untuk menghitung MoM Growth (%)",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1,
            key="q2_year_range",
        )
        year_start, year_end = year_range
        mom_results = compute_mom_month_trend(
            monthly_revenue_all, year_start=year_start, year_end=year_end
        )

        monthly_revenue_range = mom_results["monthly_revenue_range"].copy()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(
            data=monthly_revenue_range,
            x="year_month",
            y="revenue",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"Revenue per Bulan dalam Rentang {year_start}-{year_end}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        month_trend = mom_results["month_trend"].copy()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=month_trend.sort_values("month"),
            x="month",
            y="avg_mom_growth_pct",
            ax=ax,
        )
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([MONTH_NAME_ID[i] for i in range(1, 13)], rotation=45)
        ax.set_title("Rata-rata MoM Growth (%) per Bulan Kalender (Skala Log)")
        ax.set_ylabel("Avg MoM Growth (%) - symlog")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.caption(
            f"Bulan kalender paling meningkat pada rentang {year_start}-{year_end}: "
            f"{mom_results['best_month_name']}"
            + (
                f" (bulan {mom_results['best_month_num']})"
                if mom_results["best_month_num"] is not None
                else ""
            )
        )

        # Drilldown: pilih year_month dalam rentang, lihat top kontributor.
        available_year_months = (
            monthly_revenue_range["year_month"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not available_year_months:
            st.warning("Tidak ada data year_month untuk rentang tahun yang dipilih.")
            st.stop()

        selected_year_month = st.selectbox(
            "Pilih bulan untuk drilldown kontribusi sales",
            available_year_months,
            key="q2_selected_year_month",
        )
        level = st.radio(
            "Kontribusi sales berdasarkan apa?",
            options=["category", "product_id"],
            format_func=lambda x: "Kategori" if x == "category" else "Product ID",
            horizontal=True,
            key="q2_contrib_level",
        )
        top_n = st.slider(
            "Top N kontributor",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            key="q2_top_n",
        )

        contrib = get_top_contributors(
            order_merge_enriched=order_merge_enriched,
            year_month=selected_year_month,
            top_n=top_n,
            level=level,
        )
        contributors = contrib["contributors"].copy()

        st.metric(
            label=f"Total Revenue {selected_year_month}",
            value=f"{contrib['total_revenue']:,.2f}",
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(
            data=contributors,
            x=contributors.columns[0],
            y="revenue",
            ax=ax,
        )
        ax.set_title(
            f"Top {top_n} Kontributor Sales - {selected_year_month} ({'Kategori' if level == 'category' else 'Product ID'})"
        )
        ax.set_xlabel("Kontributor")
        ax.set_ylabel("Revenue (sum price)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.dataframe(contributors, use_container_width=True)

    with tab_q3:
        # Q3
        st.subheader("Q3. Top kategori produk terlaris per bulan kalender")
        year_min = int(q1_q2_q3["yearly_revenue"]["year"].min())
        year_max = int(q1_q2_q3["yearly_revenue"]["year"].max())

        q3_year_range = st.slider(
            "Pilih rentang tahun untuk ranking Q3 (opsional)",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
            step=1,
            key="q3_year_range",
        )
        q3_year_start, q3_year_end = q3_year_range

        calendar_month = st.selectbox(
            "Pilih bulan kalender (1-12)",
            options=list(range(1, 13)),
            format_func=lambda m: MONTH_NAME_ID[m],
            key="q3_calendar_month",
        )
        top_n = st.slider(
            "Jumlah top kategori",
            min_value=3,
            max_value=20,
            value=10,
            step=1,
            key="q3_top_n",
        )

        result = get_top_categories_for_calendar_month(
            order_merge_enriched=order_merge_enriched,
            year_start=q3_year_start,
            year_end=q3_year_end,
            calendar_month=calendar_month,
            top_n=top_n,
        )

        st.metric(
            label=f"Total Revenue {result['month_name']} ({q3_year_start}-{q3_year_end})",
            value=f"{result['total_revenue']:,.2f}",
        )

        contributors = result["contributors"].copy()
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.barplot(data=contributors, x="category", y="revenue", ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(
            f"Top {top_n} Kategori Produk Terlaris - {result['month_name']} ({q3_year_start}-{q3_year_end})"
        )
        ax.set_xlabel("Kategori")
        ax.set_ylabel("Revenue (sum price)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.dataframe(contributors, use_container_width=True)

    with tab_q4:
        # Q4
        st.subheader("Q4. Persentase customer 'sultan' yang menggunakan e-commerce")

        monetary_weight = st.slider(
            "Weight Monetary",
            min_value=0,
            max_value=10,
            value=int(RFM_WEIGHTS["Monetary"]),
            step=1,
            key="rfm_monetary_weight",
        )
        frequency_weight = st.slider(
            "Weight Frequency",
            min_value=0,
            max_value=10,
            value=int(RFM_WEIGHTS["Frequency"]),
            step=1,
            key="rfm_frequency_weight",
        )
        recency_weight = st.slider(
            "Weight Recency",
            min_value=0,
            max_value=10,
            value=int(RFM_WEIGHTS["Recency"]),
            step=1,
            key="rfm_recency_weight",
        )

        sultan_quantile = st.slider(
            "Sultan quantile (top % by weighted RFM)",
            min_value=0.50,
            max_value=0.99,
            value=float(SULTAN_QUANTILE),
            step=0.01,
            key="sultan_quantile",
        )
        medium_quantile = st.slider(
            "Medium quantile (split Medium vs Low)",
            min_value=0.40,
            max_value=0.80,
            value=0.60,
            step=0.01,
            key="medium_quantile",
        )

        with st.spinner("Menghitung RFM & Sultan berdasarkan parameter..."):
            q4_q5 = compute_q4_q5(
                dfs=dfs,
                order_merge=order_merge,
                monetary_weight=monetary_weight,
                frequency_weight=frequency_weight,
                recency_weight=recency_weight,
                sultan_quantile=sultan_quantile,
                medium_quantile=medium_quantile,
            )

        st.metric(label="% Sultan Customers", value=f"{q4_q5['sultan_percent']:.2f}%")
        st.caption(
            f"Sultan definition: top {int(sultan_quantile*100)}% weighted RFM score "
            f"(weighted = {monetary_weight}*Monetary + {frequency_weight}*Frequency + {recency_weight}*Recency)."
        )

        st.subheader("Visualisasi Segmentasi Customer (RFM)")
        segment_counts = q4_q5["segment_counts"].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=segment_counts,
            x="rfm_segment",
            y="n_customers",
            order=["Low", "Medium", "High (Sultan)"],
            ax=ax,
        )
        ax.set_title("Segmentasi Customer Berdasarkan Weighted RFM")
        ax.set_xlabel("RFM Segment")
        ax.set_ylabel("Jumlah Customer")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.subheader("Top Sultan Customers (Top 20 by weighted score)")
        top_sultan_customers = q4_q5["top_sultan_customers"].copy()
        cols = [
            "customer_unique_id",
            "weighted_score",
            "Recency_score",
            "Frequency_score",
            "Monetary_score",
            "recency_days",
            "frequency",
            "monetary",
        ]
        st.dataframe(top_sultan_customers[cols], use_container_width=True)

    with tab_q5:
        # Q5
        st.subheader("Q5. Kota dengan potensi customer loyal terbesar (RFM + geospatial)")

        if q4_q5 is None:
            q4_q5 = compute_q4_q5(
                dfs=dfs,
                order_merge=order_merge,
                monetary_weight=RFM_WEIGHTS["Monetary"],
                frequency_weight=RFM_WEIGHTS["Frequency"],
                recency_weight=RFM_WEIGHTS["Recency"],
                sultan_quantile=SULTAN_QUANTILE,
                medium_quantile=0.60,
            )

        segment_choice = st.radio(
            "Segment untuk peta state",
            options=["All customers", "Sultan only"],
            index=0,
            horizontal=True,
            key="q5_segment_choice",
        )
        top_states_n = st.slider(
            "Jumlah state terbaik yang ditampilkan",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            key="q5_top_states_n",
        )

        with st.spinner("Menghitung potensi state..."):
            q5_results = compute_state_potential(
                rfm_df=q4_q5["rfm_df"],
                customers_df=dfs["customers_df"],
                geo_df=dfs["geo_df"],
                segment_choice=segment_choice,
                top_states_n=top_states_n,
            )

        st.caption(
            f"State terbaik (avg RFM tertinggi): **{q5_results['best_state']}** "
            f"(avg_rfm={q5_results['best_rfm']:.4f}, total_customers={q5_results['best_count']})"
        )
        st.plotly_chart(q5_results["q5_fig"], use_container_width=True)
        st.dataframe(q5_results["top_states"], use_container_width=True)


if __name__ == "__main__":
    main()

