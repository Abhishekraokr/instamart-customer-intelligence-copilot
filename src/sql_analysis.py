"""src/sql_analysis.py -- Part A Tasks 8 & 9: SQL Analysis and Pandas Cross-Checks
Executes queries Q1-Q7 from sql/queries.sql against data/instamart.db,
validates results against checkpoints, performs pandas aggregations, and computes distribution statistics.
"""
import os
import sqlite3
import numpy as np
import pandas as pd

import shutil
import tempfile

def get_connection(db_path):
    try:
        temp_dir = tempfile.gettempdir()
        temp_db = os.path.join(temp_dir, f"instamart_read_{os.getpid()}.db")
        shutil.copy2(db_path, temp_db)
        return sqlite3.connect(temp_db)
    except Exception:
        return sqlite3.connect(db_path)

def run_sql_queries():
    db_path = os.path.join("data", "instamart.db")
    sql_path = os.path.join("sql", "queries.sql")
    conn = get_connection(db_path)

    with open(sql_path, "r", encoding="utf-8") as f:
        full_sql = f.read()

    # Strip comments first so semicolons in comments never cause spurious splits
    clean_lines = []
    for line in full_sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        clean_lines.append(line)
    clean_sql = "\n".join(clean_lines)
    queries = [q.strip() for q in clean_sql.split(";") if q.strip()]

    print(f"Loaded {len(queries)} queries from {sql_path}")

    # Q1: WHERE then HAVING
    df_q1 = pd.read_sql_query(queries[0], conn)
    print("\n--- Q1: WHERE then HAVING ---")
    print(f"Customer count: {len(df_q1)} (Expected: 30)")
    assert len(df_q1) == 30, f"Q1 failed: expected 30, got {len(df_q1)}"

    # Q2: Three-table join
    df_q2 = pd.read_sql_query(queries[1], conn)
    total_rev_q2 = round(df_q2["revenue"].sum(), 2)
    bengaluru_electronics = df_q2[(df_q2["city"] == "Bengaluru") & (df_q2["category"] == "Electronics")]["revenue"].values[0]
    print("\n--- Q2: Three-Table Join (City x Category) ---")
    print(f"Row count: {len(df_q2)} (Expected: 30)")
    print(f"Total revenue: INR {total_rev_q2:,.2f} (Expected: INR 2,998,668.80)")
    print(f"Bengaluru / Electronics: INR {bengaluru_electronics:,.2f} (Expected: INR 233,909.45)")
    assert len(df_q2) == 30, f"Q2 row count mismatch: {len(df_q2)}"
    assert abs(total_rev_q2 - 2998668.80) < 0.01, f"Q2 total rev mismatch: {total_rev_q2}"
    assert abs(bengaluru_electronics - 233909.45) < 0.01, f"Q2 Bengaluru/Electronics mismatch: {bengaluru_electronics}"

    # Q3: Scalar subquery
    df_q3 = pd.read_sql_query(queries[2], conn)
    print("\n--- Q3: Products above average catalogue price ---")
    print(f"Product count: {len(df_q3)} (Expected: 11)")
    assert len(df_q3) == 11, f"Q3 count mismatch: {len(df_q3)}"

    # Q4: LEFT JOIN
    df_q4 = pd.read_sql_query(queries[3], conn)
    print("\n--- Q4: Customers with no completed order ---")
    print(f"Customer count: {len(df_q4)} (Expected: 25)")
    assert len(df_q4) == 25, f"Q4 count mismatch: {len(df_q4)}"

    # Q5: CTE
    df_q5 = pd.read_sql_query(queries[4], conn)
    # Compute CTE average spend
    cur = conn.cursor()
    cur.execute("""
        WITH customer_spend AS (
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
        )
        SELECT ROUND(AVG(total_spend), 2) FROM customer_spend;
    """)
    avg_spend = cur.fetchone()[0]
    print("\n--- Q5: CTE Spend Above Average ---")
    print(f"Above-average customer count: {len(df_q5)} (Expected: 550)")
    print(f"Average completed spend: INR {avg_spend} (Expected: INR 2,552.06)")
    assert len(df_q5) == 550, f"Q5 count mismatch: {len(df_q5)}"
    assert abs(avg_spend - 2552.06) < 0.01, f"Q5 avg spend mismatch: {avg_spend}"

    # Q6: Window Functions
    df_q6 = pd.read_sql_query(queries[5], conn)
    # Top customer per city
    top_per_city = df_q6[df_q6["rnk"] == 1].set_index("city")["customer_id"].to_dict()
    top_spend_per_city = df_q6[df_q6["rnk"] == 1].set_index("city")["total_spend"].to_dict()
    diff_rank_row = (df_q6["rnk"] != df_q6["row_num"]).sum()
    print("\n--- Q6: Ranking Window Functions ---")
    print("Top customer per city:")
    for city, cid in sorted(top_per_city.items()):
        print(f"  {city}: {cid} (INR {top_spend_per_city[city]:,.2f})")
    print(f"Rows where RANK() != ROW_NUMBER(): {diff_rank_row} (Expected: 8)")
    assert top_per_city == {"Bengaluru": "C0082", "Chennai": "C0930", "Delhi": "C0291",
                            "Hyderabad": "C0743", "Mumbai": "C0693", "Pune": "C0426"}, "Top customers mismatch"
    assert diff_rank_row == 8, f"Difference count mismatch: {diff_rank_row}"

    # Q7: Running Total
    df_q7 = pd.read_sql_query(queries[6], conn)
    last_cum_rev = df_q7["cumulative_revenue"].iloc[-1]
    print("\n--- Q7: Running Daily Revenue ---")
    print(f"Daily rows count: {len(df_q7)} (Expected: 183)")
    print(f"Final cumulative revenue: INR {last_cum_rev:,.2f} (Expected: INR 2,998,668.80)")
    assert len(df_q7) == 183, f"Q7 rows mismatch: {len(df_q7)}"
    assert abs(last_cum_rev - 2998668.80) < 0.01, f"Q7 final cumulative mismatch: {last_cum_rev}"

    conn.close()
    return {
        "q1_count": len(df_q1),
        "q2_total": total_rev_q2,
        "q3_count": len(df_q3),
        "q4_count": len(df_q4),
        "q5_count": len(df_q5),
        "q5_avg": avg_spend,
        "q6_ties": diff_rank_row,
        "q7_rows": len(df_q7),
        "q7_cum": last_cum_rev,
    }

def run_pandas_eda():
    print("\n=== PANDAS CROSS-CHECKS AND EXPLORATORY ANALYSIS ===")
    orders = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
    customers = pd.read_csv(os.path.join("data", "clean", "customers.csv"))
    products = pd.read_csv(os.path.join("data", "clean", "products.csv"))

    # Join clean tables
    merged = orders.merge(customers, on="customer_id").merge(products, on="product_id")
    completed = merged[merged["status"] == "completed"].copy()

    # (a) Rebuild Q2 with pd.pivot_table(..., margins=True)
    pivot = pd.pivot_table(
        completed,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="Total"
    ).round(2)
    grand_total = pivot.loc["Total", "Total"]
    print(f"Pivot table grand total: INR {grand_total:,.2f} (Matches SQL total: INR 2,998,668.80)")
    assert abs(grand_total - 2998668.80) < 0.01

    # City-level aggregation
    city_summary = completed.groupby("city").agg(
        total_amount=("order_amount", "sum"),
        mean_amount=("order_amount", "mean"),
        order_count=("order_amount", "count")
    ).round(2)
    print("\nCity Aggregations:")
    print(city_summary)

    # (b) Completed order_amount statistics
    amounts = completed["order_amount"]
    mean_val = round(amounts.mean(), 2)
    median_val = round(amounts.median(), 2)
    skewness = round(amounts.skew(), 2)
    q1 = round(amounts.quantile(0.25), 2)
    q3 = round(amounts.quantile(0.75), 2)
    iqr = round(q3 - q1, 2)
    lower_fence = round(q1 - 1.5 * iqr, 2)
    upper_fence = round(q3 + 1.5 * iqr, 2)
    outliers_above = (amounts > upper_fence).sum()
    outliers_below = (amounts < lower_fence).sum()

    print(f"\nOrder Amount Statistics:")
    print(f"  Mean: {mean_val}")
    print(f"  Median: {median_val}")
    print(f"  Skewness: {skewness}")
    print(f"  Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"  Lower Fence: {lower_fence} (Outliers: {outliers_below})")
    print(f"  Upper Fence: {upper_fence} (Outliers: {outliers_above})")

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed[corr_cols].corr().round(3)
    print("\nCorrelation Matrix:")
    print(corr_matrix)

if __name__ == "__main__":
    run_sql_queries()
    run_pandas_eda()
