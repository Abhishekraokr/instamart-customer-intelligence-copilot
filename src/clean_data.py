"""src/clean_data.py -- Part A Data Foundation: Clean, Store, Validate
Audits raw data, executes the exact documented cleaning rules, stores the clean
tables in data/clean/ and data/instamart.db, and validates reference checkpoints.
"""
import os
import sqlite3
import numpy as np
import pandas as pd

def audit_raw_data():
    raw_dir = os.path.join("data", "raw")
    orders_path = os.path.join(raw_dir, "orders.csv")
    customers_path = os.path.join(raw_dir, "customers.csv")
    products_path = os.path.join(raw_dir, "products.csv")
    tickets_path = os.path.join(raw_dir, "tickets.csv")

    orders = pd.read_csv(orders_path, keep_default_na=False)
    customers = pd.read_csv(customers_path, keep_default_na=False)
    products = pd.read_csv(products_path, keep_default_na=False)
    tickets = pd.read_csv(tickets_path, keep_default_na=False)

    print("=== RAW DATA AUDIT ===")
    print(f"orders.csv: shape={orders.shape}")
    print(f"customers.csv: shape={customers.shape}")
    print(f"products.csv: shape={products.shape}")
    print(f"tickets.csv: shape={tickets.shape}\n")

    # Audit calculations
    raw_orders_rows = len(orders)
    exact_duplicates = orders.duplicated().sum()
    pending_rows = (orders["status"] == "pending").sum()
    distinct_order_ids = orders["order_id"].nunique()

    # Non-numeric amounts
    non_numeric_amount = 0
    for v in orders["order_amount"]:
        try:
            float(v)
        except ValueError:
            non_numeric_amount += 1

    # Delivery minutes checks
    negative_deliv = 0
    blank_deliv = 0
    for v in orders["delivery_minutes"]:
        v_str = str(v).strip()
        if v_str == "":
            blank_deliv += 1
        else:
            try:
                if float(v_str) < 0:
                    negative_deliv += 1
            except ValueError:
                pass

    # Customer audit
    blank_age = 0
    outside_age = 0
    for v in customers["age"]:
        v_str = str(v).strip()
        if v_str == "":
            blank_age += 1
        else:
            try:
                num = float(v_str)
                if num < 10 or num > 100:
                    outside_age += 1
            except ValueError:
                blank_age += 1

    distinct_raw_cities = customers["city"].nunique()
    raw_genders = [str(g).strip() for g in customers["gender"]]
    distinct_nonblank_genders = len(set(g for g in raw_genders if g != ""))
    blank_gender = sum(1 for g in raw_genders if g == "")

    audit_metrics = {
        "raw_orders_rows": raw_orders_rows,
        "exact_duplicates": exact_duplicates,
        "pending_rows": pending_rows,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amount": non_numeric_amount,
        "negative_deliv": negative_deliv,
        "blank_deliv": blank_deliv,
        "blank_age": blank_age,
        "outside_age": outside_age,
        "distinct_raw_cities": distinct_raw_cities,
        "distinct_nonblank_genders": distinct_nonblank_genders,
        "blank_gender": blank_gender,
    }

    print("Audit Metrics:")
    for k, v in audit_metrics.items():
        print(f"  {k}: {v}")

    return audit_metrics

def clean_data():
    raw_dir = os.path.join("data", "raw")
    clean_dir = os.path.join("data", "clean")
    os.makedirs(clean_dir, exist_ok=True)

    products = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    customers = pd.read_csv(os.path.join(raw_dir, "customers.csv"), keep_default_na=False)
    orders = pd.read_csv(os.path.join(raw_dir, "orders.csv"), keep_default_na=False)

    # 1. Clean customers
    # (a) Standardise city
    customers["city"] = customers["city"].astype(str).str.strip().str.title()

    # (b) Map every gender spelling
    def map_gender(g):
        g_clean = str(g).strip().lower()
        if g_clean in ["male", "m"]:
            return "male"
        elif g_clean in ["female", "f"]:
            return "female"
        else:
            return "unknown"

    customers["gender"] = customers["gender"].apply(map_gender)

    # (c) Treat ages outside 10-100 as missing, then fill with median age of customer's city
    def parse_age(a):
        try:
            val = float(a)
            if 10 <= val <= 100:
                return val
            return np.nan
        except (ValueError, TypeError):
            return np.nan

    customers["age"] = customers["age"].apply(parse_age)
    city_medians = customers.groupby("city")["age"].transform("median")
    customers["age"] = customers["age"].fillna(city_medians).astype(int)

    # (d) Convert signup_date to datetime
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    # 2. Clean orders
    # i. Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # ii. Remove exact duplicate rows
    orders = orders.drop_duplicates().copy()

    # iii. Keep only record with latest order_ts per order_id
    orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last").copy()

    # iv. Convert order_amount to numeric, turning unparseable into missing, fill from product price
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    price_map = dict(zip(products["product_id"], products["price_inr"]))
    orders["price_inr"] = orders["product_id"].map(price_map)
    calc_amount = (orders["price_inr"] * orders["quantity"].astype(int) * (1.0 - orders["discount_pct"].astype(float) / 100.0)).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(calc_amount)

    # v. Set negative delivery_minutes to missing
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

    # For completed orders, fill missing delivery_minutes with median delivery time of that order's product category
    cat_map = dict(zip(products["product_id"], products["category"]))
    orders["category"] = orders["product_id"].map(cat_map)

    completed_mask = orders["status"] == "completed"
    category_medians = orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(category_medians)
    # Cancelled orders remain NaN for delivery_minutes

    # Format orders for export
    orders_clean = orders[["order_id", "customer_id", "product_id", "quantity",
                           "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]].copy()

    # Format dates as string for storage
    customers_export = customers.copy()
    customers_export["signup_date"] = customers_export["signup_date"].dt.strftime("%Y-%m-%d")

    orders_export = orders_clean.copy()
    orders_export["order_ts"] = orders_export["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Write clean CSVs
    customers_export.to_csv(os.path.join(clean_dir, "customers.csv"), index=False)
    products.to_csv(os.path.join(clean_dir, "products.csv"), index=False)
    orders_export.to_csv(os.path.join(clean_dir, "orders.csv"), index=False)

    # Write to SQLite via atomic local build to support network/WSL shares
    import tempfile, shutil
    db_path = os.path.join("data", "instamart.db")
    temp_dir = tempfile.gettempdir()
    temp_db_path = os.path.join(temp_dir, f"instamart_build_{os.getpid()}.db")
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # Create customers table
    cursor.execute("""
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            city TEXT,
            signup_date TEXT,
            age INTEGER,
            gender TEXT
        );
    """)
    cursor.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?);",
        customers_export[["customer_id", "city", "signup_date", "age", "gender"]].values.tolist()
    )

    # Create products table
    cursor.execute("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            category TEXT,
            price_inr REAL
        );
    """)
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?);",
        products[["product_id", "product_name", "category", "price_inr"]].values.tolist()
    )

    # Create orders table
    cursor.execute("""
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            product_id TEXT,
            quantity INTEGER,
            order_ts TEXT,
            status TEXT,
            discount_pct REAL,
            order_amount REAL,
            delivery_minutes REAL
        );
    """)
    orders_rows = orders_export[["order_id", "customer_id", "product_id", "quantity",
                                 "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]].copy()
    orders_rows["delivery_minutes"] = orders_rows["delivery_minutes"].where(pd.notna(orders_rows["delivery_minutes"]), None)
    cursor.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
        orders_rows.values.tolist()
    )

    conn.commit()
    conn.close()

    # Atomically copy to data/instamart.db
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
    shutil.copy2(temp_db_path, db_path)
    try:
        os.remove(temp_db_path)
    except OSError:
        pass

    print("\nClean data stored to data/clean/ and data/instamart.db")

def validate_checkpoints():
    print("\n=== RUNNING VALIDATION SCRIPT ===")
    clean_dir = os.path.join("data", "clean")
    customers = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))

    # Assert checkpoints
    # 1. Total order rows: 12,309
    assert len(orders) == 12309, f"Expected 12309 order rows, got {len(orders)}"
    # 2. Distinct order_id: 12,309
    assert orders["order_id"].nunique() == 12309, f"Distinct order_id mismatch"
    # 3. Status breakdown: 11,110 completed, 1,199 cancelled, 0 pending
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed", 0) == 11110, f"Expected 11110 completed, got {status_counts.get('completed')}"
    assert status_counts.get("cancelled", 0) == 1199, f"Expected 1199 cancelled, got {status_counts.get('cancelled')}"
    assert "pending" not in status_counts or status_counts["pending"] == 0, "Pending orders remain!"
    # 4. Zero missing order_amount
    assert orders["order_amount"].isna().sum() == 0, "Missing order_amount detected"
    # 5. Missing delivery_minutes: 0 among completed, 1199 among cancelled
    completed_deliv_na = orders[orders["status"] == "completed"]["delivery_minutes"].isna().sum()
    cancelled_deliv_na = orders[orders["status"] == "cancelled"]["delivery_minutes"].isna().sum()
    assert completed_deliv_na == 0, f"Expected 0 missing deliv in completed, got {completed_deliv_na}"
    assert cancelled_deliv_na == 1199, f"Expected 1199 missing deliv in cancelled, got {cancelled_deliv_na}"
    # 6. Sum of completed order_amount: INR 2,998,668.80
    completed_rev = round(orders[orders["status"] == "completed"]["order_amount"].sum(), 2)
    assert abs(completed_rev - 2998668.80) < 0.01, f"Expected INR 2998668.80, got {completed_rev}"
    # 7. Customer cities breakdown: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184
    city_counts = customers["city"].value_counts().to_dict()
    assert city_counts == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, f"City counts mismatch: {city_counts}"
    # 8. Customer genders breakdown: female 610, male 560, unknown 30
    gender_counts = customers["gender"].value_counts().to_dict()
    assert gender_counts == {"female": 610, "male": 560, "unknown": 30}, f"Gender counts mismatch: {gender_counts}"
    # 9. Ages: 0 missing, every age between 18 and 60
    assert customers["age"].isna().sum() == 0, "Missing age detected"
    assert (customers["age"] >= 18).all() and (customers["age"] <= 60).all(), "Ages outside [18, 60] detected"

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    audit_raw_data()
    clean_data()
    validate_checkpoints()
