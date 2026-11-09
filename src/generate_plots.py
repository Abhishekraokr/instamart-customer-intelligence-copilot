import os
import sys
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIRS = [
    os.path.join(BASE_DIR, "outputs", "plots")
]
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

orders_path = os.path.join(BASE_DIR, "data", "clean", "orders.csv")
cust_path = os.path.join(BASE_DIR, "data", "clean", "customers.csv")

customers = {}
with open(cust_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        customers[row["customer_id"]] = row["city"]

orders = []
with open(orders_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        orders.append(row)

completed_orders = [o for o in orders if o["status"] == "completed"]
order_amounts = [float(o["order_amount"]) for o in completed_orders]
delivery_times = [float(o["delivery_minutes"]) for o in completed_orders if o.get("delivery_minutes")]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

print(f"Loaded {len(completed_orders)} completed orders. Matplotlib available: {HAS_MATPLOTLIB}")

if HAS_MATPLOTLIB:
    plt.figure(figsize=(10, 6), dpi=150)
    plt.hist(order_amounts, bins=50, range=(0, 1500), color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.axvline(135.0, color="green", linestyle="--", linewidth=2, label="Median: INR 135.00")
    plt.axvline(269.91, color="red", linestyle="--", linewidth=2, label="Mean: INR 269.91")
    plt.axvline(645.30, color="orange", linestyle=":", linewidth=2, label="Upper Fence: INR 645.30 (897 outliers)")
    plt.title("Instamart Order Amount Distribution (Right-Skewed, Skewness = 4.61)", fontsize=14, fontweight="bold")
    plt.xlabel("Order Amount (INR)", fontsize=12)
    plt.ylabel("Number of Orders", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    for d in OUTPUT_DIRS:
        plt.savefig(os.path.join(d, "order_amount_distribution.png"))
    plt.close()

    daily_rev = {}
    for o in completed_orders:
        day = o["order_ts"][:10]
        daily_rev[day] = daily_rev.get(day, 0.0) + float(o["order_amount"])
    sorted_days = sorted(daily_rev.keys())
    cum_rev = []
    running = 0.0
    for d in sorted_days:
        running += daily_rev[d]
        cum_rev.append(running)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(range(len(sorted_days)), cum_rev, color="#2ca02c", linewidth=2.5, label="Cumulative Spend")
    plt.title(f"Cumulative Gross Revenue over 183 Days (Final: INR {running:,.2f})", fontsize=14, fontweight="bold")
    plt.xlabel("Days from 2025-04-01 to 2025-09-30", fontsize=12)
    plt.ylabel("Cumulative Revenue (INR)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    for d in OUTPUT_DIRS:
        plt.savefig(os.path.join(d, "daily_revenue_cumulative.png"))
    plt.close()

    plt.figure(figsize=(10, 6), dpi=150)
    plt.hist(delivery_times, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.7)
    mean_dt = sum(delivery_times) / len(delivery_times)
    plt.axvline(mean_dt, color="darkblue", linestyle="--", linewidth=2, label=f"Mean Delivery Time: {mean_dt:.2f} mins")
    plt.title("Instamart Delivery Time Distribution (Completed Orders)", fontsize=14, fontweight="bold")
    plt.xlabel("Delivery Time (Minutes)", fontsize=12)
    plt.ylabel("Number of Orders", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    for d in OUTPUT_DIRS:
        plt.savefig(os.path.join(d, "delivery_time_distribution.png"))
    plt.close()

    city_spend = {}
    for o in completed_orders:
        c = customers.get(o["customer_id"], "Unknown")
        city_spend[c] = city_spend.get(c, 0.0) + float(o["order_amount"])
    sorted_cities = sorted(city_spend.items(), key=lambda x: x[1], reverse=True)
    c_names = [x[0] for x in sorted_cities]
    c_vals = [x[1] for x in sorted_cities]

    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.bar(c_names, c_vals, color="#9467bd", edgecolor="black", alpha=0.8)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 5000, f"INR {h:,.0f}", ha="center", va="bottom", fontsize=10)
    plt.title("Total Spend by Metro City (INR)", fontsize=14, fontweight="bold")
    plt.xlabel("City", fontsize=12)
    plt.ylabel("Total Spend (INR)", fontsize=12)
    plt.ylim(0, max(c_vals) * 1.15)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    for d in OUTPUT_DIRS:
        plt.savefig(os.path.join(d, "city_spend_breakdown.png"))
    plt.close()

else:
    from PIL import Image, ImageDraw

    def create_hist_plot(values, out_name, title, xlabel, median_val, mean_val, fence_val):
        W, H = 1000, 600
        img = Image.new("RGB", (W, H), "#ffffff")
        draw = ImageDraw.Draw(img)

        lm, rm, tm, bm = 80, 50, 70, 70
        pw = W - lm - rm
        ph = H - tm - bm

        draw.rectangle([lm, tm, lm + pw, tm + ph], outline="#cccccc", fill="#fafafa")

        max_x = 1200.0 if "Order Amount" in title else 60.0
        nbins = 40
        bins = [0] * nbins
        for v in values:
            b = int(min(v, max_x - 1) / (max_x / nbins))
            bins[b] += 1
        max_count = max(bins)

        bin_w = pw / nbins
        for i, c in enumerate(bins):
            bh = (c / max_count) * (ph - 30)
            x0 = lm + i * bin_w
            y0 = tm + ph - bh
            x1 = x0 + bin_w - 1
            y1 = tm + ph
            draw.rectangle([x0, y0, x1, y1], fill="#1f77b4" if "Order Amount" in title else "#ff7f0e", outline="#104e7a")

        def x_pos(val):
            return lm + (min(val, max_x) / max_x) * pw

        xm = x_pos(median_val)
        draw.line([xm, tm, xm, tm + ph], fill="#2ca02c", width=3)
        draw.text((xm + 5, tm + 20), f"Median: {median_val:.2f}", fill="#2ca02c")

        xmean = x_pos(mean_val)
        draw.line([xmean, tm, xmean, tm + ph], fill="#d62728", width=3)
        draw.text((xmean + 5, tm + 45), f"Mean: {mean_val:.2f}", fill="#d62728")

        if fence_val:
            xfence = x_pos(fence_val)
            draw.line([xfence, tm, xfence, tm + ph], fill="#ff7f0e", width=3)
            draw.text((xfence + 5, tm + 70), f"Upper Fence: {fence_val:.2f} (897 outliers)", fill="#ff7f0e")

        draw.text((W // 2 - 250, 25), title, fill="#111111")
        draw.text((W // 2 - 50, H - 40), xlabel, fill="#222222")
        draw.text((15, H // 2), "Count", fill="#222222")

        for d in OUTPUT_DIRS:
            img.save(os.path.join(d, out_name))

    create_hist_plot(order_amounts, "order_amount_distribution.png",
                     "Instamart Order Amount Distribution (Right-Skewed, Skewness = 4.61)",
                     "Order Amount (INR)", 135.0, 269.91, 645.30)

    daily_rev = {}
    for o in completed_orders:
        day = o["order_ts"][:10]
        daily_rev[day] = daily_rev.get(day, 0.0) + float(o["order_amount"])
    sorted_days = sorted(daily_rev.keys())
    cum_rev = []
    running = 0.0
    for d in sorted_days:
        running += daily_rev[d]
        cum_rev.append(running)

    W, H = 1000, 600
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)
    lm, rm, tm, bm = 90, 50, 70, 70
    pw, ph = W - lm - rm, H - tm - bm
    draw.rectangle([lm, tm, lm + pw, tm + ph], outline="#cccccc", fill="#fafafa")

    points = []
    max_rev = cum_rev[-1]
    for i, cr in enumerate(cum_rev):
        x = lm + (i / (len(cum_rev) - 1)) * pw
        y = tm + ph - (cr / max_rev) * ph
        points.append((x, y))
    draw.line(points, fill="#2ca02c", width=4)

    for tick in range(0, int(max_rev) + 1, 500000):
        y = tm + ph - (tick / max_rev) * ph
        draw.line([lm, y, lm + pw, y], fill="#e0e0e0", width=1)
        draw.text((10, y - 7), f"INR {tick//1000}k", fill="#555555")

    draw.text((W // 2 - 250, 25), f"Cumulative Gross Revenue over 183 Days (Final: INR {running:,.2f})", fill="#111111")
    draw.text((W // 2 - 50, H - 40), "Days (2025-04-01 to 2025-09-30)", fill="#222222")
    for d in OUTPUT_DIRS:
        img.save(os.path.join(d, "daily_revenue_cumulative.png"))

    create_hist_plot(delivery_times, "delivery_time_distribution.png",
                     "Instamart Delivery Time Distribution (Completed Orders)",
                     "Delivery Time (Minutes)", 28.0, 29.58, None)

    city_spend = {}
    for o in completed_orders:
        c = customers.get(o["customer_id"], "Unknown")
        city_spend[c] = city_spend.get(c, 0.0) + float(o["order_amount"])
    sorted_cities = sorted(city_spend.items(), key=lambda x: x[1], reverse=True)

    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)
    draw.rectangle([lm, tm, lm + pw, tm + ph], outline="#cccccc", fill="#fafafa")
    max_c_spend = max(x[1] for x in sorted_cities) * 1.15

    bw = pw / len(sorted_cities)
    for i, (city, spend) in enumerate(sorted_cities):
        bh = (spend / max_c_spend) * ph
        x0 = lm + i * bw + 20
        x1 = lm + (i + 1) * bw - 20
        y0 = tm + ph - bh
        y1 = tm + ph
        draw.rectangle([x0, y0, x1, y1], fill="#9467bd", outline="#663399")
        draw.text((x0, y0 - 20), f"INR {spend:,.0f}", fill="#333333")
        draw.text((x0 + 10, tm + ph + 10), city, fill="#222222")

    draw.text((W // 2 - 200, 25), "Total Spend by Metro City (INR)", fill="#111111")
    draw.text((10, H // 2), "Spend (INR)", fill="#222222")
    for d in OUTPUT_DIRS:
        img.save(os.path.join(d, "city_spend_breakdown.png"))

print("All 4 plots successfully generated and saved to part_1/plots/ and outputs/plots/!")
