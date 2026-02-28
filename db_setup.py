"""
db_setup.py
Loads CSV into SQLite with correct column types.
Run once: python db_setup.py
"""
import os
import sqlite3
import pandas as pd


CSV_PATH = "docs/flipkard.csv"
DB_PATH  = "db/flipkard.db"

# ── Exact column → SQLite type mapping based on your data ──
COLUMN_TYPES = {
    "product_id"       : "TEXT",
    "product_name"     : "TEXT",
    "category"         : "TEXT",
    "brand"            : "TEXT",
    "seller"           : "TEXT",
    "seller_city"      : "TEXT",
    "price"            : "REAL",
    "discount_percent" : "REAL",
    "final_price"      : "REAL",
    "rating"           : "REAL",
    "review_count"     : "INTEGER",
    "stock_available"  : "INTEGER",
    "units_sold"       : "INTEGER",
    "listing_date"     : "TEXT",
    "delivery_days"    : "INTEGER",
    "weight_g"         : "REAL",
    "warranty_months"  : "INTEGER",
    "color"            : "TEXT",
    "size"             : "TEXT",
    "return_policy_days": "INTEGER",
    "is_returnable"    : "INTEGER",   # True/False → 1/0
    "payment_modes"    : "TEXT",
    "shipping_weight_g": "REAL",
    "product_score"    : "REAL",
    "seller_rating"    : "REAL",
}


def setup_sqlite(csv_path: str = CSV_PATH, db_path: str = DB_PATH):
    os.makedirs("db", exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # ── Type conversions ──
    df["is_returnable"]   = df["is_returnable"].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    ).fillna(0).astype(int)

    df["price"]           = pd.to_numeric(df["price"],           errors="coerce").fillna(0)
    df["discount_percent"]= pd.to_numeric(df["discount_percent"],errors="coerce").fillna(0)
    df["final_price"]     = pd.to_numeric(df["final_price"],     errors="coerce").fillna(0)
    df["rating"]          = pd.to_numeric(df["rating"],          errors="coerce").fillna(0)
    df["review_count"]    = pd.to_numeric(df["review_count"],    errors="coerce").fillna(0).astype(int)
    df["stock_available"] = pd.to_numeric(df["stock_available"], errors="coerce").fillna(0).astype(int)
    df["units_sold"]      = pd.to_numeric(df["units_sold"],      errors="coerce").fillna(0).astype(int)
    df["delivery_days"]   = pd.to_numeric(df["delivery_days"],   errors="coerce").fillna(0).astype(int)
    df["weight_g"]        = pd.to_numeric(df["weight_g"],        errors="coerce").fillna(0)
    df["warranty_months"] = pd.to_numeric(df["warranty_months"], errors="coerce").fillna(0).astype(int)
    df["return_policy_days"] = pd.to_numeric(df["return_policy_days"], errors="coerce").fillna(0).astype(int)
    df["shipping_weight_g"]  = pd.to_numeric(df["shipping_weight_g"],  errors="coerce").fillna(0)
    df["product_score"]   = pd.to_numeric(df["product_score"],   errors="coerce").fillna(0)
    df["seller_rating"]   = pd.to_numeric(df["seller_rating"],   errors="coerce").fillna(0)

    conn = sqlite3.connect(db_path)
    df.to_sql("products", conn, if_exists="replace", index=False)

    # ── Create indexes for fast filtering ──
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_brand    ON products(brand)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price    ON products(final_price)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating   ON products(rating)")
    conn.commit()
    conn.close()

    print(f"✅ SQLite ready at '{db_path}'")
    print(f"   Rows loaded : {len(df)}")
    print(f"   Columns     : {list(df.columns)}")
    print(f"   Indexes     : brand, category, final_price, rating")


if __name__ == "__main__":
    setup_sqlite()