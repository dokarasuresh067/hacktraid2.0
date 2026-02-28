# text_to_sql.py
# Technology : SQLite + rule-based SQL generation
# Algorithm  : Pattern matching → parameterized SQL queries

import sqlite3
import re

DB_PATH = "db/flipkard.db"

BRANDS = [
    "adidas", "nike", "dell", "sony", "lg", "puma",
    "boat", "redmi", "samsung", "apple", "hp", "lenovo",
]

CATEGORIES = [
    "electronics", "fashion", "appliances", "toys",
    "sports", "mobiles", "home & kitchen", "beauty",
]


def _extract_brand(q: str) -> str | None:
    for b in BRANDS:
        if b in q:
            return b.capitalize()
    return None


def _extract_category(q: str) -> str | None:
    for c in CATEGORIES:
        if c in q:
            return c.title()
    return None


def _extract_price(q: str) -> int | None:
    """Extract numeric price limit from question"""
    match = re.search(r'₹?\s*(\d[\d,]*)', q)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _build_where(brand=None, category=None) -> str:
    parts = []
    if brand:    parts.append(f"brand = '{brand}'")
    if category: parts.append(f"category = '{category}'")
    return f"WHERE {' AND '.join(parts)}" if parts else ""


def generate_sql(question: str) -> str:
    """
    Converts natural language → SQL query
    Technology: Rule-based pattern matching
    """
    q     = question.lower()
    brand = _extract_brand(q)
    cat   = _extract_category(q)
    price = _extract_price(q)
    where = _build_where(brand, cat)

    # ── COUNT ──
    if any(w in q for w in ["how many", "count", "number of", "total number"]):
        return f"SELECT COUNT(*) as total FROM products {where}"

    # ── AVERAGE ──
    if any(w in q for w in ["average rating", "avg rating"]):
        return f"SELECT ROUND(AVG(rating), 2) as avg_rating FROM products {where}"

    if any(w in q for w in ["average price", "avg price"]):
        return f"SELECT ROUND(AVG(final_price), 2) as avg_price FROM products {where}"

    if any(w in q for w in ["average discount", "avg discount"]):
        return f"SELECT ROUND(AVG(discount_percent), 2) as avg_discount FROM products {where}"

    # ── SUM ──
    if any(w in q for w in ["total stock", "total inventory"]):
        return f"SELECT SUM(stock_available) as total_stock FROM products {where}"

    if any(w in q for w in ["total sales", "total units sold", "units sold"]):
        return f"SELECT SUM(units_sold) as total_sold FROM products {where}"

    if any(w in q for w in ["total revenue", "total value"]):
        return f"SELECT ROUND(SUM(final_price * units_sold), 2) as total_revenue FROM products {where}"

    # ── PRICE FILTER ──
    if price and any(w in q for w in ["under", "below", "less than", "cheaper than"]):
        extra = f"AND final_price < {price}" if where else f"WHERE final_price < {price}"
        return (
            f"SELECT product_name, brand, final_price, rating "
            f"FROM products {where} {extra} "
            f"ORDER BY final_price ASC LIMIT 10"
        )

    if price and any(w in q for w in ["above", "over", "more than", "greater than"]):
        extra = f"AND final_price > {price}" if where else f"WHERE final_price > {price}"
        return (
            f"SELECT product_name, brand, final_price, rating "
            f"FROM products {where} {extra} "
            f"ORDER BY final_price DESC LIMIT 10"
        )

    # ── TOP / RANKING ──
    if any(w in q for w in ["most expensive", "highest price"]):
        return f"SELECT product_name, brand, final_price FROM products {where} ORDER BY final_price DESC LIMIT 5"

    if any(w in q for w in ["cheapest", "lowest price"]):
        return f"SELECT product_name, brand, final_price FROM products {where} ORDER BY final_price ASC LIMIT 5"

    if any(w in q for w in ["best rated", "highest rated", "top rated"]):
        return f"SELECT product_name, brand, rating FROM products {where} ORDER BY rating DESC LIMIT 5"

    if any(w in q for w in ["worst rated", "lowest rated"]):
        return f"SELECT product_name, brand, rating FROM products {where} ORDER BY rating ASC LIMIT 5"

    if any(w in q for w in ["best selling", "most sold", "top selling", "popular"]):
        return f"SELECT product_name, brand, units_sold FROM products {where} ORDER BY units_sold DESC LIMIT 5"

    if any(w in q for w in ["highest discount", "most discount", "best deal"]):
        return f"SELECT product_name, brand, discount_percent, final_price FROM products {where} ORDER BY discount_percent DESC LIMIT 5"

    # ── LIST ALL ──
    if any(w in q for w in ["list all", "show all", "all products"]):
        return f"SELECT product_name, brand, final_price, rating FROM products {where} LIMIT 20"

    # ── DEFAULT fallback ──
    return f"SELECT COUNT(*) as total FROM products {where}"


def run_sql(sql: str) -> list[dict]:
    """Execute SQL against SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]


# def format_answer(question: str, results: list[dict], sql: str) -> str:
def format_answer(results: list[dict], question: str = "") -> str:
    """Convert SQL results → clean human readable answer"""

    if not results:
        return "No products found matching your query."

    if "error" in results[0]:
        return f"Query error: {results[0]['error']}"

    row = results[0]

    # Single value responses
    if "total" in row:
        return f"There are **{int(row['total'])}** products matching your query."

    if "avg_rating" in row:
        return f"The average rating is **{row['avg_rating']} / 5**."

    if "avg_price" in row:
        return f"The average price is **₹{row['avg_price']:,}**."

    if "avg_discount" in row:
        return f"The average discount is **{row['avg_discount']}%**."

    if "total_stock" in row:
        return f"Total stock available: **{int(row['total_stock']):,} units**."

    if "total_sold" in row:
        return f"Total units sold: **{int(row['total_sold']):,}**."

    if "total_revenue" in row:
        return f"Total revenue generated: **₹{row['total_revenue']:,}**."

    # Multi-row table response
    lines = []
    for i, r in enumerate(results, 1):
        parts = []
        if "product_name"    in r: parts.append(f"**{r['product_name']}**")
        if "brand"           in r: parts.append(f"{r['brand']}")
        if "final_price"     in r: parts.append(f"₹{r['final_price']:,}")
        if "rating"          in r: parts.append(f"⭐ {r['rating']}")
        if "units_sold"      in r: parts.append(f"{r['units_sold']:,} sold")
        if "discount_percent" in r: parts.append(f"{r['discount_percent']}% off")
        lines.append(f"{i}. {' — '.join(str(p) for p in parts)}")

    return "\n".join(lines)
