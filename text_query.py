import sqlite3

conn = sqlite3.connect("db/flipkard.db")
cursor = conn.cursor()

# ✅ Execute SQL properly
cursor.execute("SELECT COUNT(*) FROM products")
result1 = cursor.fetchall()
print("Total rows:", result1)

cursor.execute("SELECT product_name, final_price FROM products LIMIT 5")
result2 = cursor.fetchall()
print("Sample products:", result2)

conn.close()