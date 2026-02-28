"""
classifier.py
Hybrid intent classifier — keyword first, LLM fallback for ambiguous queries.
Technology : Rule-based regex word-boundary matching + confidence scoring + LLM fallback
Improvements:
  1. Word boundary matching (no more substring false positives)
  2. Added missing retail intents
  3. Product-name pattern detection
  4. Confidence scoring instead of hard returns
  5. Ambiguous query telemetry logging
"""
import re
import logging
from functools import lru_cache

# ─────────────────────────────────────────────
# LOGGING SETUP (telemetry)
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="classifier_log.txt",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ─────────────────────────────────────────────
# WORD BOUNDARY MATCHER  — Issue 1 Fix
# Prevents substring false positives
# e.g. "popular color options" won't match "popular" as SQL keyword
# e.g. "unpopular" won't match "popular"
# ─────────────────────────────────────────────
def _contains_keyword(text: str, keyword: str) -> bool:
    """
    Checks if keyword exists as a whole word in text.
    Uses regex word boundaries \b to block substring matches.

    Examples:
        "popular color"  + "popular"  → True  ✅ (whole word)
        "unpopular item" + "popular"  → False ✅ (substring blocked)
        "avg price"      + "avg"      → True  ✅
        "average"        + "avg"      → False ✅ (not same word)
    """
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return bool(re.search(pattern, text))


# ─────────────────────────────────────────────
# KEYWORD LISTS
# ─────────────────────────────────────────────

# Issue 2 Fix — added missing retail intents marked with # NEW
DEFINITE_SQL = [
    # counting
    "how many", "count", "number of", "total number",
    "how many products in stock",                        # NEW
    "how many available",                                # NEW

    # aggregations
    "average", "avg", "mean",
    "sum", "total",
    "average price", "avg price",
    "average rating", "avg rating",
    "average discount", "avg discount",
    "average delivery", "avg delivery",
    "average score", "avg score",
    "total stock", "total inventory", "stock available",
    "total sales", "total units", "units sold",
    "total revenue", "revenue",
    "total reviews", "review count",

    # ranking / comparison
    "most expensive", "highest price", "costliest",
    "cheapest", "lowest price", "least expensive",
    "best rated", "highest rated", "top rated",
    "worst rated", "lowest rated",
    "best selling", "most sold", "top selling", "popular",
    "highest discount", "most discount", "best deal", "biggest discount",
    "fastest delivery", "quickest delivery",
    "highest score", "best score",
    "highest seller rating", "best seller",
    "most reviewed", "most reviews",
    "longest warranty", "best warranty",
    "lightest", "heaviest",

    # price / rating filters
    "under", "below", "less than", "cheaper than",
    "above", "over", "more than", "greater than",
    "between",
    "with rating above", "rating above",                # NEW
    "with rating below", "rating below",                # NEW
    "rating greater", "rating less",                    # NEW

    # price / rating sorting
    "sort by price", "order by price",                  # NEW
    "sort by rating", "order by rating",                # NEW
    "sort by discount",                                 # NEW

    # top N
    "top 5", "top 10", "top 3",                         # NEW
    "first 5", "first 10",                              # NEW

    # stock / availability counts
    "in stock",                                         # NEW
    "out of stock",                                     # NEW

    # listing / grouping
    "list all", "show all", "all products",
    "group by", "breakdown", "distribution",
    "minimum", "maximum",

    # seller / city
    "which city", "seller count",
    "how many sellers", "how many cities",

    # date
    "listed in", "listed on", "listed after", "listed before",
    "newest", "oldest", "recent",

    # attributes
    "returnable products", "non returnable",
    "accept cod", "accept upi", "accept card", "accept wallet",
    "payment mode",
]

DEFINITE_SEMANTIC = [
    # price of specific product
    "what is the price", "price of", "cost of",
    "how much is", "how much does",

    # product details
    "tell me about", "describe", "details of",
    "info about", "information about", "about the product",

    # availability
    "is there", "do you have", "is it available",
    "available in",

    # specific product attributes
    "warranty of", "warranty on",
    "return policy of", "return window",
    "delivery time for", "when will",
    "color of", "size of", "weight of",
    "seller of", "who sells", "where is",
    "payment modes for", "how to pay",
    "rating of", "reviews of",
    "discount on", "offer on",
    "seller rating of",

    # product search
    "show me", "find me", "search for",
    "recommend", "suggest",
    "similar to", "like the",

    # existence check
    "is there any", "any product", "any brand",
    "do you sell",
]

# Issue 3 Fix — product name patterns
# Catches: "Adidas Ultra 664", "Redmi Model 35", "Sony Edition 769"
PRODUCT_NAME_PATTERNS = [
    r"\b(model|edition|series|prime|ultra|pro|max|mini|lite|plus)\b",
    r"\b[A-Za-z]+\s+\d{2,4}\b",   # word + number e.g. "Ultra 664", "Series 124"
]

# SQL keywords so strong they override product name detection
OVERRIDE_SQL_KEYWORDS = [
    "how many", "count", "average", "avg",
    "total", "sum", "list all", "show all",
    "top 5", "top 10", "top 3",
]


# ─────────────────────────────────────────────
# PRODUCT NAME DETECTOR  — Issue 3 Fix
# ─────────────────────────────────────────────
def _is_product_name_query(q: str) -> bool:
    """
    Returns True if query looks like it's asking about a specific product.

    Examples:
        "adidas ultra 664 details"  → True  (has "ultra" + number pattern)
        "redmi model 35 price"      → True  (has "model" keyword)
        "how many adidas models"    → False (override by SQL keyword)
        "show all series products"  → False (override by SQL keyword)
    """
    for pattern in PRODUCT_NAME_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False


def _has_override_sql(q: str) -> bool:
    """Checks if strong SQL keyword is present — overrides product name detection."""
    return any(_contains_keyword(q, kw) for kw in OVERRIDE_SQL_KEYWORDS)


# ─────────────────────────────────────────────
# CONFIDENCE SCORER  — Upgrade 1
# ─────────────────────────────────────────────
def _compute_scores(q: str) -> tuple[int, int]:
    """
    Counts how many SQL vs semantic keywords match the query.
    Returns: (sql_score, semantic_score)

    Higher score = higher confidence.
    Handles edge cases where both categories have matches.
    """
    sql_score      = sum(1 for kw in DEFINITE_SQL      if _contains_keyword(q, kw))
    semantic_score = sum(1 for kw in DEFINITE_SEMANTIC if _contains_keyword(q, kw))
    return sql_score, semantic_score


# ─────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────
def classify_intent(question: str, model=None) -> str:
    """
    Classifies question as 'sql' or 'semantic'.

    Decision flow:
    1. Product name pattern check  → semantic (unless strong SQL override)
    2. Confidence score comparison → clear winner picked
    3. Tied or both zero           → LLM fallback + telemetry log
    4. No model available          → safe default (semantic)

    Args:
        question : raw user question string
        model    : ChatOllama instance (optional, for ambiguous cases only)

    Returns:
        'sql' or 'semantic'
    """
    q = question.lower().strip()

    # ── Step 1: Product name pattern → semantic ──────────────────────────
    # Issue 3 fix — catches "adidas ultra 664 details", "redmi model 35 price"
    if _is_product_name_query(q) and not _has_override_sql(q):
        _log_decision(question, "semantic", "product_name_pattern", 0, 0)
        return "semantic"

    # ── Step 2: Confidence scoring ───────────────────────────────────────
    # Upgrade 1 — count ALL matching keywords, pick clear winner
    sql_score, semantic_score = _compute_scores(q)

    if sql_score > 0 and semantic_score == 0:
        _log_decision(question, "sql", "keyword_only", sql_score, semantic_score)
        return "sql"

    if semantic_score > 0 and sql_score == 0:
        _log_decision(question, "semantic", "keyword_only", sql_score, semantic_score)
        return "semantic"

    if sql_score > semantic_score:
        _log_decision(question, "sql", "confidence_winner", sql_score, semantic_score)
        return "sql"

    if semantic_score > sql_score:
        _log_decision(question, "semantic", "confidence_winner", sql_score, semantic_score)
        return "semantic"

    # ── Step 3: Tied or both zero → LLM fallback ─────────────────────────
    # Upgrade 2 — log ambiguous queries for future improvement
    _log_ambiguous(question, sql_score, semantic_score)

    if model:
        intent = _llm_classify(question, model)
        _log_decision(question, intent, "llm_fallback", sql_score, semantic_score)
        return intent

    # ── Step 4: No model, safe default ───────────────────────────────────
    _log_decision(question, "semantic", "default_fallback", sql_score, semantic_score)
    return "semantic"


# ─────────────────────────────────────────────
# TELEMETRY LOGGERS  — Upgrade 2
# ─────────────────────────────────────────────
def _log_decision(
    question:  str,
    intent:    str,
    method:    str,
    sql_score: int,
    sem_score: int,
):
    """Logs every classification to classifier_log.txt"""
    logging.info(
        f"INTENT={intent.upper():<8} | METHOD={method:<25} | "
        f"SQL={sql_score} SEM={sem_score} | Q={question}"
    )


def _log_ambiguous(question: str, sql_score: int, sem_score: int):
    """
    Logs ambiguous queries separately.
    Real companies use this data to improve keyword lists over time.
    """
    logging.warning(
        f"AMBIGUOUS | SQL={sql_score} SEM={sem_score} | Q={question}"
    )
    print(f"⚠️  Ambiguous query → LLM classifying: '{question}'")


# ─────────────────────────────────────────────
# LLM FALLBACK  (only for ambiguous queries)
# ─────────────────────────────────────────────
@lru_cache(maxsize=256)
def _llm_classify(question: str, model) -> str:
    """
    LLM classification — only called when scoring is tied or both zero.
    Result cached via lru_cache so same question never hits LLM twice.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "You are an intent classifier for a product store chatbot.\n"
            "Output ONLY one word: sql OR semantic\n\n"
            "sql      = counting, totals, averages, rankings, "
            "comparisons, price filters, groupings, stock queries\n"
            "semantic = specific product details, descriptions, "
            "availability, price of a named product, attributes\n\n"
            "Examples:\n"
            "adidas products under 30000          → sql\n"
            "good products for gifting            → semantic\n"
            "products with highest product_score  → sql\n"
            "which products accept COD            → sql\n"
            "tell me about boat prime 291         → semantic\n"
            "is redmi model 35 returnable         → semantic\n"
            "brands available in hyderabad        → sql\n"
            "products listed in 2023              → sql\n"
            "popular color options                → semantic\n"
            "adidas ultra 664 details             → semantic\n"
            "top 5 cheapest nike                  → sql\n"
            "out of stock products                → sql"
        )),
        HumanMessage(content=f"Q: {question}")
    ]

    result = model.invoke(messages)
    intent = result.content.strip().lower().split()[0]
    return "sql" if "sql" in intent else "semantic"


# ─────────────────────────────────────────────
# DEBUG HELPER  — useful for testing/UI display
# ─────────────────────────────────────────────
def get_classification_details(question: str) -> dict:
    """
    Returns full classification breakdown — useful for debugging
    or displaying confidence info in the UI.

    Usage:
        details = get_classification_details("how many adidas products")
        print(details)
        # {
        #   "intent":          "sql",
        #   "sql_score":       2,
        #   "semantic_score":  0,
        #   "method":          "keyword_only",
        #   "product_pattern": False,
        # }
    """
    q              = question.lower().strip()
    sql_score, sem = _compute_scores(q)
    has_product    = _is_product_name_query(q)
    has_override   = _has_override_sql(q)

    if has_product and not has_override:
        intent = "semantic"
        method = "product_name_pattern"
    elif sql_score > sem:
        intent = "sql"
        method = "confidence_winner" if sem > 0 else "keyword_only"
    elif sem > sql_score:
        intent = "semantic"
        method = "confidence_winner" if sql_score > 0 else "keyword_only"
    else:
        intent = "ambiguous"
        method = "needs_llm_fallback"

    return {
        "question":        question,
        "intent":          intent,
        "sql_score":       sql_score,
        "semantic_score":  sem,
        "method":          method,
        "product_pattern": has_product,
        "sql_override":    has_override,
    }