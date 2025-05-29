# art_resell_dashboard/app.py
"""
Streamlit dashboard: ML‑based art resell platform recommender — **enhanced**
===========================================================================

🔹 **New in this version**
1. **Real transaction data** — upload a CSV of historical deals to retrain the model.
2. **Net‑profit optimisation** — fee + shipping dictionaries calculate platform net profit and rank them.
3. **Customisable UI** — tabbed layout, editable fee/sliders, light/dark aware.
4. **IFTTT webhook alerts** — receive KakaoTalk-compatible alerts via IFTTT when recommendations are made.
5. **Live eBay crawler** — fetch recent trending art listings on eBay for inspiration and sourcing.
6. **⏳ 판매 시기 예측** — 예상 판매 완료까지 걸리는 일수 표시
7. **💡 예산 내 최적 추천 리스트** — 입력 예산 내에서 가장 수익성 높은 아이템 5개 추천

> **How to run**
> ```bash
> pip install streamlit scikit-learn pandas numpy requests beautifulsoup4
> streamlit run app.py
> ```

CSV schema expected (header row required):
```
title,category,price,platform,days_to_sell
```
`price` → historical purchase price (USD). `days_to_sell` → 실제 판매까지 걸린 일수 (선택).
"""

import json
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import requests
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup

###############################################################################
# 📁 Paths & constants
###############################################################################
BASE_DIR = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.pkl"

# default platform fee (%) and shipping cost (USD) — editable in UI
PLATFORM_FEES: Dict[str, float] = {
    "eBay": 0.13,
    "Grailed": 0.09,
    "StockX": 0.125,
    "KREAM": 0.10,
    "번개장터": 0.07,
}
PLATFORM_SHIPPING: Dict[str, float] = {
    "eBay": 25,
    "Grailed": 22,
    "StockX": 28,
    "KREAM": 15,
    "번개장터": 10,
}
CANDIDATE_PLATFORMS: List[str] = list(PLATFORM_FEES.keys())

###############################################################################
# 🔔 IFTTT notification (KakaoTalk-compatible)
###############################################################################
IFTTT_KEY = "your_ifttt_key_here"  # replace with your Webhook key
def notify_ifttt(title: str, profit: float, platform: str):
    url = f"https://maker.ifttt.com/trigger/art_profit/with/key/{IFTTT_KEY}"
    data = {
        "value1": title,
        "value2": f"${profit:.2f}",
        "value3": platform,
    }
    try:
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        print("Notification failed:", e)

###############################################################################
# 🧠 Model helpers
###############################################################################

SEED_DATA = [
    {"title": "Takashi Murakami Signed Print Flower 2020", "category": "print", "price": 450, "platform": "Grailed", "days_to_sell": 14},
    {"title": "KAWS Companion Figure 20cm", "category": "figure", "price": 280, "platform": "번개장터", "days_to_sell": 10},
    {"title": "BTS RM Photo Book Limited", "category": "k-pop", "price": 70, "platform": "KREAM", "days_to_sell": 7},
    {"title": "Banksy Canvas 2005", "category": "canvas", "price": 1500, "platform": "eBay", "days_to_sell": 21},
    {"title": "Bearbrick 400% Andy Warhol", "category": "figure", "price": 450, "platform": "StockX", "days_to_sell": 12},
    {"title": "Keith Haring Skate Deck", "category": "print", "price": 320, "platform": "Grailed", "days_to_sell": 9},
]

def load_training_frame(uploaded: pd.DataFrame | None) -> pd.DataFrame:
    if uploaded is not None and not uploaded.empty:
        df = uploaded.copy()
    else:
        df = pd.DataFrame(SEED_DATA)
    return df.dropna(subset=["title", "price", "platform"])

def train_model(df: pd.DataFrame) -> Pipeline:
    X = df[["title", "price"]]
    y = df["platform"]
    pipe = Pipeline([
        ("pre", ColumnTransformer(
            transformers=[
                ("txt", TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b", max_features=600), ["title"]),
                ("price", StandardScaler(), ["price"]),
            ]
        )),
        ("clf", LogisticRegression(max_iter=1000, multi_class="ovr"))
    ])
    pipe.fit(X, y)
    return pipe

def train_sell_time_model(df: pd.DataFrame) -> Pipeline | None:
    if "days_to_sell" not in df.columns:
        return None
    df = df.dropna(subset=["days_to_sell"])
    X = df[["title", "price"]]
    y = df["days_to_sell"]
    pipe = Pipeline([
        ("pre", ColumnTransformer(
            transformers=[
                ("txt", TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b", max_features=600), ["title"]),
                ("price", StandardScaler(), ["price"]),
            ]
        )),
        ("reg", LinearRegression())
    ])
    pipe.fit(X, y)
    return pipe

def load_or_train(df: pd.DataFrame) -> Pipeline:
    return train_model(df)

###############################################################################
# 💰 Net‑profit helpers
###############################################################################

def expected_sale_price(purchase_price: float, markup: float = 1.35) -> float:
    return purchase_price * markup

def net_profit(purchase_price: float, sale_price: float, fee_rate: float, shipping: float) -> float:
    gross = sale_price * (1 - fee_rate)
    return gross - purchase_price - shipping

def best_platform(purchase_price: float) -> pd.DataFrame:
    rows = []
    sale_price = expected_sale_price(purchase_price)
    for p in CANDIDATE_PLATFORMS:
        fee = PLATFORM_FEES[p]
        ship = PLATFORM_SHIPPING[p]
        profit = net_profit(purchase_price, sale_price, fee, ship)
        rows.append({"platform": p, "net_profit": profit, "fee": fee, "shipping": ship})
    return pd.DataFrame(rows).sort_values("net_profit", ascending=False)

###############################################################################
# 🛒 eBay crawler
###############################################################################

def fetch_ebay_art_listings(query: str = "murakami art") -> List[str]:
    url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}&_sop=10"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        listings = [item.text.strip() for item in soup.select(".s-item__title") if "Shop on eBay" not in item.text]
        return listings[:10]
    except Exception:
        return ["Failed to fetch eBay listings"]

###############################################################################
# 🌐 Streamlit UI
###############################################################################

st.set_page_config(page_title="Art Resell Optimiser", layout="wide", page_icon="🎨")
st.title("🎨 Art Resell Optimiser — ML + Profit + ⏳ 판매시기 예측")

with st.sidebar:
    st.header("Upload training data (optional)")
    csv_file = st.file_uploader("CSV file", type="csv")
    uploaded_df = pd.read_csv(csv_file) if csv_file else None

    st.header("Edit platform fee & shipping")
    for p in CANDIDATE_PLATFORMS:
        PLATFORM_FEES[p] = st.number_input(f"{p} fee %", 0.0, 0.3, step=0.005, value=float(PLATFORM_FEES[p]))
        PLATFORM_SHIPPING[p] = st.number_input(f"{p} shipping $", 0.0, 50.0, step=1.0, value=float(PLATFORM_SHIPPING[p]))

    st.header("New item details")
    in_title = st.text_input("Title", value="Takashi Murakami offset print")
    in_category = st.selectbox("Category", ["print", "figure", "canvas", "k-pop", "other"], 0)
    in_price = st.number_input("Purchase price (USD)", min_value=1.0, step=1.0, value=350.0)
    predict_btn = st.button("🔮 Recommend & optimise")

train_df = load_training_frame(uploaded_df)
model = load_or_train(train_df)
sell_model = train_sell_time_model(train_df)

if predict_btn:
    tabs = st.tabs(["📈 Recommendation", "📊 Profit table", "🛒 eBay listings", "🗄 Training data"])

    pred_label = model.predict(pd.DataFrame([{"title": in_title, "price": in_price}]))[0]
    with tabs[0]:
        st.subheader("ML Recommendation")
        st.metric(label="Suggested platform", value=pred_label)

    profit_df = best_platform(in_price)
    best_row = profit_df.iloc[0]
    with tabs[1]:
        st.subheader("Net‑profit by platform (after fee + shipping)")
        st.dataframe(profit_df, use_container_width=True, hide_index=True)
        st.success(
            f"🏆 Highest net‑profit: **{best_row['platform']}** → ${best_row['net_profit']:.2f}",
            icon="💰",
        )

    notify_ifttt(in_title, best_row["net_profit"], best_row["platform"])

    with tabs[2]:
        st.subheader("eBay trending listings")
        listings = fetch_ebay_art_listings(in_title)
        for l in listings:
            st.markdown(f"- {l}")

    with tabs[3]:
        st.subheader("Training data preview & model accuracy")
        st.dataframe(train_df, use_container_width=True)
        acc = (model.predict(train_df[["title", "price"]]) == train_df["platform"]).mean()
        st.caption(f"Training accuracy: {acc:.2%} (small seed set)")

    if sell_model:
        days = sell_model.predict(pd.DataFrame([{"title": in_title, "price": in_price}]))[0]
        with tabs[0]:
            st.metric(label="⏳ 예상 판매 시기", value=f"{int(days)}일 이내")

    with tabs[0]:
        st.subheader("💡 예산 내 구매 추천 리스트")
        budget = st.number_input("예산 (USD)", min_value=1.0, step=10.0, value=1000.0)

        candidates = train_df.copy()
        candidates = candidates[candidates["price"] <= budget]

        def profit_row(row):
            est_sale = expected_sale_price(row["price"])
            fee = PLATFORM_FEES.get(row["platform"], 0.1)
            shipping = PLATFORM_SHIPPING.get(row["platform"], 20.0)
            return net_profit(row["price"], est_sale, fee, shipping)

        candidates["net_profit"] = candidates.apply(profit_row, axis=1)
        top_candidates = candidates.sort_values("net_profit", ascending=False).head(5)

        if top_candidates.empty:
            st.warning("예산 내 추천 가능한 아이템이 없습니다.")
        else:
            st.dataframe(
                top_candidates[["title", "category", "price", "platform", "net_profit"]],
                use_container_width=True,
                hide_index=True
            )

###############################################################################
# 📝  Footer
###############################################################################

st.markdown(
    """---  
    **Tip:** Replace the naive `expected_sale_price` function with real market‑price regression or recent sold‑price scraping for higher precision.  
    All fee & shipping sliders are saved only for the current session.
    """
)
