
import os
import streamlit as st
import pandas as pd
import numpy as np
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
import requests
from bs4 import BeautifulSoup

# 자동 테마 설정
config_path = ".streamlit/config.toml"
if not os.path.exists(config_path):
    os.makedirs(".streamlit", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("[theme]\nbase = \"auto\"\n")

# 기본 설정
CANDIDATE_PLATFORMS = ["eBay", "Grailed", "번개장터", "중고나라"]
PLATFORM_FEES = {"eBay": 0.1, "Grailed": 0.09, "번개장터": 0.05, "중고나라": 0.03}
PLATFORM_SHIPPING = {"eBay": 20, "Grailed": 15, "번개장터": 7, "중고나라": 5}
SEED_DATA = [
    {"title": "KAWS Figure", "category": "figure", "price": 250, "platform": "Grailed", "days_to_sell": 7},
    {"title": "Murakami Print", "category": "print", "price": 180, "platform": "eBay", "days_to_sell": 10}
]

# 언어 및 텍스트
lang = st.sidebar.selectbox("🌐 Language", ["ko", "en"])
TXT = {
    "title": {"ko": "🎨 아트 리셀 대시보드", "en": "🎨 Art Resell Dashboard"},
    "predict_btn": {"ko": "🔮 추천 & 최적화", "en": "🔮 Recommend & Optimise"},
    "rec_tab": {"ko": "추천 플랫폼", "en": "Recommended Platform"},
    "list_tab": {"ko": "📈 트렌딩 아트 작품 (eBay)", "en": "📈 Trending Art on eBay"},
}
TXT = {k: v[lang] if isinstance(v, dict) else v for k, v in TXT.items()}

# 페이지 구성
st.set_page_config(page_title=TXT["title"], layout="wide")
st.title(TXT["title"])

# 입력 UI
in_title = st.sidebar.text_input("🎨 작품 제목", "Takashi Murakami Flower")
in_price = st.sidebar.number_input("💰 구매 가격 ($)", value=300)
budget = st.sidebar.number_input("📦 예산 ($)", value=1000)
clicked = st.sidebar.button(TXT["predict_btn"])

# 함수 정의
def expected_sale_price(price): return price * 1.5

def net_profit(price, sale_price, fee_rate, ship_cost):
    return sale_price * (1 - fee_rate) - price - ship_cost

def train_clf(df):
    X = TfidfVectorizer(token_pattern=r"(?u)\b[\w가-힣]+\b").fit_transform(df["title"])
    y = df["platform"]
    return RidgeClassifierCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)

def train_reg(df):
    X, y = df[["price"]], df["days_to_sell"]
    return GradientBoostingRegressor().fit(X, y), None

def fetch_ebay_art_listings(keyword):
    try:
        url = f"https://www.ebay.com/sch/i.html?_nkw={keyword.replace(' ', '+')}&LH_Sold=1"
        soup = BeautifulSoup(requests.get(url, timeout=10).text, "html.parser")
        return [x.get_text(strip=True) for x in soup.select(".s-item__title")[:5]]
    except Exception:
        return ["(크롤링 실패)"]

def notify_ifttt(title, profit, platform): pass

def prepare_data(df_raw):
    if df_raw is None or len(df_raw) == 0:
        return pd.DataFrame(SEED_DATA)
    return df_raw

# 데이터 준비
df_up = None
df = prepare_data(df_up)

# 메인 로직
if clicked:
    clf = train_clf(df)
    reg, mae = train_reg(df)

    col1, col2 = st.columns([2, 3])

    with col1:
        prob = clf.predict_proba(pd.DataFrame([[in_title, in_price]], columns=["title", "price"]))[0]
        best_idx = np.argmax(prob)
        best_platform = clf.classes_[best_idx]
        sale_price = expected_sale_price(in_price)
        profit = net_profit(in_price, sale_price, PLATFORM_FEES[best_platform], PLATFORM_SHIPPING[best_platform])
        days_pred = int(reg.predict([[in_price]])[0]) if reg else "-"
        days_txt = f"{days_pred}일" if lang == "ko" else f"{days_pred} days"
        explanation = f"""**{best_platform}** {'에서 판매 추천 🙌' if lang=='ko' else 'recommended'}  
- 예상 판매가: **${sale_price:.0f}**  
- 순이익: **${profit:.0f}**  
- 예상 판매 기간: **{days_txt}**"""
        st.markdown(f"<div class='card'><h3>{TXT['rec_tab']}</h3>{explanation}</div>", unsafe_allow_html=True)

    with col2:
        rows = []
        for p in CANDIDATE_PLATFORMS:
            sale = expected_sale_price(in_price)
            prof = net_profit(in_price, sale, PLATFORM_FEES[p], PLATFORM_SHIPPING[p])
            rows.append({"Platform": p, "Net profit": prof, "Fee%": PLATFORM_FEES[p], "Ship $": PLATFORM_SHIPPING[p]})
        st.dataframe(pd.DataFrame(rows).sort_values("Net profit", ascending=False), use_container_width=True)

    st.markdown("### 💡 예산 내 추천")
    filt = df[df["price"] <= budget].copy()
    if filt.empty:
        st.info("❗ 추천 아이템이 없습니다" if lang == "ko" else "No items within budget")
    else:
        filt["net_profit"] = filt.apply(lambda r: net_profit(r.price, expected_sale_price(r.price),
                                    PLATFORM_FEES.get(r.platform, 0.1), PLATFORM_SHIPPING.get(r.platform, 20)), axis=1)
        st.dataframe(filt.sort_values("net_profit", ascending=False).head(5), use_container_width=True)

    st.markdown(f"### {TXT['list_tab']}")
    for item in fetch_ebay_art_listings(in_title):
        st.write("•", item)

st.caption("© 2025 ArtResell AI — powered by Streamlit")
