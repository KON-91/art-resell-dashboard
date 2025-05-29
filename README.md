# 🎨 Art Resell Optimiser

AI‑powered Streamlit dashboard for buying art at the right price and reselling for maximum profit.

## Features
- 🇰🇷/🇺🇸 bilingual UI toggle
- Machine‑learning platform recommender (Tfidf + RidgeClassifierCV)
- Sell‑time predictor (GradientBoostingRegressor + CV)
- Net‑profit calculator (fees & shipping)
- eBay trending scraper
- Budget‑aware item suggestions
- Tailwind‑styled dark/light responsive design

## Quick start

```bash
git clone https://github.com/<your‑id>/art-resell-dashboard.git
cd art-resell-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## CSV schema

```csv
title,category,price,platform,days_to_sell
Takashi Murakami Signed Print Flower 2020,print,450,Grailed,14
KAWS Companion Figure 20cm,figure,280,번개장터,10
...
```

## Deploy on Streamlit Cloud
1. Connect GitHub repo.
2. Set **Main file path** to `app.py`.
3. Click *Deploy*.

MIT License
