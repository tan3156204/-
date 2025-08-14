import streamlit as st
import math
from dataclasses import dataclass
from typing import Dict

st.set_page_config(page_title="City Pharmacy Simulator", layout="wide")

# ----- Model parameters -----
LOCATIONS = ["MEDICAL_CENTER", "NEIGHBORHOOD", "SHOPPING_CENTER"]

CITY_DEMAND = {
    "MEDICAL_CENTER":   {"rx_scripts": 2200, "other_units": 2600},
    "NEIGHBORHOOD":     {"rx_scripts": 1800, "other_units": 3400},
    "SHOPPING_CENTER":  {"rx_scripts": 1000, "other_units": 4200},
}

LOCATION_BASE_UTILITY = {
    "MEDICAL_CENTER":   0.20,
    "NEIGHBORHOOD":     0.10,
    "SHOPPING_CENTER":  0.00,
}

BASE = {
    "rx_ingredient_cost": 300.0,
    "other_unit_cost":    120.0,
}

ELASTIC = {
    "beta_price_rx":     -0.004,
    "beta_price_other":  -0.0015,
    "beta_hours":         0.010,
    "beta_promo":         0.00003,
    "beta_service":       0.05,
    "beta_thirdparty":    0.06,
    "beta_hmo":           0.09,
    "card_share": 0.55,
    "card_fee_pct": 0.018,
    "expiry_loss_pct_of_cogs": 0.004,
}

@dataclass
class Decisions:
    location: str
    rx_markup_pct: float
    rx_fee_thb: float
    rx_copay_discount_thb: float
    other_markup_pct: float
    hours_per_week: float
    promo_budget_thb: float
    delivery: int
    patient_records: int
    store_credit: int
    third_party: int
    hmo: int

def rx_effective_price(d: Decisions) -> float:
    return max(BASE["rx_ingredient_cost"] * (1 + d.rx_markup_pct/100.0) + d.rx_fee_thb - d.rx_copay_discount_thb, 0.0)

def other_price(d: Decisions) -> float:
    return BASE["other_unit_cost"] * (1 + d.other_markup_pct/100.0)

def service_score(d: Decisions) -> float:
    return d.delivery + d.patient_records + d.store_credit

def utility_rx(d: Decisions) -> float:
    u = LOCATION_BASE_UTILITY[d.location]
    u += ELASTIC["beta_price_rx"] * rx_effective_price(d)
    u += ELASTIC["beta_hours"] * d.hours_per_week
    u += ELASTIC["beta_promo"] * d.promo_budget_thb
    u += ELASTIC["beta_service"] * service_score(d)
    u += ELASTIC["beta_thirdparty"] * d.third_party
    u += ELASTIC["beta_hmo"] * d.hmo
    return u

def utility_other(d: Decisions) -> float:
    u = LOCATION_BASE_UTILITY[d.location]
    u += ELASTIC["beta_price_other"] * other_price(d)
    u += ELASTIC["beta_hours"] * d.hours_per_week
    u += ELASTIC["beta_promo"] * d.promo_budget_thb
    u += ELASTIC["beta_service"] * service_score(d)
    return u

def softmax_shares(utilities: Dict[str, float]) -> Dict[str, float]:
    mx = max(utilities.values()) if utilities else 0.0
    exps = {k: math.exp(u - mx) for k, u in utilities.items()}
    s = sum(exps.values())
    return {k: (v/s if s>0 else 0.0) for k, v in exps.items()}

def simulate_one_round(stores: Dict[str, Decisions]) -> Dict[str, Dict]:
    by_loc = {loc: [name for name, d in stores.items() if d.location == loc] for loc in LOCATIONS}
    shares_rx, shares_other = {}, {}
    for loc in LOCATIONS:
        util_rx = {name: utility_rx(stores[name]) for name in by_loc[loc]}
        util_ot = {name: utility_other(stores[name]) for name in by_loc[loc]}
        shares_rx[loc] = softmax_shares(util_rx) if util_rx else {}
        shares_other[loc] = softmax_shares(util_ot) if util_ot else {}

    results = {}
    for name, d in stores.items():
        loc = d.location
        rx_scripts = CITY_DEMAND[loc]["rx_scripts"] * shares_rx[loc].get(name, 0.0)
        other_units = CITY_DEMAND[loc]["other_units"] * shares_other[loc].get(name, 0.0)
        rx_p = rx_effective_price(d);  oth_p = other_price(d)
        rx_sales = rx_scripts * rx_p
        other_sales = other_units * oth_p
        cogs = rx_scripts * BASE["rx_ingredient_cost"] + other_units * BASE["other_unit_cost"]
        merchant_fee = (rx_sales + other_sales) * ELASTIC["card_share"] * ELASTIC["card_fee_pct"]
        promo_cost = d.promo_budget_thb
        expiry_loss = cogs * ELASTIC["expiry_loss_pct_of_cogs"]
        gross_profit = (rx_sales + other_sales) - cogs - expiry_loss
        net_profit = gross_profit - merchant_fee - promo_cost
        results[name] = {
            "location": loc, "rx_scripts": round(rx_scripts,2), "other_units": round(other_units,2),
            "rx_price": round(rx_p,2), "other_price": round(oth_p,2),
            "rx_sales": round(rx_sales,2), "other_sales": round(other_sales,2),
            "sales_total": round(rx_sales+other_sales,2), "cogs": round(cogs,2),
            "expiry_loss": round(expiry_loss,2), "merchant_fee": round(merchant_fee,2),
            "promo_cost": round(promo_cost,2), "gross_profit": round(gross_profit,2),
            "net_profit": round(net_profit,2),
            "u_rx": round(utility_rx(d),4), "u_other": round(utility_other(d),4)
        }
    return results

st.title("🏥 City Pharmacy Simulator — 1 เมือง / 7 ร้าน / 3 ทำเล")

# Sidebar: Global parameters
st.sidebar.header("Global Parameters")
BASE["rx_ingredient_cost"] = st.sidebar.number_input("Rx Ingredient Cost", 50.0, 1000.0, BASE["rx_ingredient_cost"], step=10.0)
BASE["other_unit_cost"]    = st.sidebar.number_input("Other Unit Cost", 20.0, 1000.0, BASE["other_unit_cost"], step=5.0)
ELASTIC["card_share"]      = st.sidebar.slider("Card Share", 0.0, 1.0, ELASTIC["card_share"], 0.01)
ELASTIC["card_fee_pct"]    = st.sidebar.number_input("Card Fee %", 0.0, 0.1, ELASTIC["card_fee_pct"], step=0.001, format="%.3f")
ELASTIC["expiry_loss_pct_of_cogs"] = st.sidebar.number_input("Expiry Loss % of COGS", 0.0, 0.1, ELASTIC["expiry_loss_pct_of_cogs"], step=0.001, format="%.3f")

st.sidebar.subheader("Utility Coefficients (Advanced)")
ranges = {
    "beta_price_rx": (-0.02, 0.00),
    "beta_price_other": (-0.02, 0.00),
    "beta_hours": (0.00, 0.05),
    "beta_promo": (0.00, 0.01),
    "beta_service": (0.00, 0.20),
    "beta_thirdparty": (0.00, 0.20),
    "beta_hmo": (0.00, 0.30),
}
for key in ["beta_price_rx","beta_price_other","beta_hours","beta_promo","beta_service","beta_thirdparty","beta_hmo"]:
    mn, mx = ranges[key]
    step = 0.001 if key != "beta_promo" else 0.0001
    ELASTIC[key] = st.sidebar.number_input(key, mn, mx, ELASTIC[key], step=step, format="%.3f")

# Main: Store controls
st.subheader("Stores — Decisions")
stores = {}
cols = st.columns(7)
for i in range(7):
    with cols[i]:
        st.markdown(f"**Store_{i+1}**")
        loc = st.selectbox("Location", LOCATIONS, key=f"loc{i}", index=0 if i<3 else (1 if i<5 else 2))
        rxm = st.number_input("Rx Markup %", 0.0, 100.0, float([22,25,28,20,24,26,18][i]), step=1.0, key=f"rxm{i}")
        rxf = st.number_input("Rx Fee", 0.0, 100.0, float([15,20,15,10,15,25,10][i]), step=1.0, key=f"rxf{i}")
        cop = st.number_input("Rx Copay Discount", 0.0, 100.0, float([0,0,5,0,0,10,0][i]), step=1.0, key=f"cop{i}")
        othm= st.number_input("Other Markup %", 0.0, 200.0, float([35,35,40,30,35,40,30][i]), step=1.0, key=f"othm{i}")
        hrs = st.number_input("Hours/Week", 0.0, 120.0, float([60,60,64,56,60,72,56][i]), step=1.0, key=f"hrs{i}")
        promo = st.number_input("Promo Budget", 0.0, 1000000.0, float([20000,20000,30000,10000,20000,30000,10000][i]), step=1000.0, key=f"promo{i}")
        delv = st.selectbox("Delivery", [0,1], index=[1,1,1,0,1,1,0][i], key=f"delv{i}")
        rec  = st.selectbox("Patient Records", [0,1], index=[1,0,1,1,0,1,0][i], key=f"rec{i}")
        cred = st.selectbox("Store Credit", [0,1], index=[0,0,1,0,1,1,0][i], key=f"cred{i}")
        tp   = st.selectbox("Third-Party", [0,1], index=[1,1,1,0,1,1,0][i], key=f"tp{i}")
        hmo  = st.selectbox("HMO", [0,1], index=[0,0,1,0,0,0,0][i], key=f"hmo{i}")
        stores[f"Store_{i+1}"] = Decisions(loc, rxm, rxf, cop, othm, hrs, promo, delv, rec, cred, tp, hmo)

# Run simulation
if st.button("Run Round Simulation"):
    res = simulate_one_round(stores)
    st.success("Simulation completed.")
    # Show per-store table
    import pandas as pd
    df = pd.DataFrame(res).T.reset_index().rename(columns={"index":"Store"})
    st.dataframe(df, use_container_width=True)
    # Summaries by location
    st.subheader("Summary by Location")
    summary = df.groupby("location")[["rx_scripts","other_units","sales_total","net_profit"]].sum().reset_index()
    st.dataframe(summary, use_container_width=True)
    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download results (CSV)", data=csv, file_name="round_results.csv", mime="text/csv")

st.caption("Tip: ปรับค่าสัมประสิทธิ์ในแถบ Sidebar เพื่อดูความไวของตลาดต่อราคา/ชั่วโมง/โปรโมชัน ฯลฯ")
