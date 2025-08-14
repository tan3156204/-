import streamlit as st
import math
from dataclasses import dataclass, asdict
from typing import Dict, List
import pandas as pd
from collections import deque

st.set_page_config(page_title="Communi‑Pharm Advanced — Multi‑Round", layout="wide")

# ===== City & Locations =====
LOCATIONS = ["MEDICAL_CENTER", "NEIGHBORHOOD", "SHOPPING_CENTER"]
CITY_DEMAND = {
    "MEDICAL_CENTER":   {"rx_scripts": 2200, "other_units": 2600},
    "NEIGHBORHOOD":     {"rx_scripts": 1800, "other_units": 3400},
    "SHOPPING_CENTER":  {"rx_scripts": 1000, "other_units": 4200},
}
LOCATION_BASE_UTILITY = {"MEDICAL_CENTER":0.20, "NEIGHBORHOOD":0.10, "SHOPPING_CENTER":0.00}

# ===== Economic Parameters =====
BASE = {"rx_ingredient_cost":300.0, "other_unit_cost":120.0}
ELASTIC = {
    # utility coefficients
    "beta_price_rx":-0.004, "beta_price_other":-0.0015, "beta_hours":0.010, "beta_promo":0.00003,
    "beta_service":0.05, "beta_thirdparty":0.06, "beta_hmo":0.09,
    # payments & losses
    "expiry_loss_pct_of_cogs":0.004,
}
POLICY = {
    # AR policy
    "credit_sales_share_if_enabled":0.30,  # portion of sales to AR if store offers credit
    "ar_collection_rate":0.50,             # fraction of opening AR collected this round
    "ar_interest_income_annual_pct":12.0,  # interest charged to customers (annual)
    # Card settlement policy
    "card_share":0.55,
    "card_fee_pct":0.018,
    "card_settlement_lag_rounds":1,        # lag: 0 = cash now; 1 = receive net next round
    # Short term borrowing policy
    "min_cash_buffer":100000.0,            # default minimum cash target if store doesn't set its own
    "st_borrow_annual_pct":10.0,           # overdraft APR when topped up
    # Payroll & benefits
    "weeks_per_round":4.33,
    "benefit_life_per_emp":300.0,
    "benefit_health_per_emp":600.0,
}

# ===== Decisions (36-like) =====
@dataclass
class Decisions:
    location: str
    rx_markup_pct: float
    rx_fee_thb: float
    rx_copay_discount_thb: float
    delivery: int
    patient_records: int
    store_credit: int
    hours_per_week: float
    promo_budget_thb: float
    promo_rx_pct: float
    invest_thb: float
    invest_project: int
    withdraw_thb: float
    withdraw_project: int
    other_markup_pct: float
    rx_purchases_thb: float
    other_purchases_thb: float
    n_pharmacists: int
    pharm_pay_rate: float
    n_clerks: int
    clerk_pay_rate: float
    manager_salary: float
    manager_time_rx_pct: float
    manager_hours_per_week: float
    mortgage_payment: float
    sent_to_collections: float
    min_cash_balance: float
    rx_returns_thb: float
    other_returns_thb: float
    ap_payment_thb: float
    lt_debt_written_thb: float
    lt_debt_payment_thb: float
    ar_interest_rate_pct: float
    life_insurance: int
    health_insurance: int
    third_party: int
    hmo: int

# ===== Financial State carried between rounds =====
@dataclass
class StoreState:
    cash: float = 200000.0
    ar: float = 0.0
    ap: float = 0.0
    inventory_value: float = 200000.0
    lt_debt: float = 0.0
    fixed_assets: float = 0.0
    # settlement pipeline for card receipts: queue of net amounts to be received in future rounds
    card_pipeline: deque = deque()

def rx_effective_price(d: Decisions) -> float:
    return max(BASE["rx_ingredient_cost"]*(1+d.rx_markup_pct/100.0)+d.rx_fee_thb-d.rx_copay_discount_thb, 0.0)
def other_price(d: Decisions) -> float:
    return BASE["other_unit_cost"]*(1+d.other_markup_pct/100.0)
def service_score(d: Decisions) -> float:
    return d.delivery + d.patient_records + d.store_credit

def utility_rx(d: Decisions) -> float:
    u = LOCATION_BASE_UTILITY[d.location]
    u += ELASTIC["beta_price_rx"]*rx_effective_price(d)
    u += ELASTIC["beta_hours"]*d.hours_per_week
    u += ELASTIC["beta_promo"]*d.promo_budget_thb*(1+0.5*d.promo_rx_pct/100.0)
    u += ELASTIC["beta_service"]*service_score(d)
    u += ELASTIC["beta_thirdparty"]*d.third_party
    u += ELASTIC["beta_hmo"]*d.hmo
    return u

def utility_other(d: Decisions) -> float:
    u = LOCATION_BASE_UTILITY[d.location]
    u += ELASTIC["beta_price_other"]*other_price(d)
    u += ELASTIC["beta_hours"]*d.hours_per_week
    u += ELASTIC["beta_promo"]*d.promo_budget_thb
    u += ELASTIC["beta_service"]*service_score(d)
    return u

def softmax_shares(utilities: Dict[str, float]) -> Dict[str, float]:
    if not utilities: return {}
    mx = max(utilities.values())
    exps = {k: math.exp(v - mx) for k, v in utilities.items()}
    s = sum(exps.values())
    return {k:(v/s if s>0 else 0.0) for k,v in exps.items()}

def simulate_round(decisions: Dict[str, Decisions], states: Dict[str, StoreState]):
    # ----- 1) Market share per location -----
    by_loc = {loc:[name for name,d in decisions.items() if d.location==loc] for loc in LOCATIONS}
    shares_rx, shares_oth = {}, {}
    for loc in LOCATIONS:
        util_rx = {name: utility_rx(decisions[name]) for name in by_loc[loc]}
        util_ot = {name: utility_other(decisions[name]) for name in by_loc[loc]}
        shares_rx[loc] = softmax_shares(util_rx)
        shares_oth[loc] = softmax_shares(util_ot)

    results = {}
    for name, d in decisions.items():
        stt = states[name]
        loc = d.location

        # ----- 2) Demand allocation -----
        rx_scripts   = CITY_DEMAND[loc]["rx_scripts"] * shares_rx[loc].get(name,0.0)
        other_units  = CITY_DEMAND[loc]["other_units"] * shares_oth[loc].get(name,0.0)
        rx_p = rx_effective_price(d);  oth_p = other_price(d)
        rx_sales = rx_scripts * rx_p
        other_sales = other_units * oth_p
        sales_total = rx_sales + other_sales

        # COGS recognized
        cogs_rx = rx_scripts * BASE["rx_ingredient_cost"]
        cogs_oth = other_units * BASE["other_unit_cost"]
        cogs = cogs_rx + cogs_oth
        expiry_loss = cogs * ELASTIC["expiry_loss_pct_of_cogs"]

        # ----- 3) Revenue recognition & receipts policy -----
        card_share = POLICY["card_share"]
        card_fee_pct = POLICY["card_fee_pct"]
        card_net = sales_total * card_share * (1 - card_fee_pct)
        cash_sales = sales_total * (1 - card_share)

        # Credit sales if store offers credit
        credit_share = POLICY["credit_sales_share_if_enabled"] if d.store_credit==1 else 0.0
        credit_amount = cash_sales * credit_share
        cash_sales -= credit_amount

        # AR collections + AR interest this round
        ar_collections = stt.ar * POLICY["ar_collection_rate"]
        stt.ar -= ar_collections
        ar_interest_income = 0.0
        if d.store_credit==1:
            ar_interest_income = stt.ar * (d.ar_interest_rate_pct/100.0) / 12.0
            stt.ar += ar_interest_income
        # New AR
        stt.ar += credit_amount

        # Card settlement lag: receive previous queued amount(s)
        card_receipts_now = 0.0
        if not hasattr(stt, "card_pipeline") or stt.card_pipeline is None:
            stt.card_pipeline = deque()
        # receive head of queue
        if len(stt.card_pipeline)>0:
            card_receipts_now = stt.card_pipeline.popleft()
        # enqueue current net amount
        lag = int(POLICY["card_settlement_lag_rounds"])
        if lag==0:
            card_receipts_enqueue = 0.0
            card_receipts_now += card_net
        else:
            # ensure queue length
            while len(stt.card_pipeline) < lag-1:
                stt.card_pipeline.append(0.0)
            stt.card_pipeline.append(card_net)

        # ----- 4) Inventory and payables -----
        net_purchases = d.rx_purchases_thb + d.other_purchases_thb - d.rx_returns_thb - d.other_returns_thb
        stt.inventory_value += net_purchases - cogs - expiry_loss
        stt.ap += max(net_purchases,0)

        ap_payment = min(d.ap_payment_thb, stt.ap)
        stt.ap -= ap_payment

        # ----- 5) OPEX: wages + benefits + promo + mortgage -----
        weeks = POLICY["weeks_per_round"]
        labor_cash = d.n_pharmacists*d.pharm_pay_rate*d.hours_per_week*weeks + \
                     d.n_clerks*d.clerk_pay_rate*d.hours_per_week*weeks + \
                     d.manager_salary
        benefits = (d.life_insurance*POLICY["benefit_life_per_emp"] + d.health_insurance*POLICY["benefit_health_per_emp"]) * (d.n_pharmacists + d.n_clerks)
        promo_cost = d.promo_budget_thb
        mortgage = d.mortgage_payment

        # ----- 6) Investments / LT debt -----
        stt.fixed_assets += d.invest_thb - d.withdraw_thb
        invest_cf = -d.invest_thb + d.withdraw_thb
        stt.lt_debt = max(stt.lt_debt - d.lt_debt_written_thb - d.lt_debt_payment_thb, 0.0)
        lt_debt_cf = -d.lt_debt_payment_thb

        # ----- 7) Cash flows this round -----
        operating_cash_in = cash_sales + card_receipts_now + ar_collections + ar_interest_income
        operating_cash_out = ap_payment + labor_cash + benefits + promo_cost + mortgage
        cash_delta = operating_cash_in - operating_cash_out + invest_cf + lt_debt_cf
        stt.cash += cash_delta

        # Minimum cash enforcement with short-term borrowing cost
        target_min = max(d.min_cash_balance, POLICY["min_cash_buffer"])
        overdraft_topup = 0.0
        overdraft_interest = 0.0
        if stt.cash < target_min:
            overdraft_topup = target_min - stt.cash
            stt.cash += overdraft_topup
            overdraft_interest = overdraft_topup * (POLICY["st_borrow_annual_pct"]/100.0) / 12.0
            stt.cash -= overdraft_interest  # pay interest immediately

        # Sent to collection agency: reduce AR (no immediate cash)
        sent_to_agency = min(d.sent_to_collections, stt.ar)
        stt.ar -= sent_to_agency

        # ----- 8) Profit statements (accrual) -----
        revenue = sales_total
        merchant_fee_exp = sales_total * card_share * card_fee_pct  # expense when sale happens
        gp = revenue - cogs - expiry_loss
        opex_total = labor_cash + benefits + promo_cost + mortgage
        operating_profit = gp - (opex_total + merchant_fee_exp)
        profit_before_interest = operating_profit
        interest_net = overdraft_interest - ar_interest_income  # expense - income
        net_profit = profit_before_interest - interest_net

        # ----- 9) Build statements -----
        pnl = {
            "Revenue": revenue, "COGS": cogs, "ExpiryLoss": expiry_loss, "GrossProfit": gp,
            "Payroll": labor_cash, "Benefits": benefits, "Promo": promo_cost, "Mortgage": mortgage,
            "MerchantFee": merchant_fee_exp, "OperatingProfit": operating_profit,
            "InterestExpense(ST)": overdraft_interest, "InterestIncome(AR)": ar_interest_income,
            "NetProfit": net_profit
        }

        bs = {
            "Cash": stt.cash, "AR": stt.ar, "Inventory": stt.inventory_value, "AP": stt.ap,
            "LT_Debt": stt.lt_debt, "FixedAssets": stt.fixed_assets
        }

        cf = {
            "CashSales": cash_sales, "CardReceipts": card_receipts_now, "AR_Collections": ar_collections,
            "AR_InterestIn": ar_interest_income, "AP_Payment": -ap_payment, "Payroll": -labor_cash,
            "Benefits": -benefits, "Promo": -promo_cost, "Mortgage": -mortgage,
            "Investments": invest_cf, "LT_DebtPay": lt_debt_cf, "OverdraftTopUp": overdraft_topup,
            "OverdraftInterest": -overdraft_interest, "NetCashChange": cash_delta - overdraft_interest + overdraft_topup
        }

        results[name] = {
            "location": loc,
            "u_rx": round(utility_rx(d),4), "u_other": round(utility_other(d),4),
            "rx_scripts": rx_scripts, "other_units": other_units,
            "rx_price": rx_p, "other_price": oth_p,
            "statements": {"P&L": pnl, "BS": bs, "CF": cf}
        }

    return results, states

# ===== UI =====
st.title("🧮 Communi‑Pharm Advanced — Multi‑Round (P&L · BS · CF)")

# Global knobs
st.sidebar.header("Global Parameters")
BASE["rx_ingredient_cost"] = st.sidebar.number_input("Rx Ingredient Cost", 50.0, 1000.0, BASE["rx_ingredient_cost"], 10.0)
BASE["other_unit_cost"]    = st.sidebar.number_input("Other Unit Cost", 20.0, 1000.0, BASE["other_unit_cost"], 5.0)
POLICY["card_share"]       = st.sidebar.slider("Card Share", 0.0, 1.0, POLICY["card_share"], 0.01)
POLICY["card_fee_pct"]     = st.sidebar.number_input("Card Fee %", 0.0, 0.1, POLICY["card_fee_pct"], 0.001, format="%.3f")
POLICY["card_settlement_lag_rounds"] = st.sidebar.number_input("Card Settlement Lag (rounds)", 0, 6, POLICY["card_settlement_lag_rounds"], 1)
POLICY["credit_sales_share_if_enabled"] = st.sidebar.slider("Credit Sales Share", 0.0, 1.0, POLICY["credit_sales_share_if_enabled"], 0.05)
POLICY["ar_collection_rate"] = st.sidebar.slider("AR Collection Rate", 0.0, 1.0, POLICY["ar_collection_rate"], 0.05)
POLICY["st_borrow_annual_pct"] = st.sidebar.number_input("Short‑term Borrow APR %", 0.0, 50.0, POLICY["st_borrow_annual_pct"], 0.5)
POLICY["min_cash_buffer"] = st.sidebar.number_input("Default Min Cash Buffer", 0.0, 5_000_000.0, POLICY["min_cash_buffer"], 10000.0)

st.sidebar.subheader("Utility Coefficients")
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
    ELASTIC[key] = st.sidebar.number_input(key, mn, mx, ELASTIC[key], 0.001, format="%.3f")

# Session state
if "round" not in st.session_state:
    st.session_state.round = 1
if "states" not in st.session_state:
    st.session_state.states = {f"Store_{i+1}": StoreState() for i in range(7)}

st.write(f"### Round {st.session_state.round} — Enter Decisions (7 Stores)")

# Defaults for 7 stores
defaults = [
    ("MEDICAL_CENTER",22,15,0,1,1,0,60,20000,5,0,0,0,0,35,300000,200000,2,350,2,120,50000,40,40,0,0,100000,0,0,200000,0,0,2.0,0,0,1,0),
    ("MEDICAL_CENTER",25,20,0,1,0,0,60,20000,5,0,0,0,0,35,300000,200000,2,350,2,120,50000,40,40,0,0,100000,0,0,200000,0,0,2.0,0,0,1,0),
    ("MEDICAL_CENTER",28,15,5,1,1,1,64,30000,10,0,0,0,0,40,300000,200000,2,350,2,120,50000,40,40,0,0,100000,0,0,200000,0,0,2.0,1,1,1,1),
    ("NEIGHBORHOOD",20,10,0,0,1,0,56,10000,0,0,0,0,0,30,100000, 80000,2,350,1,120,30000,30,40,0,0,100000,0,0, 50000,0,0,2.0,0,0,0,0),
    ("NEIGHBORHOOD",24,15,0,1,0,1,60,20000,5,0,0,0,0,35,200000,150000,2,350,2,120,40000,40,40,0,0,100000,0,0,100000,0,0,2.0,0,0,1,0),
    ("SHOPPING_CENTER",26,25,10,1,1,1,72,30000,10,0,0,0,0,40,300000,200000,2,350,2,120,60000,40,40,0,0,100000,0,0,200000,0,0,2.0,1,1,1,0),
    ("SHOPPING_CENTER",18,10,0,0,0,0,56,10000,0,0,0,0,0,30, 80000, 60000,1,350,1,120,25000,20,30,0,0, 50000,0,0, 30000,0,0,2.0,0,0,0,0),
]

labels = [
    "location","rx_markup_pct","rx_fee_thb","rx_copay_discount_thb","delivery","patient_records","store_credit","hours_per_week",
    "promo_budget_thb","promo_rx_pct","invest_thb","invest_project","withdraw_thb","withdraw_project","other_markup_pct",
    "rx_purchases_thb","other_purchases_thb","n_pharmacists","pharm_pay_rate","n_clerks","clerk_pay_rate","manager_salary",
    "manager_time_rx_pct","manager_hours_per_week","mortgage_payment","sent_to_collections","min_cash_balance","rx_returns_thb",
    "other_returns_thb","ap_payment_thb","lt_debt_written_thb","lt_debt_payment_thb","ar_interest_rate_pct","life_insurance",
    "health_insurance","third_party","hmo"
]

stores: Dict[str, Decisions] = {}
cols = st.columns(7)
for i in range(7):
    with cols[i]:
        st.markdown(f"**Store_{i+1}**")
        vals = list(defaults[i])
        d = {}
        d["location"] = st.selectbox("Location", LOCATIONS, index=LOCATIONS.index(vals[0]), key=f"loc{i}")
        d["rx_markup_pct"] = st.number_input("Rx Markup %", 0.0, 300.0, float(vals[1]), 1.0, key=f"rxm{i}")
        d["rx_fee_thb"] = st.number_input("Rx Fee", 0.0, 1000.0, float(vals[2]), 1.0, key=f"rxf{i}")
        d["rx_copay_discount_thb"] = st.number_input("Rx Copay Discount", 0.0, 1000.0, float(vals[3]), 1.0, key=f"cop{i}")
        d["delivery"] = st.selectbox("Delivery", [0,1], index=int(vals[4]), key=f"delv{i}")
        d["patient_records"] = st.selectbox("Patient Records", [0,1], index=int(vals[5]), key=f"rec{i}")
        d["store_credit"] = st.selectbox("Store Credit", [0,1], index=int(vals[6]), key=f"cred{i}")
        d["hours_per_week"] = st.number_input("Hours/Week", 0.0, 120.0, float(vals[7]), 1.0, key=f"hrs{i}")
        d["promo_budget_thb"] = st.number_input("Promo Budget", 0.0, 1_000_000.0, float(vals[8]), 1000.0, key=f"promo{i}")
        d["promo_rx_pct"] = st.number_input("% Promo on Rx", 0.0, 100.0, float(vals[9]), 1.0, key=f"prx{i}")
        d["invest_thb"] = st.number_input("Invest THB", 0.0, 10_000_000.0, float(vals[10]), 10000.0, key=f"inv{i}")
        d["invest_project"] = st.number_input("Invest Project #", 0, 99, int(vals[11]), 1, key=f"invp{i}")
        d["withdraw_thb"] = st.number_input("Withdraw THB", 0.0, 10_000_000.0, float(vals[12]), 10000.0, key=f"wd{i}")
        d["withdraw_project"] = st.number_input("Withdraw Project #", 0, 99, int(vals[13]), 1, key=f"wdp{i}")
        d["other_markup_pct"] = st.number_input("Other Markup %", 0.0, 300.0, float(vals[14]), 1.0, key=f"om{i}")
        d["rx_purchases_thb"] = st.number_input("Rx Purchases THB", 0.0, 10_000_000.0, float(vals[15]), 10000.0, key=f"rxp{i}")
        d["other_purchases_thb"] = st.number_input("Other Purchases THB", 0.0, 10_000_000.0, float(vals[16]), 10000.0, key=f"otp{i}")
        d["n_pharmacists"] = st.number_input("# Pharmacists", 0, 20, int(vals[17]), 1, key=f"nph{i}")
        d["pharm_pay_rate"] = st.number_input("Pharm Pay Rate", 0.0, 5000.0, float(vals[18]), 10.0, key=f"phr{i}")
        d["n_clerks"] = st.number_input("# Sales Clerks", 0, 50, int(vals[19]), 1, key=f"ncl{i}")
        d["clerk_pay_rate"] = st.number_input("Clerk Pay Rate", 0.0, 2000.0, float(vals[20]), 10.0, key=f"clr{i}")
        d["manager_salary"] = st.number_input("Manager Salary", 0.0, 1_000_000.0, float(vals[21]), 1000.0, key=f"ms{i}")
        d["manager_time_rx_pct"] = st.number_input("Manager % Time Rx", 0.0, 100.0, float(vals[22]), 1.0, key=f"mt{i}")
        d["manager_hours_per_week"] = st.number_input("Mgr Hours/Week", 0.0, 120.0, float(vals[23]), 1.0, key=f"mh{i}")
        d["mortgage_payment"] = st.number_input("Mortgage Payment", 0.0, 1_000_000.0, float(vals[24]), 1000.0, key=f"mp{i}")
        d["sent_to_collections"] = st.number_input("Sent to Collection", 0.0, 1_000_000.0, float(vals[25]), 1000.0, key=f"col{i}")
        d["min_cash_balance"] = st.number_input("Min Cash Balance", 0.0, 5_000_000.0, float(vals[26]), 10000.0, key=f"mcb{i}")
        d["rx_returns_thb"] = st.number_input("Rx Returns THB", 0.0, 1_000_000.0, float(vals[27]), 1000.0, key=f"rxr{i}")
        d["other_returns_thb"] = st.number_input("Other Returns THB", 0.0, 1_000_000.0, float(vals[28]), 1000.0, key=f"otr{i}")
        d["ap_payment_thb"] = st.number_input("AP Payment THB", 0.0, 10_000_000.0, float(vals[29]), 10000.0, key=f"app{i}")
        d["lt_debt_written_thb"] = st.number_input("LT Debt Written", 0.0, 10_000_000.0, float(vals[30]), 10000.0, key=f"ltw{i}")
        d["lt_debt_payment_thb"] = st.number_input("LT Debt Payment", 0.0, 10_000_000.0, float(vals[31]), 10000.0, key=f"ltp{i}")
        d["ar_interest_rate_pct"] = st.number_input("AR Interest Rate %", 0.0, 100.0, float(vals[32]), 0.1, key=f"ari{i}")
        d["life_insurance"] = st.selectbox("Life Insurance (0/1)", [0,1], index=int(vals[33]), key=f"life{i}")
        d["health_insurance"] = st.selectbox("Health Insurance (0/1)", [0,1], index=int(vals[34]), key=f"hlth{i}")
        d["third_party"] = st.selectbox("Third-Party (0/1)", [0,1], index=int(vals[35]), key=f"tp{i}")
        d["hmo"] = st.selectbox("HMO (0/1)", [0,1], index=int(vals[36]), key=f"hmo{i}")
        stores[f"Store_{i+1}"] = Decisions(**d)

# Buttons
colA, colB, colC = st.columns(3)
if colA.button("Run Round"):
    res, st.session_state.states = simulate_round(stores, st.session_state.states)
    # Build tables
    pnl_rows, bs_rows, cf_rows = [], [], []
    overview_rows = []
    for name, v in res.items():
        pnl = v["statements"]["P&L"]; bs = v["statements"]["BS"]; cf = v["statements"]["CF"]
        pnl_rows.append({"Store":name, **pnl})
        bs_rows.append({"Store":name, **bs})
        cf_rows.append({"Store":name, **cf})
        overview_rows.append({"Store":name, "Location":v["location"], "Revenue":pnl["Revenue"], "NetProfit":pnl["NetProfit"]})
    st.success(f"Round {st.session_state.round} completed.")
    st.subheader("Overview")
    df_over = pd.DataFrame(overview_rows)
    st.dataframe(df_over, use_container_width=True)
    st.subheader("P&L (Accrual)")
    st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True)
    st.subheader("Balance Sheet (End of Round)")
    st.dataframe(pd.DataFrame(bs_rows), use_container_width=True)
    st.subheader("Cash Flow (This Round)")
    st.dataframe(pd.DataFrame(cf_rows), use_container_width=True)
    # Downloads
    st.download_button("Download P&L CSV", pd.DataFrame(pnl_rows).to_csv(index=False).encode("utf-8-sig"), "pnl.csv", "text/csv")
    st.download_button("Download BS CSV", pd.DataFrame(bs_rows).to_csv(index=False).encode("utf-8-sig"), "balance_sheet.csv", "text/csv")
    st.download_button("Download CF CSV", pd.DataFrame(cf_rows).to_csv(index=False).encode("utf-8-sig"), "cash_flow.csv", "text/csv")

if colB.button("Next Round ➡️"):
    st.session_state.round += 1
    st.success(f"Move to Round {st.session_state.round}")

if colC.button("Reset All ⛔"):
    st.session_state.round = 1
    st.session_state.states = {f"Store_{i+1}": StoreState() for i in range(7)}
    st.warning("All states reset")

st.caption("Advanced prototype: OPEX (wages+benefits), AR policy, card settlement lag, overdraft interest, and full statements (P&L/BS/CF).")
