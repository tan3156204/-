import streamlit as st
import math
from dataclasses import dataclass, asdict
from typing import Dict, List

st.set_page_config(page_title="Communi-Pharm Prototype — Multi-Round", layout="wide")

# ==== City & Location Setup ====
LOCATIONS = ["MEDICAL_CENTER", "NEIGHBORHOOD", "SHOPPING_CENTER"]
CITY_DEMAND = {
    "MEDICAL_CENTER":   {"rx_scripts": 2200, "other_units": 2600},
    "NEIGHBORHOOD":     {"rx_scripts": 1800, "other_units": 3400},
    "SHOPPING_CENTER":  {"rx_scripts": 1000, "other_units": 4200},
}
LOCATION_BASE_UTILITY = {"MEDICAL_CENTER":0.20,"NEIGHBORHOOD":0.10,"SHOPPING_CENTER":0.00}

# ==== Economic & Elastic Coefficients ====
BASE = {"rx_ingredient_cost":300.0,"other_unit_cost":120.0}
ELASTIC = {
    "beta_price_rx":-0.004,"beta_price_other":-0.0015,"beta_hours":0.010,"beta_promo":0.00003,
    "beta_service":0.05,"beta_thirdparty":0.06,"beta_hmo":0.09,
    "card_share":0.55,"card_fee_pct":0.018,"expiry_loss_pct_of_cogs":0.004,
    "credit_sales_share":0.30,  # ถ้าให้เครดิต: สัดส่วนยอดขายที่กลายเป็น AR
    "ar_collection_rate":0.50,  # อัตราเก็บหนี้ในรอบถัดไป
    "periods_per_year":12
}

# ==== Decision Form (36) ====
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

# ==== Store State ====
@dataclass
class StoreState:
    cash: float = 200000.0
    ar: float = 0.0
    ap: float = 0.0
    inventory_value: float = 200000.0
    lt_debt: float = 0.0
    fixed_assets: float = 0.0

# ==== Utility functions ====
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
def softmax_shares(utilities: Dict[str,float]) -> Dict[str,float]:
    if not utilities: return {}
    m = max(utilities.values())
    exps = {k: math.exp(v-m) for k,v in utilities.items()}
    s = sum(exps.values())
    return {k:(v/s if s>0 else 0.0) for k,v in exps.items()}

# ==== One Round Simulation ====
def simulate_round(decisions: Dict[str, Decisions], states: Dict[str, StoreState]):
    # Shares per location
    by_loc = {loc:[name for name,d in decisions.items() if d.location==loc] for loc in LOCATIONS}
    shares_rx, shares_oth = {}, {}
    for loc in LOCATIONS:
        util_rx = {name: utility_rx(decisions[name]) for name in by_loc[loc]}
        util_ot = {name: utility_other(decisions[name]) for name in by_loc[loc]}
        shares_rx[loc] = softmax_shares(util_rx)
        shares_oth[loc] = softmax_shares(util_ot)

    results = {}
    for name, d in decisions.items():
        s = states[name]
        loc = d.location

        # Demand allocation
        rx_scripts = CITY_DEMAND[loc]["rx_scripts"]*shares_rx[loc].get(name,0.0)
        other_units = CITY_DEMAND[loc]["other_units"]*shares_oth[loc].get(name,0.0)

        # Prices & sales
        rx_p = rx_effective_price(d); oth_p = other_price(d)
        rx_sales = rx_scripts*rx_p
        other_sales = other_units*oth_p
        sales_total = rx_sales + other_sales

        # COGS recognized this round (sales-based)
        cogs = rx_scripts*BASE["rx_ingredient_cost"] + other_units*BASE["other_unit_cost"]
        expiry_loss = cogs*ELASTIC["expiry_loss_pct_of_cogs"]

        # Merchant fee on card receipts
        merchant_fee = sales_total*ELASTIC["card_share"]*ELASTIC["card_fee_pct"]

        # ---- Cash & accrual flows ----
        # Sales receipts split: card (cash now minus fee), cash (now), credit (to AR)
        credit_share = ELASTIC["credit_sales_share"] if d.store_credit==1 else 0.0
        cash_now = sales_total*(1-credit_share)
        card_fee_deduction = merchant_fee
        cash_in_sales = cash_now - card_fee_deduction

        # AR collections from previous round
        ar_collections = states[name].ar * ELASTIC["ar_collection_rate"]
        states[name].ar = states[name].ar - ar_collections  # remaining

        # AR interest income charged to customers (if offer credit)
        ar_interest_income = 0.0
        if d.store_credit==1:
            ar_interest_income = states[name].ar * (d.ar_interest_rate_pct/100.0) / ELASTIC["periods_per_year"]
            states[name].ar += ar_interest_income  # capitalize to AR
        # New AR from this round's sales
        new_ar = sales_total*credit_share
        states[name].ar += new_ar

        # Purchases/Returns affect Inventory and AP
        net_purchases = d.rx_purchases_thb + d.other_purchases_thb - d.rx_returns_thb - d.other_returns_thb
        states[name].inventory_value += net_purchases - cogs - expiry_loss
        states[name].ap += max(net_purchases,0)  # assume all net purchases on credit

        # AP payment
        ap_payment = min(d.ap_payment_thb, states[name].ap)
        states[name].ap -= ap_payment

        # Investments / Withdrawals (Fixed assets)
        states[name].fixed_assets += d.invest_thb - d.withdraw_thb
        cash_flow_invest = -d.invest_thb + d.withdraw_thb

        # Long-term debt
        states[name].lt_debt = max(states[name].lt_debt - d.lt_debt_written_thb - d.lt_debt_payment_thb, 0.0)
        cash_flow_debt = -d.lt_debt_payment_thb

        # Mortgage payment
        cash_flow_mortgage = -d.mortgage_payment

        # Collections sent to agency (reduce AR now)
        sent_to_agency = min(d.sent_to_collections, states[name].ar)
        states[name].ar -= sent_to_agency
        # assume no immediate cash in

        # Promo cost cash out
        cash_flow_promo = -d.promo_budget_thb

        # Labor & manager costs (simple cash opex)
        weeks = 4.33
        labor_cash = d.n_pharmacists*d.pharm_pay_rate*d.hours_per_week*weeks + d.n_clerks*d.clerk_pay_rate*d.hours_per_week*weeks + d.manager_salary
        # Benefits (flat per employee per round; optional — set 0 for now)

        # Cash update
        cash_delta = cash_in_sales + ar_collections + ar_interest_income + cash_flow_invest + cash_flow_debt + cash_flow_mortgage + cash_flow_promo - ap_payment - labor_cash
        states[name].cash += cash_delta

        # Minimum cash balance policy (simple top-up from short-term facility; not tracked as debt here)
        if states[name].cash < d.min_cash_balance:
            topup = d.min_cash_balance - states[name].cash
            states[name].cash += topup  # assume overdraft/credit line
            overdraft = topup
        else:
            overdraft = 0.0

        gp = sales_total - cogs - expiry_loss
        net_profit = gp - (merchant_fee + d.promo_budget_thb + labor_cash)

        results[name] = {
            "location": loc,
            "rx_scripts": round(rx_scripts,2),
            "other_units": round(other_units,2),
            "rx_price": round(rx_p,2),
            "other_price": round(oth_p,2),
            "sales_total": round(sales_total,2),
            "cogs": round(cogs,2),
            "expiry_loss": round(expiry_loss,2),
            "merchant_fee": round(merchant_fee,2),
            "promo_cost": round(d.promo_budget_thb,2),
            "labor_cash": round(labor_cash,2),
            "gross_profit": round(gp,2),
            "net_profit": round(net_profit,2),
            # cash flows
            "cash_in_sales": round(cash_in_sales,2),
            "ar_new": round(new_ar,2),
            "ar_collections": round(ar_collections,2),
            "ar_interest_income": round(ar_interest_income,2),
            "ap_payment": round(ap_payment,2),
            "invest": round(d.invest_thb,2),
            "withdraw": round(d.withdraw_thb,2),
            "mortgage_payment": round(d.mortgage_payment,2),
            "lt_debt_payment": round(d.lt_debt_payment_thb,2),
            "sent_to_agency": round(sent_to_agency,2),
            "overdraft_topup": round(overdraft,2),
            # ending state
            "cash_end": round(states[name].cash,2),
            "ar_end": round(states[name].ar,2),
            "ap_end": round(states[name].ap,2),
            "inventory_end": round(states[name].inventory_value,2),
            "lt_debt_end": round(states[name].lt_debt,2),
            "fixed_assets_end": round(states[name].fixed_assets,2),
            "u_rx": round(utility_rx(d),4),
            "u_other": round(utility_other(d),4),
        }
    return results, states

# ==== UI ====
st.title("🧪 Communi‑Pharm Prototype — Multi‑Round (7 Stores, 3 Locations)")

# Sidebar: global params
st.sidebar.header("Global Parameters")
BASE["rx_ingredient_cost"] = st.sidebar.number_input("Rx Ingredient Cost", 50.0, 1000.0, BASE["rx_ingredient_cost"], 10.0)
BASE["other_unit_cost"] = st.sidebar.number_input("Other Unit Cost", 20.0, 1000.0, BASE["other_unit_cost"], 5.0)
ELASTIC["card_share"] = st.sidebar.slider("Card Share", 0.0, 1.0, ELASTIC["card_share"], 0.01)
ELASTIC["card_fee_pct"] = st.sidebar.number_input("Card Fee %", 0.0, 0.1, ELASTIC["card_fee_pct"], 0.001, format="%.3f")
ELASTIC["credit_sales_share"] = st.sidebar.slider("Credit Sales Share (if credit enabled)", 0.0, 1.0, ELASTIC["credit_sales_share"], 0.05)
ELASTIC["ar_collection_rate"] = st.sidebar.slider("AR Collection Rate per round", 0.0, 1.0, ELASTIC["ar_collection_rate"], 0.05)

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
    step = 0.001 if key != "beta_promo" else 0.0001
    ELASTIC[key] = st.sidebar.number_input(key, mn, mx, ELASTIC[key], step=step, format="%.3f")

# Session state for store states
if "round" not in st.session_state:
    st.session_state.round = 1
if "states" not in st.session_state:
    st.session_state.states = {f"Store_{i+1}": StoreState() for i in range(7)}

st.write(f"### Round {st.session_state.round} — Enter Decisions")

# Decision inputs for 7 stores
stores: Dict[str, Decisions] = {}
cols = st.columns(7)
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
    "other_returns_thb","ap_payment_thb","lt_debt_written_thb","lt_debt_payment_thb","ar_interest_rate_pct","life_insurance","health_insurance","third_party","hmo"
]

for i in range(7):
    with cols[i]:
        st.markdown(f"**Store_{i+1}**")
        vals = list(defaults[i])
        # Build widgets
        d = {}
        d["location"] = st.selectbox("Location", LOCATIONS, index=LOCATIONS.index(vals[0]), key=f"loc{i}")
        d["rx_markup_pct"] = st.number_input("Rx Markup %", 0.0, 300.0, float(vals[1]), 1.0, key=f"rxm{i}")
        d["rx_fee_thb"] = st.number_input("Rx Fee", 0.0, 500.0, float(vals[2]), 1.0, key=f"rxf{i}")
        d["rx_copay_discount_thb"] = st.number_input("Rx Copay Discount", 0.0, 200.0, float(vals[3]), 1.0, key=f"cop{i}")
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
    import pandas as pd
    df = pd.DataFrame(res).T.reset_index().rename(columns={"index":"Store"})
    st.success(f"Round {st.session_state.round} completed.")
    st.subheader("Per-Store Results")
    st.dataframe(df, use_container_width=True)
    # State table
    st.subheader("Ending State (Balance Sheet-ish)")
    st.dataframe(pd.DataFrame({k:asdict(v) for k,v in st.session_state.states.items()}).T, use_container_width=True)
    # Download
    st.download_button("Download results CSV", df.to_csv(index=False).encode("utf-8-sig"), "round_results.csv", "text/csv")
    st.download_button("Download state CSV", pd.DataFrame({k:asdict(v) for k,v in st.session_state.states.items()}).T.to_csv().encode("utf-8-sig"), "ending_state.csv", "text/csv")

if colB.button("Next Round ➡️"):
    st.session_state.round += 1
    st.success(f"Move to Round {st.session_state.round}. Keep previous ending states.")

if colC.button("Reset All ⛔"):
    st.session_state.round = 1
    st.session_state.states = {f"Store_{i+1}": StoreState() for i in range(7)}
    st.warning("All states reset.")
st.caption("หมายเหตุ: แบบจำลองนี้เป็นเวอร์ชันต้นแบบ ตั้งใจเลียนแบบตรรกะสำคัญของ Communi‑Pharm: แบ่งดีมานด์ตาม utility, มีงบลงทุน/สต๊อก/ลูกหนี้‑เจ้าหนี้ และเงินสดไหลข้ามรอบ")
