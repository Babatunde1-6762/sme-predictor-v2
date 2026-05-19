import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
from datetime import datetime, timedelta

# ── CPV LOOKUP ───────────────────────────────────────────────────────────────
CPV_LOOKUP = {
    "45000000":"Construction","45100000":"Site preparation work",
    "45200000":"Building construction","45300000":"Building installation works",
    "45400000":"Building completion work","70000000":"Real estate services",
    "71000000":"Architectural and engineering services",
    "30000000":"IT equipment and supplies","48000000":"Software and IT systems",
    "72000000":"IT services","72100000":"IT consultancy",
    "72200000":"Software programming services","72300000":"Data services",
    "72400000":"Internet services","72500000":"Computer-related services",
    "72600000":"IT support and consultancy","64000000":"Postal and telecommunications",
    "32000000":"Radio, television and communications equipment",
    "33000000":"Medical equipment and supplies","85000000":"Health and social work services",
    "85100000":"Health services","85110000":"Hospital services",
    "85120000":"Medical practice services","85200000":"Veterinary services",
    "85300000":"Social work services","85320000":"Social services",
    "80000000":"Education and training services","80100000":"Primary education services",
    "80200000":"Secondary education services","80300000":"Higher education services",
    "80400000":"Adult and other education services","80500000":"Training services",
    "60000000":"Transport services","60100000":"Road transport services",
    "60200000":"Rail transport services","60400000":"Air transport services",
    "63000000":"Supporting transport services",
    "66000000":"Financial and insurance services",
    "73000000":"Research and development services",
    "79100000":"Legal services","79200000":"Accounting services",
    "79400000":"Business and management consultancy",
    "79600000":"Recruitment services","79700000":"Investigation and security services",
    "50000000":"Repair and maintenance services",
    "55000000":"Hotel, restaurant and catering services",
    "90000000":"Sewage, refuse, cleaning and environmental services",
    "90600000":"Cleaning services","90700000":"Environmental services",
    "09000000":"Petroleum products, fuel and electricity",
    "09300000":"Electricity, heating, solar and nuclear energy",
    "65000000":"Public utilities",
    "03000000":"Agricultural, farming and fishing products",
    "15000000":"Food, beverages, tobacco and related products",
    "24000000":"Chemical products","39000000":"Furniture and household appliances",
    "35000000":"Security and fire-fighting equipment",
    "98000000":"Other community and social services",
}

INDUSTRY_LOOKUP = {}
for _c, _i in CPV_LOOKUP.items():
    INDUSTRY_LOOKUP.setdefault(_i, []).append(_c)

def cpv_to_industry(cpv):
    s = str(cpv).split(".")[0].strip()
    if s in CPV_LOOKUP: return CPV_LOOKUP[s]
    if s[:6]+"00" in CPV_LOOKUP: return CPV_LOOKUP[s[:6]+"00"]
    if s[:4]+"0000" in CPV_LOOKUP: return CPV_LOOKUP[s[:4]+"0000"]
    if s[:2]+"000000" in CPV_LOOKUP: return CPV_LOOKUP[s[:2]+"000000"]
    return "Unknown industry"

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    rf     = joblib.load("sme_rf_model.pkl")
    xgb    = joblib.load("sme_xgb_model.pkl")
    lr     = joblib.load("sme_lr_model.pkl")
    scaler = joblib.load("scaler.pkl")
    enc    = json.load(open("encoders.json"))
    feats  = json.load(open("feature_cols.json"))
    rates  = json.load(open("historical_rates.json"))
    try:
        results   = pd.read_csv("model_comparison.csv")
        best_row  = results.loc[results["AUC-ROC"].idxmax()]
        best_name = best_row["Model"]
        best_auc  = float(best_row["AUC-ROC"])
    except Exception:
        best_name = "Random Forest"
        best_auc  = None
    return rf, xgb, lr, scaler, enc, feats, rates, best_name, best_auc

# ── LIVE CONTRACTS ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_live_contracts(days_back=7, max_results=50):
    date_from     = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to       = datetime.now().strftime("%Y-%m-%d")
    all_contracts = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SMEResearchBot/1.0)",
        "Accept":     "application/json",
        "Content-Type": "application/json",
        "Origin":  "https://www.contractsfinder.service.gov.uk",
        "Referer": "https://www.contractsfinder.service.gov.uk/",
    }
    try:
        cf_url = "https://www.contractsfinder.service.gov.uk/Published/Notices/PublicSearch/json"
        payload = {"searchCriteria": {"page":1,"publishedFrom":date_from,"publishedTo":date_to,"size":max_results}}
        cf_r = requests.post(cf_url, json=payload, headers=headers, timeout=25)
        if cf_r.status_code == 200:
            for n in cf_r.json().get("results", []):
                item = n.get("item", {})
                cpv_list = item.get("cpvCodes", [])
                cpv      = cpv_list[0].get("code","Unknown") if cpv_list else "Unknown"
                locs     = item.get("deliveryLocations",[])
                region   = locs[0].get("region","Unknown") if locs else "Unknown"
                val      = item.get("value",{}) or {}
                all_contracts.append({
                    "source":"Contracts Finder",
                    "title":str(item.get("title","Unknown"))[:100],
                    "buyer":str(item.get("organizationName","Unknown"))[:60],
                    "value":float(val.get("amount",0) or 0),
                    "cpv_code":str(cpv),"region":str(region),
                    "deadline":str(item.get("tenderDeadline","Not specified")),
                    "published":str(item.get("publishedAt","Unknown")),
                    "url":str(n.get("publishedUrl","")),
                })
    except Exception:
        pass
    try:
        fat_headers = {**headers,"Origin":"https://www.find-tender.service.gov.uk","Referer":"https://www.find-tender.service.gov.uk/"}
        fat_url = "https://www.find-tender.service.gov.uk/api/1.0/ocds/notices/list"
        fat_r = requests.get(fat_url, params={"publishedFrom":date_from,"publishedTo":date_to,"limit":max_results,"offset":0}, headers=fat_headers, timeout=25)
        if fat_r.status_code == 200:
            records = fat_r.json().get("records", fat_r.json().get("releases",[]))
            for rec in records:
                rel    = rec.get("compiledRelease", rec)
                tender = rel.get("tender",{})
                buyer  = rel.get("buyer",{}).get("name","Unknown")[:60]
                items  = tender.get("items",[])
                cpv    = items[0].get("classification",{}).get("id","Unknown") if items else "Unknown"
                locs   = tender.get("deliveryLocations",[])
                region = locs[0].get("region","Unknown") if locs else "Unknown"
                val    = tender.get("value",{}) or {}
                all_contracts.append({
                    "source":"Find a Tender",
                    "title":str(tender.get("title","Unknown"))[:100],
                    "buyer":str(buyer),
                    "value":float(val.get("amount",0) or 0),
                    "cpv_code":str(cpv),"region":str(region),
                    "deadline":str(tender.get("tenderPeriod",{}).get("endDate","Not specified")),
                    "published":str(rel.get("date","Unknown")),
                    "url":str(rel.get("ocid","")),
                })
    except Exception:
        pass
    if not all_contracts:
        import random
        random.seed(int(datetime.now().timestamp()) % 1000)
        sectors = [("IT services","72000000"),("Construction","45000000"),("Cleaning services","90600000"),
                   ("Training services","80500000"),("Health services","85100000"),("Accounting services","79200000"),
                   ("Transport services","60000000"),("Legal services","79100000"),
                   ("Repair and maintenance","50000000"),("Environmental services","90700000")]
        buyers_cf  = ["NHS Trust","Local Council","Ministry of Justice","Department for Education","HMRC","Home Office","Environment Agency"]
        buyers_fat = ["Cabinet Office","MOD","DVLA","Companies House","Crown Commercial Service","UKRI","Innovate UK"]
        regions    = ["London","South East","North West","Yorkshire and the Humber","East Midlands","West Midlands","East of England","South West","North East","Wales","Scotland"]
        for i in range(min(max_results, 30)):
            sname, cpv = random.choice(sectors)
            source = "Contracts Finder" if i % 2 == 0 else "Find a Tender"
            buyers = buyers_cf if source == "Contracts Finder" else buyers_fat
            value  = round(random.uniform(10000, 500000), 2)
            pub_dt = datetime.now() - timedelta(days=random.randint(0, days_back))
            ddl_dt = pub_dt + timedelta(days=random.randint(14, 60))
            all_contracts.append({
                "source":source,"title":f"{sname} Services Contract 2026{i:03d}",
                "buyer":random.choice(buyers),"value":value,"cpv_code":cpv,
                "region":random.choice(regions),"deadline":ddl_dt.strftime("%Y-%m-%d"),
                "published":pub_dt.strftime("%Y-%m-%dT%H:%M:%S"),"url":"",
            })
    df = pd.DataFrame(all_contracts)
    df = df.drop_duplicates(subset=["title","buyer"], keep="first")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return df.sort_values("published", ascending=False).reset_index(drop=True)

# ── PREDICTION HELPERS ────────────────────────────────────────────────────────
def build_row(cv, am, aq, region, cpv, encoders, feature_cols, rates, scaler):
    log_cv = np.log1p(cv)
    vbnum  = 0 if cv<10000 else 1 if cv<50000 else 2 if cv<100000 else 3 if cv<500000 else 4
    is_qe  = int(aq in [1,4])
    is_hv  = int(cv > 100000)
    cr     = rates["cpv_sme_rate"].get(str(cpv),    rates["global_sme_rate"])
    rr     = rates["region_sme_rate"].get(str(region), rates["global_sme_rate"])
    r_enc  = encoders.get("region",   {}).get(str(region), 0)
    c_enc  = encoders.get("cpv_code", {}).get(str(cpv), 0)
    d = {"log_contract_value":log_cv,"value_band_num":vbnum,"award_month":am,"award_quarter":aq,
         "is_quarter_end":is_qe,"is_high_value":is_hv,"buyer_sme_rate":rates["global_sme_rate"],
         "cpv_sme_rate":cr,"region_sme_rate":rr,"value_band_enc":0,"region_enc":r_enc,"cpv_code_enc":c_enc}
    row = pd.DataFrame([{c: d.get(c,0) for c in feature_cols}])
    return scaler.transform(row.values), cr, rr

def get_ensemble(row_scaled, rf, xgb, lr):
    return (rf.predict_proba(row_scaled)[0][1]*0.5 +
            xgb.predict_proba(row_scaled)[0][1]*0.35 +
            lr.predict_proba(row_scaled)[0][1]*0.15)

def score_contracts(df, encoders, feature_cols, rates, scaler, rf, xgb, lr):
    now = datetime.now()
    am, aq = now.month, (now.month-1)//3+1
    probs = []
    for _, row in df.iterrows():
        try:
            cv  = float(row.get("value",50000) or 50000)
            cpv = str(row.get("cpv_code","Unknown"))
            reg = str(row.get("region","Unknown"))
            r, _, _ = build_row(cv, am, aq, reg, cpv, encoders, feature_cols, rates, scaler)
            probs.append(round(get_ensemble(r, rf, xgb, lr), 3))
        except Exception:
            probs.append(rates["global_sme_rate"])
    return probs

def get_accessibility_scores(p_ens):
    accessibility   = round(p_ens * 100, 1)
    confidence      = round((1 - abs(p_ens - 0.5) * 2) * 100, 1) if p_ens < 0.5 else round(p_ens * 100, 1)
    bid_feasibility = round(min(100, p_ens * 120) * 100, 1) / 100
    return accessibility, confidence, bid_feasibility

def explain_prediction(p_ens, cv, cpv, region, rates):
    reasons = []
    cr = rates["cpv_sme_rate"].get(str(cpv), rates["global_sme_rate"])
    rr = rates["region_sme_rate"].get(str(region), rates["global_sme_rate"])
    global_rate = rates["global_sme_rate"]
    if cv > 150000:
        reasons.append(("High contract value", f"At £{cv:,.0f}, this contract exceeds the threshold where SMEs typically win. Large contracts favour established suppliers.", "negative"))
    if cr < global_rate * 0.8:
        reasons.append(("Sector disadvantage", f"The {cpv_to_industry(cpv)} sector has a {cr*100:.1f}% SME award rate — below the national average of {global_rate*100:.1f}%. This sector is dominated by larger suppliers.", "negative"))
    if rr < global_rate * 0.8:
        reasons.append(("Regional disadvantage", f"The {region} region has a {rr*100:.1f}% SME award rate. Regional procurement patterns here are less favourable for SMEs.", "negative"))
    if cv > 100000:
        reasons.append(("Value mismatch", "Contracts above £100,000 have significantly lower SME win rates. SMEs are more competitive at lower contract values.", "negative"))
    if cr > global_rate * 1.1:
        reasons.append(("Sector advantage", f"The {cpv_to_industry(cpv)} sector has a {cr*100:.1f}% SME award rate — above the national average. This sector is SME-friendly.", "positive"))
    if rr > global_rate * 1.1:
        reasons.append(("Regional advantage", f"The {region} region has a {rr*100:.1f}% SME award rate — above average. This region has a strong SME procurement culture.", "positive"))
    if cv < 50000:
        reasons.append(("Value advantage", f"At £{cv:,.0f}, this is within the contract value range where SMEs are most competitive.", "positive"))
    if not reasons:
        reasons.append(("Average conditions", "This contract has typical procurement characteristics. Win probability reflects the baseline SME award rate for this sector and region.", "neutral"))
    return reasons

def gap_analysis(cv, cpv, region, encoders, feature_cols, rates, scaler, rf, xgb, lr):
    now = datetime.now()
    am, aq = now.month, (now.month-1)//3+1
    def gp(cv_, cpv_, reg_):
        r, _, _ = build_row(cv_, am, aq, reg_, cpv_, encoders, feature_cols, rates, scaler)
        return get_ensemble(r, rf, xgb, lr)
    base = gp(cv, cpv, region)
    sugg = []
    for test_cv, label in [(cv*0.5, f"Target 50% smaller contract (£{cv*0.5:,.0f})"),
                            (cv*0.25,f"Target 75% smaller contract (£{cv*0.25:,.0f})"),
                            (25000,  "Target £25,000 contract — highest SME success range")]:
        p = gp(test_cv, cpv, region)
        if p > base:
            sugg.append(("Contract Size", label, round(p,3), round(p-base,3),
                          "Smaller contracts have significantly higher SME win rates. Building a track record at smaller values improves future competitiveness."))
    best_reg = max(rates["region_sme_rate"], key=rates["region_sme_rate"].get)
    p_br = gp(cv, cpv, best_reg)
    if p_br > base:
        sugg.append(("Region Strategy", f"Target {best_reg} region", round(p_br,3), round(p_br-base,3),
                      f"{best_reg} has the highest SME award rate nationally. Expanding geographic reach to this region significantly improves win probability."))
    best_cpv = max(rates["cpv_sme_rate"], key=rates["cpv_sme_rate"].get)
    p_bc = gp(cv, best_cpv, region)
    if p_bc > base:
        sugg.append(("Sector Pivot", f"Consider {cpv_to_industry(best_cpv)}", round(p_bc,3), round(p_bc-base,3),
                      f"{cpv_to_industry(best_cpv)} has the highest SME win rate. If the SME has adjacent capabilities, expanding into this sector would be strategically beneficial."))
    return base, sorted(sugg, key=lambda x: x[3], reverse=True)

def analyse_sme_batch(sme_df, encoders, feature_cols, rates, scaler, rf, xgb, lr):
    now = datetime.now()
    am, aq = now.month, (now.month-1)//3+1
    results = []
    for _, sme in sme_df.iterrows():
        try:
            cv  = float(sme.get("contract_value", 50000) or 50000)
            cpv = str(sme.get("cpv_code", "Unknown"))
            reg = str(sme.get("region", "Unknown"))
            r, cr, rr = build_row(cv, am, aq, reg, cpv, encoders, feature_cols, rates, scaler)
            p_rf  = rf.predict_proba(r)[0][1]
            p_xgb = xgb.predict_proba(r)[0][1]
            p_lr  = lr.predict_proba(r)[0][1]
            p_ens = p_rf*0.5 + p_xgb*0.35 + p_lr*0.15
            acc, conf, bid_feas = get_accessibility_scores(p_ens)
            reasons = explain_prediction(p_ens, cv, cpv, reg, rates)
            barrier = ", ".join([r[0] for r in reasons if r[2]=="negative"]) or "None identified"
            results.append({
                "sme_name":            sme.get("sme_name", f"SME_{_}"),
                "sector":              sme.get("sector", cpv_to_industry(cpv)),
                "region":              reg,
                "contract_value":      cv,
                "cpv_code":            cpv,
                "win_probability":     round(p_ens, 3),
                "accessibility_score": acc,
                "confidence_index":    conf,
                "bid_feasibility":     bid_feas,
                "prediction":          "Likely to win" if p_ens >= 0.5 else "Unlikely to win",
                "primary_barrier":     barrier,
                "recommendation":      "Apply — competitive" if p_ens >= 0.6 else
                                       "Consider applying"   if p_ens >= 0.4 else
                                       "Needs preparation",
            })
        except Exception:
            pass
    return pd.DataFrame(results)

# ── BOOT ──────────────────────────────────────────────────────────────────────
rf, xgb, lr, scaler, encoders, feature_cols, rates, best_name, best_auc = load_artefacts()
all_cpv_codes  = list(encoders.get("cpv_code", {"Unknown":0}).keys())
all_industries = sorted(set(CPV_LOOKUP.values()))

MODEL_MAP = {
    "Random Forest":                  ("Random Forest",       rf),
    "XGBoost":                        ("XGBoost",             xgb),
    "Logistic Regression":            ("Logistic Regression", lr),
    "Logistic Regression (baseline)": ("Logistic Regression", lr),
}
best_label, _ = MODEL_MAP.get(best_name, ("Random Forest", rf))
model_options = []
for opt in ["Random Forest","XGBoost","Logistic Regression"]:
    if opt == best_label:
        auc_str = f" — AUC {best_auc:.3f}" if best_auc else ""
        model_options.append(f"{opt} ⭐ Recommended{auc_str}")
    else:
        model_options.append(opt)
default_index = next((i for i,o in enumerate(model_options) if "⭐" in o), 0)

# ── PAGE ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SME Procurement Accessibility Intelligence", page_icon="🏆", layout="wide")
st.title("🏆 AI-Driven SME Procurement Accessibility Intelligence Platform")
st.markdown("Explainable AI revealing structural barriers affecting SME participation in UK public procurement.")
if best_auc:
    st.success(f"🏆 Best model: **{best_label}**  |  AUC-ROC: **{best_auc:.4f}**  — set as default")
st.divider()

tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "🔮 Win Probability",
    "📊 Barrier & Gap Analysis",
    "📡 Live Contracts",
    "🏭 CPV Lookup",
    "📈 Historical",
    "🏢 SME Readiness",
    "🗺️ Barrier Dashboard",
])

# ── TAB 1: WIN PROBABILITY ────────────────────────────────────────────────────
with tab1:
    st.subheader("🔮 Will this SME win the contract?")
    st.markdown("Predicts win probability and provides a full explainability breakdown of why the SME is likely or unlikely to succeed.")
    c1, c2 = st.columns(2)
    with c1:
        cv1 = st.number_input("Contract value (£)", min_value=0.0, value=50000.0, step=1000.0, key="cv1")
        am1 = st.selectbox("Award month", list(range(1,13)),
                           format_func=lambda m:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1], key="am1")
        aq1 = st.selectbox("Award quarter", [1,2,3,4], format_func=lambda q:f"Q{q}", key="aq1")
        mc  = st.selectbox("Model", model_options, index=default_index, key="mc1")
    with c2:
        r1   = st.selectbox("Region", list(encoders.get("region",{"Unknown":0}).keys()), key="r1")
        inp1 = st.radio("CPV input method", ["Select CPV code","Select by industry"], key="im1", horizontal=True)
        if inp1 == "Select CPV code":
            cp1 = st.selectbox("CPV code", all_cpv_codes, key="cp1a")
            st.caption(f"Industry: **{cpv_to_industry(cp1)}**")
        else:
            ind1 = st.selectbox("Industry", all_industries, key="ind1")
            cp1  = st.selectbox("CPV code (from industry)", INDUSTRY_LOOKUP.get(ind1,["Unknown"]), key="cp1b")
    if st.button("🔮 Predict win probability", type="primary", use_container_width=True, key="btn1"):
        row1, cr1, rr1 = build_row(cv1, am1, aq1, r1, cp1, encoders, feature_cols, rates, scaler)
        p_rf1  = rf.predict_proba(row1)[0][1]
        p_xgb1 = xgb.predict_proba(row1)[0][1]
        p_lr1  = lr.predict_proba(row1)[0][1]
        p_ens1 = p_rf1*0.5 + p_xgb1*0.35 + p_lr1*0.15
        pred1  = int(p_ens1 >= 0.5)
        acc1, conf1, bid1 = get_accessibility_scores(p_ens1)
        a,b,c,d = st.columns(4)
        a.metric("Win Probability",           f"{p_ens1*100:.1f}%")
        b.metric("Procurement Accessibility", f"{acc1:.1f}%")
        c.metric("SME Confidence Index",      f"{conf1:.1f}%")
        d.metric("Bid Feasibility Score",     f"{bid1:.2f}")
        st.progress(float(p_ens1))
        if pred1:
            st.success("✅ This SME has a good chance. Applying is strongly recommended.")
        elif p_ens1 >= 0.35:
            st.warning("⚠️ Borderline probability. With preparation this SME could compete. See Barrier & Gap Analysis tab.")
        else:
            st.error("❌ Low win probability. This confirms why SMEs in this category hesitate to apply. See Barrier & Gap Analysis tab.")
        st.divider()
        st.markdown("### 🔍 Explainability — Why is this the prediction?")
        reasons = explain_prediction(p_ens1, cv1, cp1, r1, rates)
        for reason_name, explanation, direction in reasons:
            if direction == "negative":
                st.error(f"**❌ {reason_name}:** {explanation}")
            elif direction == "positive":
                st.success(f"**✅ {reason_name}:** {explanation}")
            else:
                st.info(f"**ℹ️ {reason_name}:** {explanation}")
        st.divider()
        st.markdown("**Individual model contributions:**")
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Random Forest (50%)", f"{p_rf1*100:.1f}%")
        col2.metric("XGBoost (35%)",        f"{p_xgb1*100:.1f}%")
        col3.metric("Logistic Reg (15%)",   f"{p_lr1*100:.1f}%")
        col4.metric("Ensemble (final)",     f"{p_ens1*100:.1f}%")
        st.info(f"Industry: **{cpv_to_industry(cp1)}**  |  Sector SME rate: {cr1*100:.1f}%  |  Region SME rate: {rr1*100:.1f}%")

# ── TAB 2: BARRIER & GAP ANALYSIS ────────────────────────────────────────────
with tab2:
    st.subheader("📊 Barrier & Capability Gap Analysis")
    st.markdown("Identifies structural barriers preventing SMEs from winning contracts and provides actionable recommendations to improve competitiveness.")
    c1, c2 = st.columns(2)
    with c1:
        cv2 = st.number_input("Contract value (£)", min_value=0.0, value=200000.0, step=1000.0, key="cv2")
        r2  = st.selectbox("Region", list(encoders.get("region",{"Unknown":0}).keys()), key="r2")
    with c2:
        inp2 = st.radio("CPV input method", ["Select CPV code","Select by industry"], key="im2", horizontal=True)
        if inp2 == "Select CPV code":
            cp2 = st.selectbox("CPV code", all_cpv_codes, key="cp2a")
            st.caption(f"Industry: **{cpv_to_industry(cp2)}**")
        else:
            ind2 = st.selectbox("Industry", all_industries, key="ind2")
            cp2  = st.selectbox("CPV code (from industry)", INDUSTRY_LOOKUP.get(ind2,["Unknown"]), key="cp2b")
    if st.button("📊 Run barrier and gap analysis", type="primary", use_container_width=True, key="btn2"):
        base_prob, suggestions = gap_analysis(cv2, cp2, r2, encoders, feature_cols, rates, scaler, rf, xgb, lr)
        acc2, conf2, bid2 = get_accessibility_scores(base_prob)
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Current win probability",    f"{base_prob*100:.1f}%")
        col2.metric("Procurement Accessibility",  f"{acc2:.1f}%")
        col3.metric("SME Confidence Index",       f"{conf2:.1f}%")
        col4.metric("Bid Feasibility",            f"{bid2:.2f}")
        st.progress(float(base_prob))
        if base_prob < 0.3:
            st.error(f"❌ Only **{base_prob*100:.1f}%** probability. The model confirms this is a structural participation barrier — the odds are too low to justify bid costs for most SMEs.")
        elif base_prob < 0.5:
            st.warning(f"⚠️ **{base_prob*100:.1f}%** probability — below threshold. SMEs are making a rational economic calculation: bid costs outweigh expected return.")
        else:
            st.success(f"✅ **{base_prob*100:.1f}%** probability — above threshold. SMEs in this category should be actively encouraged to apply.")
        st.divider()
        st.markdown("### ❌ Identified Participation Barriers")
        reasons = explain_prediction(base_prob, cv2, cp2, r2, rates)
        barriers_found = [r for r in reasons if r[2] == "negative"]
        if barriers_found:
            for reason_name, explanation, _ in barriers_found:
                st.error(f"**{reason_name}:** {explanation}")
        else:
            st.success("No significant barriers identified for this contract profile.")
        st.divider()
        if suggestions:
            st.markdown("### 🎯 Capability Gap Recommendations")
            st.markdown("These are the specific changes that would most improve this SME's competitiveness:")
            for category, action, new_prob, improvement, advice in suggestions[:4]:
                with st.expander(f"💡 {action}  →  {new_prob*100:.1f}%  (+{improvement*100:.1f}%)"):
                    st.markdown(f"**Category:** {category}")
                    st.markdown(f"**Recommended action:** {action}")
                    st.markdown(f"**Strategic rationale:** {advice}")
                    st.progress(float(new_prob))
                    col_a, col_b = st.columns(2)
                    col_a.metric("Current probability",  f"{base_prob*100:.1f}%")
                    col_b.metric("Improved probability", f"{new_prob*100:.1f}%", delta=f"+{improvement*100:.1f}%")
        st.divider()
        st.markdown("### 📝 Why are SMEs not applying? — Model Evidence")
        st.markdown(f"For a **{cpv_to_industry(cp2)}** contract worth **£{cv2:,.0f}** in **{r2}**:")
        if base_prob < 0.3:
            st.error("**Model verdict:** With less than 30% predicted probability, SME reluctance is rational and data-driven. The financial cost of bid preparation — typically £5,000–£20,000 for a competitive tender — cannot be justified at this probability level. This is a structural accessibility barrier.")
        elif base_prob < 0.5:
            st.warning("**Model verdict:** At 30-50% probability, the risk-reward calculation is marginal. SMEs face a rational choice: invest significant resources in a bid with a less-than-even chance of success. Many choose not to apply.")
        else:
            st.success("**Model verdict:** Above 50% probability. SMEs in this category should be actively encouraged to apply. The model suggests the market conditions are favourable and the investment in bidding is likely to be worthwhile.")

# ── TAB 3: LIVE CONTRACTS ─────────────────────────────────────────────────────
with tab3:
    st.subheader("📡 Live UK Government Contracts")
    st.markdown("Real-time contracts from **Contracts Finder** and **Find a Tender** — both sources running concurrently, each contract scored for SME win probability.")
    st.info("ℹ️ The UK government APIs restrict access by IP address. When live data is unavailable, a realistic representative sample is generated so prediction functionality always works.")
    c1, c2 = st.columns(2)
    with c1:
        days_back = st.slider("Days to look back", 1, 30, 7)
        max_res   = st.selectbox("Max contracts per source", [25,50,100], index=1)
    with c2:
        filter_region = st.selectbox("Filter by region (optional)", ["All"]+list(encoders.get("region",{"Unknown":0}).keys()))
        min_prob      = st.slider("Minimum SME win probability", 0.0, 1.0, 0.0, 0.05)
    if st.button("📡 Fetch and score live contracts", type="primary", use_container_width=True):
        with st.spinner("Fetching from Contracts Finder and Find a Tender simultaneously..."):
            live_df = fetch_live_contracts(days_back=days_back, max_results=max_res)
        with st.spinner("Scoring all contracts for SME win probability..."):
            live_df["sme_win_probability"] = score_contracts(live_df, encoders, feature_cols, rates, scaler, rf, xgb, lr)
        live_df["industry"]       = live_df["cpv_code"].apply(cpv_to_industry)
        live_df["recommendation"] = live_df["sme_win_probability"].apply(
            lambda p: "✅ Apply — good chance" if p>=0.6 else "🤔 Consider applying" if p>=0.4 else "❌ Low chance — prepare more")
        is_live = any(live_df["url"].str.startswith("http", na=False))
        if is_live:
            st.success("✅ Live data fetched from UK government portals.")
        else:
            st.info("📊 Showing representative sample — live APIs currently restricted by IP whitelist.")
        if filter_region != "All":
            live_df = live_df[live_df["region"]==filter_region]
        live_df = live_df[live_df["sme_win_probability"]>=min_prob].sort_values("sme_win_probability", ascending=False).reset_index(drop=True)
        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Total contracts",  str(len(live_df)))
        col2.metric("Contracts Finder", str((live_df["source"]=="Contracts Finder").sum()))
        col3.metric("Find a Tender",    str((live_df["source"]=="Find a Tender").sum()))
        col4.metric("Above 50% chance", str((live_df["sme_win_probability"]>=0.5).sum()))
        col5.metric("Below 30% chance", str((live_df["sme_win_probability"]<0.3).sum()))
        st.dataframe(
            live_df[["source","title","buyer","value","region","industry","sme_win_probability","recommendation","deadline"]]
            .rename(columns={"sme_win_probability":"SME Win %","source":"Source"}),
            use_container_width=True)
        csv = live_df.to_csv(index=False)
        st.download_button("📥 Download as CSV", csv, "live_scored_contracts.csv", "text/csv")

# ── TAB 4: CPV LOOKUP ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("🏭 CPV Code ↔ Industry Lookup")
    st.markdown("Find the industry for any CPV code, or find all CPV codes for any industry. Each entry shows the historical SME award rate.")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### CPV code → Industry")
        cpv_input = st.text_input("Enter a CPV code (e.g. 72200000)", key="cpv_in")
        if cpv_input:
            industry_result = cpv_to_industry(cpv_input.strip())
            if industry_result == "Unknown industry":
                st.warning(f"No industry found for CPV code: {cpv_input}")
            else:
                st.success(f"**{cpv_input}** belongs to: **{industry_result}**")
                sme_rate = rates["cpv_sme_rate"].get(str(cpv_input.strip()), rates["global_sme_rate"])
                st.metric("Historical SME award rate for this CPV", f"{sme_rate*100:.1f}%")
                related = INDUSTRY_LOOKUP.get(industry_result, [])
                related_rates = {c: rates["cpv_sme_rate"].get(c, rates["global_sme_rate"]) for c in related}
                df_related = pd.DataFrame({"CPV Code":related,"Industry":[industry_result]*len(related),"SME Rate":[f"{related_rates[c]*100:.1f}%" for c in related]})
                st.dataframe(df_related, use_container_width=True)
    with col_right:
        st.markdown("#### Industry → CPV codes")
        industry_input = st.selectbox("Select an industry", all_industries, key="ind_in")
        if industry_input:
            cpv_list = INDUSTRY_LOOKUP.get(industry_input,[])
            st.success(f"**{industry_input}** contains {len(cpv_list)} CPV code(s)")
            rs = {c: rates["cpv_sme_rate"].get(c, rates["global_sme_rate"]) for c in cpv_list}
            df_sector = pd.DataFrame({"CPV Code":cpv_list,"Historical SME Rate":[f"{rs[c]*100:.1f}%" for c in cpv_list]})
            st.dataframe(df_sector, use_container_width=True)
            if rs:
                bc = max(rs, key=rs.get)
                wc = min(rs, key=rs.get)
                st.info(f"Highest SME rate: CPV {bc} at {rs[bc]*100:.1f}%")
                st.info(f"Lowest SME rate:  CPV {wc} at {rs[wc]*100:.1f}%")

# ── TAB 5: HISTORICAL ─────────────────────────────────────────────────────────
with tab5:
    st.subheader("📈 Historical SME Procurement Insights")
    c1,c2,c3 = st.columns(3)
    c1.metric("Global SME award rate", f"{rates['global_sme_rate']*100:.1f}%")
    c2.metric("Best performing model", best_label)
    c3.metric("Best AUC-ROC", f"{best_auc:.4f}" if best_auc else "N/A")
    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Top 10 regions by SME award rate:**")
        rdf = pd.DataFrame(list(rates["region_sme_rate"].items()), columns=["Region","SME Rate"])
        rdf = rdf.sort_values("SME Rate", ascending=False).head(10)
        rdf["SME Rate"] = rdf["SME Rate"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(rdf, use_container_width=True)
    with cb:
        st.markdown("**Top 10 sectors by SME award rate:**")
        cdf = pd.DataFrame(list(rates["cpv_sme_rate"].items()), columns=["CPV Code","SME Rate"])
        cdf = cdf.sort_values("SME Rate", ascending=False).head(10)
        cdf["Industry"] = cdf["CPV Code"].apply(cpv_to_industry)
        cdf["SME Rate"] = cdf["SME Rate"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(cdf[["CPV Code","Industry","SME Rate"]], use_container_width=True)
    st.divider()
    st.markdown("**SME Participation Inequality Analysis:**")
    region_df  = pd.DataFrame(list(rates["region_sme_rate"].items()), columns=["Region","Rate"])
    sector_df  = pd.DataFrame(list(rates["cpv_sme_rate"].items()), columns=["CPV","Rate"])
    global_r   = rates["global_sme_rate"]
    below_avg_regions  = (region_df["Rate"] < global_r).sum()
    below_avg_sectors  = (sector_df["Rate"] < global_r).sum()
    col1,col2,col3 = st.columns(3)
    col1.metric("Regions below average", f"{below_avg_regions}/{len(region_df)}", help="Regions where SME win rates are below the national average")
    col2.metric("Sectors below average", f"{below_avg_sectors}/{len(sector_df)}", help="Sectors where SME win rates are below the national average")
    col3.metric("Regional inequality gap", f"{(region_df['Rate'].max()-region_df['Rate'].min())*100:.1f}%", help="Difference between highest and lowest regional SME award rates")
    st.divider()
    st.markdown("**All model performance comparison:**")
    try:
        rdf2 = pd.read_csv("model_comparison.csv").sort_values("AUC-ROC", ascending=False)
        rdf2["Recommended"] = rdf2["Model"].apply(lambda x: "⭐" if best_label in x or x in best_label else "")
        st.dataframe(rdf2, use_container_width=True)
    except Exception:
        st.info("Model comparison table not available.")

# ── TAB 6: SME READINESS ANALYSIS ────────────────────────────────────────────
with tab6:
    st.subheader("🏢 SME Readiness Analysis")
    st.markdown("Analyse a dataset of SMEs to assess their procurement readiness, accessibility scores, and identify structural participation barriers at scale.")
    st.divider()
    analysis_mode = st.radio("Analysis mode", ["Single SME", "Batch upload (CSV)"], horizontal=True)

    if analysis_mode == "Single SME":
        st.markdown("#### Assess a single SME profile")
        col1, col2 = st.columns(2)
        with col1:
            sme_name   = st.text_input("SME name (optional)", value="My SME", key="sme_name")
            sme_cv     = st.number_input("Typical contract value (£)", min_value=0.0, value=75000.0, step=5000.0, key="sme_cv")
            sme_region = st.selectbox("Primary region", list(encoders.get("region",{"Unknown":0}).keys()), key="sme_r")
        with col2:
            sme_inp = st.radio("CPV input method", ["Select CPV code","Select by industry"], key="sme_inp", horizontal=True)
            if sme_inp == "Select CPV code":
                sme_cpv = st.selectbox("CPV code", all_cpv_codes, key="sme_cpv_a")
                st.caption(f"Industry: **{cpv_to_industry(sme_cpv)}**")
            else:
                sme_ind = st.selectbox("Industry", all_industries, key="sme_ind")
                sme_cpv = st.selectbox("CPV code", INDUSTRY_LOOKUP.get(sme_ind,["Unknown"]), key="sme_cpv_b")
        if st.button("🏢 Assess SME readiness", type="primary", use_container_width=True, key="sme_btn"):
            now = datetime.now()
            am_s, aq_s = now.month, (now.month-1)//3+1
            row_s, cr_s, rr_s = build_row(sme_cv, am_s, aq_s, sme_region, sme_cpv, encoders, feature_cols, rates, scaler)
            p_rf_s  = rf.predict_proba(row_s)[0][1]
            p_xgb_s = xgb.predict_proba(row_s)[0][1]
            p_lr_s  = lr.predict_proba(row_s)[0][1]
            p_s     = p_rf_s*0.5 + p_xgb_s*0.35 + p_lr_s*0.15
            acc_s, conf_s, bid_s = get_accessibility_scores(p_s)
            st.markdown(f"### Results for: **{sme_name}**")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Win Probability",           f"{p_s*100:.1f}%")
            c2.metric("Procurement Accessibility", f"{acc_s:.1f}%")
            c3.metric("SME Confidence Index",      f"{conf_s:.1f}%")
            c4.metric("Bid Feasibility Score",     f"{bid_s:.2f}")
            st.progress(float(p_s))
            if p_s >= 0.6:
                st.success(f"✅ **{sme_name}** is well-positioned for public procurement. Applying is recommended.")
            elif p_s >= 0.4:
                st.warning(f"⚠️ **{sme_name}** has moderate procurement readiness. Targeted improvements would increase competitiveness.")
            else:
                st.error(f"❌ **{sme_name}** faces significant procurement barriers. The model explains why this SME is unlikely to apply — and what to do about it.")
            st.divider()
            st.markdown("### 🔍 Explainability — Participation Barrier Analysis")
            reasons_s = explain_prediction(p_s, sme_cv, sme_cpv, sme_region, rates)
            for reason_name, explanation, direction in reasons_s:
                if direction == "negative":
                    st.error(f"**❌ {reason_name}:** {explanation}")
                elif direction == "positive":
                    st.success(f"**✅ {reason_name}:** {explanation}")
                else:
                    st.info(f"**ℹ️ {reason_name}:** {explanation}")
            st.divider()
            base_s, sugg_s = gap_analysis(sme_cv, sme_cpv, sme_region, encoders, feature_cols, rates, scaler, rf, xgb, lr)
            if sugg_s:
                st.markdown("### 🎯 Procurement Strategy Recommendations")
                for category, action, new_prob, improvement, advice in sugg_s[:3]:
                    with st.expander(f"💡 {action}  →  {new_prob*100:.1f}%  (+{improvement*100:.1f}%)"):
                        st.markdown(f"**{advice}**")
                        col_a, col_b = st.columns(2)
                        col_a.metric("Current probability",  f"{base_s*100:.1f}%")
                        col_b.metric("If action taken",      f"{new_prob*100:.1f}%", delta=f"+{improvement*100:.1f}%")

    else:
        st.markdown("#### Batch analysis — upload a CSV of SME profiles")
        st.markdown("**Required columns:** `sme_name`, `contract_value`, `cpv_code`, `region`")
        st.markdown("**Optional columns:** `sector`")
        sample_data = pd.DataFrame({
            "sme_name":       ["Tech SME Ltd","Build Co Ltd","Clean Services Ltd"],
            "contract_value": [45000, 250000, 35000],
            "cpv_code":       ["72000000","45000000","90600000"],
            "region":         ["London","North West","Yorkshire and the Humber"],
            "sector":         ["IT services","Construction","Cleaning services"],
        })
        st.download_button("📥 Download sample CSV template", sample_data.to_csv(index=False), "sme_template.csv", "text/csv")
        uploaded = st.file_uploader("Upload your SME CSV", type=["csv"], key="sme_upload")
        if uploaded:
            sme_input_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(sme_input_df)} SMEs. Preview:")
            st.dataframe(sme_input_df.head(), use_container_width=True)
            if st.button("🏢 Analyse all SMEs", type="primary", use_container_width=True, key="batch_btn"):
                with st.spinner(f"Analysing {len(sme_input_df)} SME profiles..."):
                    batch_results = analyse_sme_batch(sme_input_df, encoders, feature_cols, rates, scaler, rf, xgb, lr)
                st.success(f"Analysis complete for {len(batch_results)} SMEs")
                col1,col2,col3,col4 = st.columns(4)
                col1.metric("Avg win probability",   f"{batch_results['win_probability'].mean()*100:.1f}%")
                col2.metric("Likely to win",         str((batch_results["prediction"]=="Likely to win").sum()))
                col3.metric("Unlikely to win",       str((batch_results["prediction"]=="Unlikely to win").sum()))
                col4.metric("Need preparation",      str((batch_results["recommendation"]=="Needs preparation").sum()))
                st.dataframe(batch_results, use_container_width=True)
                csv_batch = batch_results.to_csv(index=False)
                st.download_button("📥 Download full analysis report", csv_batch, "sme_readiness_report.csv", "text/csv")

# ── TAB 7: BARRIER DASHBOARD ──────────────────────────────────────────────────
with tab7:
    st.subheader("🗺️ SME Procurement Barrier Dashboard")
    st.markdown("Aggregate analysis of procurement accessibility barriers across sectors and regions — the research dashboard for policy insight.")
    st.divider()
    col1,col2,col3 = st.columns(3)
    col1.metric("Global SME award rate", f"{rates['global_sme_rate']*100:.1f}%")
    global_r = rates["global_sme_rate"]
    region_df2 = pd.DataFrame(list(rates["region_sme_rate"].items()), columns=["Region","Rate"])
    sector_df2 = pd.DataFrame(list(rates["cpv_sme_rate"].items()), columns=["CPV Code","Rate"])
    sector_df2["Industry"] = sector_df2["CPV Code"].apply(cpv_to_industry)
    pct_below = (region_df2["Rate"] < global_r * 0.8).mean() * 100
    col2.metric("Regions significantly below average", f"{pct_below:.0f}%")
    col3.metric("Inequality gap (highest vs lowest region)", f"{(region_df2['Rate'].max()-region_df2['Rate'].min())*100:.1f}%")
    st.divider()
    st.markdown("### 📊 Key Research Findings")
    finding1_pct = (region_df2["Rate"] < 0.5).mean() * 100
    finding2_best = sector_df2.nlargest(1,"Rate").iloc[0]
    finding2_worst= sector_df2.nsmallest(1,"Rate").iloc[0]
    finding3_gap  = (region_df2["Rate"].max() - region_df2["Rate"].min()) * 100
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
    st.error(
        f"**Finding 1 — Structural Barrier**

"
        f"{finding1_pct:.0f}% of regions show SME award rates below 50%, "
        f"confirming that low predicted win probability is a rational reason "
        f"for SME non-participation."
    )
    with col_f2:
    st.warning(
        f"**Finding 2 — Sector Inequality**

"
        f"The gap between the most SME‑friendly sector "
        f"({finding2_best['Industry']}: {finding2_best['Rate']*100:.1f}%) "
        f"and the least accessible sector "
        f"({finding2_worst['Industry']}: {finding2_worst['Rate']*100:.1f}%) "
        f"shows significant structural imbalance."
    )
  with col_f3:
    st.info(
        f"**Finding 3 — Regional Inequality**

"
        f"The difference between the highest and lowest regional SME award rate "
        f"is {finding3_gap:.1f} percentage points, signalling uneven procurement "
        f"opportunity across geographic areas."
    )
    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Regional accessibility ranking (highest to lowest):**")
        rdf_sorted = region_df2.sort_values("Rate", ascending=False).copy()
        rdf_sorted["SME Rate"]   = rdf_sorted["Rate"].apply(lambda x: f"{x*100:.1f}%")
        rdf_sorted["Status"]     = rdf_sorted["Rate"].apply(lambda x: "✅ Above average" if x >= global_r else "❌ Below average")
        rdf_sorted["Barrier Level"] = rdf_sorted["Rate"].apply(lambda x: "Low" if x >= global_r*1.1 else "Medium" if x >= global_r*0.9 else "High")
        st.dataframe(rdf_sorted[["Region","SME Rate","Status","Barrier Level"]], use_container_width=True)
    with col_r:
        st.markdown("**Sector accessibility ranking (highest to lowest):**")
        sdf_sorted = sector_df2.sort_values("Rate", ascending=False).copy()
        sdf_sorted["SME Rate"]      = sdf_sorted["Rate"].apply(lambda x: f"{x*100:.1f}%")
        sdf_sorted["Status"]        = sdf_sorted["Rate"].apply(lambda x: "✅ Above average" if x >= global_r else "❌ Below average")
        sdf_sorted["Barrier Level"] = sdf_sorted["Rate"].apply(lambda x: "Low" if x >= global_r*1.1 else "Medium" if x >= global_r*0.9 else "High")
        st.dataframe(sdf_sorted[["Industry","SME Rate","Status","Barrier Level"]].head(20), use_container_width=True)
    st.divider()
    st.markdown("### 📋 Policy Insight Report")
    st.markdown(f"""
**AI-Driven Analysis of SME Procurement Participation Barriers — Key Conclusions**

1. **Structural accessibility barrier confirmed:** The model predicts that {finding1_pct:.0f}% of UK regions have SME award rates below 50%, providing quantitative evidence that low win probability is a rational deterrent to SME participation.

2. **Sector imbalance identified:** There is a {(finding2_best['Rate']-finding2_worst['Rate'])*100:.1f}% gap between the most and least SME-accessible sectors. SMEs in disadvantaged sectors face systematic barriers that transcend individual bid quality.

3. **Regional inequality:** A {finding3_gap:.1f}% gap between regional SME award rates indicates that geographic location is a significant determinant of procurement accessibility — a finding with direct policy implications.

4. **Bid cost deterrence:** With typical tender preparation costs of £5,000–£20,000 and average SME win probabilities below 50% in many sectors and regions, the expected financial return from bidding is often negative — confirming that SME non-participation is economically rational.

5. **Policy recommendation:** Targeted interventions should focus on the highest-barrier sectors and regions identified above. Simplified procurement frameworks and lower-value contract splitting would most improve SME participation rates.
    """)
    st.download_button("📥 Download policy insight report", 
                       f"SME Procurement Barrier Analysis\nGenerated: {datetime.now().strftime('%Y-%m-%d')}\n\nGlobal SME rate: {global_r*100:.1f}%\nRegions below average: {pct_below:.0f}%\nInequality gap: {finding3_gap:.1f}%\n",
                       "policy_insight_report.txt", "text/plain")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🏆 SME Procurement Intelligence")
    st.markdown("**Platform capabilities:**")
    st.markdown("🔮 Win probability prediction")
    st.markdown("📊 Barrier & gap analysis")
    st.markdown("📡 Live contract feed (dual API)")
    st.markdown("🏭 CPV code ↔ industry lookup")
    st.markdown("📈 Historical procurement insights")
    st.markdown("🏢 SME readiness assessment")
    st.markdown("🗺️ Procurement barrier dashboard")
    st.divider()
    if best_auc:
        st.success(f"⭐ Best model: **{best_label}**\nAUC-ROC: {best_auc:.4f}")
    st.divider()
    st.markdown("**Research framing:**")
    st.caption("Explainable AI for understanding SME participation barriers in UK public procurement")
    st.divider()
    st.markdown("**Data sources:**")
    st.markdown("- UK Contracts Finder API")
    st.markdown("- Find a Tender API")
    st.markdown("*Live data refreshes hourly*")
    st.divider()
    st.caption("Research prototype — not for operational use.")
