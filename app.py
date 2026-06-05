import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

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

UNIVERSAL_CERTS = [
    {"name":"Cyber Essentials","type":"Mandatory","issuer":"NCSC / IASME","cost":"£300 - £500","timeline":"1-2 weeks","required_by":"All UK government contracts involving IT or data (PPN 014)","why":"Mandatory baseline for all public sector contracts involving technology or personal data.","url":"https://www.ncsc.gov.uk/cyberessentials/overview"},
    {"name":"Cyber Essentials Plus","type":"Strongly recommended","issuer":"NCSC / IASME","cost":"£1,500 - £4,000","timeline":"2-4 weeks","required_by":"MOD contracts, NHS sensitive data, Scottish Government digital","why":"Higher assurance level with independent technical audit.","url":"https://www.ncsc.gov.uk/cyberessentials/overview"},
    {"name":"ISO 9001 — Quality Management","type":"Strongly recommended","issuer":"UKAS-accredited body","cost":"£3,000 - £15,000","timeline":"3-6 months","required_by":"NHS contracts, most central government frameworks, Crown Commercial Service","why":"Demonstrates quality management systems. Heavily weighted in evaluation criteria across all public sector contracts.","url":"https://www.bsigroup.com/en-GB/iso-9001-quality-management/"},
    {"name":"ISO 14001 — Environmental Management","type":"Strongly recommended","issuer":"UKAS-accredited body","cost":"£3,000 - £12,000","timeline":"3-6 months","required_by":"Contracts over £5m, construction, transport, facilities","why":"Required under PPN 06/21 for contracts over £5m.","url":"https://www.bsigroup.com/en-GB/iso-14001-environmental-management/"},
]

INDUSTRY_CERTS = {
    "Construction": [
        {"name":"ISO 45001 — Occupational Health and Safety","type":"Mandatory","issuer":"UKAS-accredited body","cost":"£3,000 - £12,000","timeline":"3-6 months","required_by":"All construction contracts","why":"Mandatory for all public construction contracts under CDM Regulations 2015."},
        {"name":"SSIP — Safety Schemes in Procurement","type":"Mandatory","issuer":"CHAS, SafeContractor, Constructionline","cost":"£300 - £800 per year","timeline":"2-4 weeks","required_by":"All construction public contracts","why":"Industry-standard pre-qualification for construction."},
        {"name":"Constructionline Gold","type":"Strongly recommended","issuer":"Constructionline","cost":"£500 - £2,000 per year","timeline":"2-4 weeks","required_by":"Major public construction frameworks","why":"UK leading procurement and supply chain management service."},
    ],
    "Health services": [
        {"name":"CQC Registration","type":"Mandatory","issuer":"Care Quality Commission","cost":"£895 - £2,895 per year","timeline":"8-16 weeks","required_by":"All NHS and local authority health contracts","why":"Legally required to provide regulated health and social care activities."},
        {"name":"DSP Toolkit","type":"Mandatory","issuer":"NHS England","cost":"Free","timeline":"4-8 weeks","required_by":"All NHS contracts involving patient data","why":"Mandatory for all suppliers accessing NHS systems or patient data."},
        {"name":"ISO 27001 — Information Security","type":"Strongly recommended","issuer":"UKAS-accredited body","cost":"£6,000 - £25,000","timeline":"6-12 months","required_by":"NHS contracts handling sensitive patient data","why":"NHS Digital increasingly requires ISO 27001 for suppliers handling sensitive health data."},
    ],
    "IT services": [
        {"name":"ISO 27001 — Information Security Management","type":"Mandatory","issuer":"UKAS-accredited body","cost":"£6,000 - £25,000","timeline":"6-12 months","required_by":"HMRC, MOD, Home Office, Cabinet Office, NHS digital contracts","why":"Effectively mandatory for IT service suppliers to central government."},
        {"name":"G-Cloud Supplier Registration","type":"Mandatory","issuer":"Crown Commercial Service","cost":"Free","timeline":"2-4 weeks","required_by":"All cloud technology contracts with central government","why":"Primary route for IT suppliers to sell cloud services to public sector."},
        {"name":"UK GDPR / Data Protection Registration","type":"Mandatory","issuer":"ICO","cost":"£40 - £2,900 per year","timeline":"1-2 weeks","required_by":"All IT contracts handling personal data","why":"Legally required for any organisation processing personal data."},
    ],
    "Software programming services": [
        {"name":"ISO 27001 — Information Security Management","type":"Mandatory","issuer":"UKAS-accredited body","cost":"£6,000 - £25,000","timeline":"6-12 months","required_by":"All government software development contracts","why":"Essential for software suppliers to public sector."},
        {"name":"G-Cloud Supplier Registration","type":"Mandatory","issuer":"Crown Commercial Service","cost":"Free","timeline":"2-4 weeks","required_by":"All SaaS and software contracts with public sector","why":"Primary route to sell software to public sector."},
    ],
    "Education and training services": [
        {"name":"Ofsted Registration","type":"Mandatory","issuer":"Ofsted","cost":"Free","timeline":"Variable","required_by":"DfE, local authority education contracts","why":"Required for providers of education and training services to publicly funded bodies."},
        {"name":"Matrix Standard","type":"Strongly recommended","issuer":"Matrix Standard","cost":"£1,500 - £3,500","timeline":"3-6 months","required_by":"DWP employment and training contracts","why":"Quality standard for information, advice, and guidance services."},
    ],
    "Transport services": [
        {"name":"Operator Licence (O-Licence)","type":"Mandatory","issuer":"Traffic Commissioners","cost":"£257 - £401","timeline":"6-9 weeks","required_by":"All HGV and PSV transport contracts","why":"Legal requirement to operate goods vehicles over 3.5 tonnes or passenger vehicles."},
        {"name":"ISO 45001 — Occupational Health and Safety","type":"Strongly recommended","issuer":"UKAS-accredited body","cost":"£3,000 - £12,000","timeline":"3-6 months","required_by":"All transport public contracts","why":"Required for rail, aviation, and major logistics frameworks."},
    ],
    "Cleaning services": [
        {"name":"ISO 45001 — Occupational Health and Safety","type":"Mandatory","issuer":"UKAS-accredited body","cost":"£3,000 - £12,000","timeline":"3-6 months","required_by":"NHS, schools, government buildings cleaning contracts","why":"Mandatory for cleaning contracts due to chemical handling risks."},
        {"name":"COSHH Assessment Compliance","type":"Mandatory","issuer":"Internal (HSE requirement)","cost":"£500 - £2,000","timeline":"1-2 weeks","required_by":"All cleaning contracts involving chemical use","why":"Legal requirement under Control of Substances Hazardous to Health regulations."},
    ],
    "Legal services": [
        {"name":"SRA Authorisation","type":"Mandatory","issuer":"Solicitors Regulation Authority","cost":"Varies by firm size","timeline":"Variable","required_by":"All public sector legal services contracts","why":"Legally required to provide legal services."},
        {"name":"Lexcel — Law Society Quality Mark","type":"Strongly recommended","issuer":"The Law Society","cost":"£1,000 - £3,000","timeline":"3-6 months","required_by":"Legal Aid Agency contracts, local authority legal panels","why":"Required by Legal Aid Agency for legal aid contracts."},
    ],
    "Environmental services": [
        {"name":"ISO 14001 — Environmental Management","type":"Mandatory","issuer":"UKAS-accredited body","cost":"£3,000 - £12,000","timeline":"3-6 months","required_by":"All environmental services public contracts","why":"Mandatory for environmental services contracts."},
        {"name":"Environment Agency Waste Carrier Licence","type":"Mandatory","issuer":"Environment Agency","cost":"£154 (3-year registration)","timeline":"1-2 weeks","required_by":"All waste collection and disposal contracts","why":"Legal requirement to transport waste."},
    ],
    "Repair and maintenance services": [
        {"name":"ISO 45001 — Occupational Health and Safety","type":"Mandatory","issuer":"UKAS-accredited body","cost":"£3,000 - £12,000","timeline":"3-6 months","required_by":"All facilities management and maintenance contracts","why":"Mandatory for maintenance contracts in public buildings."},
        {"name":"Gas Safe Registration","type":"Mandatory (if applicable)","issuer":"Gas Safe Register","cost":"£165 - £655 per year","timeline":"Variable","required_by":"Any contract involving gas appliances or pipework","why":"Legal requirement to work on gas appliances."},
        {"name":"NICEIC / ECA Registration","type":"Mandatory (if applicable)","issuer":"NICEIC / ECA","cost":"£400 - £1,500 per year","timeline":"2-6 weeks","required_by":"Any contract involving electrical installation","why":"Required for electrical work in public buildings."},
    ],
    "Social work services": [
        {"name":"CQC Registration","type":"Mandatory","issuer":"Care Quality Commission","cost":"£895 - £2,895 per year","timeline":"8-16 weeks","required_by":"All local authority and NHS social care contracts","why":"Legally required to provide regulated social care activities."},
    ],
    "Accounting services": [
        {"name":"ICAEW / ACCA Membership","type":"Mandatory","issuer":"ICAEW or ACCA","cost":"Annual membership fees","timeline":"Variable","required_by":"Public sector audit and accounts contracts","why":"Professional body membership required for statutory audit work."},
    ],
    "Hotel, restaurant and catering services": [
        {"name":"Food Hygiene Rating 4 or 5","type":"Mandatory","issuer":"Local Authority Environmental Health","cost":"Free (inspection-based)","timeline":"Variable","required_by":"All public sector catering contracts","why":"Minimum rating 4 required for public sector catering contracts."},
        {"name":"ISO 22000 — Food Safety Management","type":"Strongly recommended","issuer":"UKAS-accredited body","cost":"£4,000 - £15,000","timeline":"3-6 months","required_by":"NHS catering contracts","why":"Required by NHS Supply Chain for catering suppliers."},
    ],
    "Financial and insurance services": [
        {"name":"FCA Authorisation","type":"Mandatory","issuer":"Financial Conduct Authority","cost":"£1,500 - £25,000","timeline":"6-12 months","required_by":"All public sector financial services contracts","why":"Legally required for all regulated financial activities."},
    ],
    "Research and development services": [
        {"name":"Innovate UK Registration","type":"Required for specific contracts","issuer":"Innovate UK / UKRI","cost":"Free","timeline":"1-2 weeks","required_by":"All Innovate UK and UKRI research contracts","why":"Required to receive Innovate UK grants and bid on UKRI contracts."},
    ],
}

def get_certs_for_industry(industry_name):
    return UNIVERSAL_CERTS + INDUSTRY_CERTS.get(industry_name, [])

def get_certs_for_cpv(cpv_code):
    industry = cpv_to_industry(cpv_code)
    return industry, get_certs_for_industry(industry)

def compute_sme_capability_score(turnover, years_active, employee_count, prior_contracts, certifications, contract_value):
    score = 0; breakdown = []
    if turnover > 0 and contract_value > 0:
        ratio = turnover / contract_value
        if ratio >= 2.0:
            score += 25; breakdown.append(("Turnover vs Contract Value", 25, "Strong", "Turnover is 2x+ contract value — meets eligibility threshold."))
        elif ratio >= 1.0:
            score += 15; breakdown.append(("Turnover vs Contract Value", 15, "Moderate", "Turnover meets contract value but below 2x threshold preferred by buyers."))
        else:
            score += 5; breakdown.append(("Turnover vs Contract Value", 5, "Weak", "Turnover below contract value — most buyers require 1-2x turnover. Critical barrier."))
    if years_active >= 5:
        score += 20; breakdown.append(("Years Active", 20, "Strong", str(years_active) + " years — established track record."))
    elif years_active >= 3:
        score += 12; breakdown.append(("Years Active", 12, "Moderate", str(years_active) + " years — some track record."))
    elif years_active >= 1:
        score += 6; breakdown.append(("Years Active", 6, "Weak", str(years_active) + " year(s) — early stage. Many contracts require 2-3 years minimum."))
    else:
        breakdown.append(("Years Active", 0, "Critical barrier", "Under 1 year — most frameworks exclude very new businesses."))
    if employee_count >= 50:
        score += 20; breakdown.append(("Employee Count", 20, "Strong", str(employee_count) + " employees — sufficient capacity for most public contracts."))
    elif employee_count >= 10:
        score += 13; breakdown.append(("Employee Count", 13, "Moderate", str(employee_count) + " employees — adequate for smaller contracts."))
    elif employee_count >= 3:
        score += 7; breakdown.append(("Employee Count", 7, "Weak", str(employee_count) + " employees — may need subcontracting capability."))
    else:
        breakdown.append(("Employee Count", 2, "Critical barrier", "Under 3 employees — most buyers question delivery capacity."))
    if prior_contracts >= 5:
        score += 20; breakdown.append(("Prior Public Contracts", 20, "Strong", str(prior_contracts) + " contracts — strong track record."))
    elif prior_contracts >= 2:
        score += 12; breakdown.append(("Prior Public Contracts", 12, "Moderate", str(prior_contracts) + " contracts — some experience."))
    elif prior_contracts == 1:
        score += 6; breakdown.append(("Prior Public Contracts", 6, "Weak", "Only 1 prior contract — limited track record."))
    else:
        breakdown.append(("Prior Public Contracts", 0, "Critical barrier", "No prior contracts — most common SME barrier."))
    if certifications >= 3:
        score += 15; breakdown.append(("Certifications", 15, "Strong", str(certifications) + " certs — ISO, Cyber Essentials improve bid scores significantly."))
    elif certifications >= 1:
        score += 8; breakdown.append(("Certifications", 8, "Moderate", str(certifications) + " cert(s) — additional certs would strengthen bids."))
    else:
        breakdown.append(("Certifications", 0, "Weak", "No certifications — many contracts require ISO 9001 or Cyber Essentials minimum."))
    return min(score, 100), breakdown

def combined_score(win_prob, capability_score):
    return round((win_prob * 0.6 + capability_score / 100 * 0.4) * 100, 1)

def get_accessibility_scores(p):
    acc  = round(p * 100, 1)
    conf = round((1 - abs(p - 0.5) * 2) * 100, 1) if p < 0.5 else round(p * 100, 1)
    bid  = round(min(100, p * 120) * 100, 1) / 100
    return acc, conf, bid

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

@st.cache_data(ttl=3600)
def fetch_live_contracts(days_back=7, max_results=50):
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to   = datetime.now().strftime("%Y-%m-%d")
    all_contracts = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SMEResearchBot/1.0)",
        "Accept": "application/json", "Content-Type": "application/json",
        "Origin": "https://www.contractsfinder.service.gov.uk",
        "Referer": "https://www.contractsfinder.service.gov.uk/",
    }
    try:
        cf_r = requests.post("https://www.contractsfinder.service.gov.uk/Published/Notices/PublicSearch/json",
            json={"searchCriteria": {"page":1,"publishedFrom":date_from,"publishedTo":date_to,"size":max_results}},
            headers=headers, timeout=25)
        if cf_r.status_code == 200:
            for n in cf_r.json().get("results", []):
                item = n.get("item", {}); cpv_list = item.get("cpvCodes", [])
                cpv = cpv_list[0].get("code","Unknown") if cpv_list else "Unknown"
                locs = item.get("deliveryLocations",[]); region = locs[0].get("region","Unknown") if locs else "Unknown"
                val = item.get("value",{}) or {}
                all_contracts.append({"source":"Contracts Finder","title":str(item.get("title","Unknown"))[:100],
                    "buyer":str(item.get("organizationName","Unknown"))[:60],"value":float(val.get("amount",0) or 0),
                    "cpv_code":str(cpv),"region":str(region),"deadline":str(item.get("tenderDeadline","Not specified")),
                    "published":str(item.get("publishedAt","Unknown")),"url":str(n.get("publishedUrl",""))})
    except Exception: pass
    try:
        fat_h = {**headers,
                 "Origin":"https://www.find-tender.service.gov.uk","Referer":"https://www.find-tender.service.gov.uk/"}
        fat_r = requests.get("https://www.find-tender.service.gov.uk/api/1.0/ocds/notices/list",
            params={"publishedFrom":date_from,"publishedTo":date_to,"limit":max_results,"offset":0},
            headers=fat_h, timeout=25)
        if fat_r.status_code == 200:
            for rec in fat_r.json().get("records", fat_r.json().get("releases",[])):
                rel = rec.get("compiledRelease",rec); tender = rel.get("tender",{})
                buyer = rel.get("buyer",{}).get("name","Unknown")[:60]
                items = tender.get("items",[]); cpv = items[0].get("classification",{}).get("id","Unknown") if items else "Unknown"
                locs = tender.get("deliveryLocations",[]); region = locs[0].get("region","Unknown") if locs else "Unknown"
                val = tender.get("value",{}) or {}
                all_contracts.append({"source":"Find a Tender","title":str(tender.get("title","Unknown"))[:100],
                    "buyer":str(buyer),"value":float(val.get("amount",0) or 0),"cpv_code":str(cpv),"region":str(region),
                    "deadline":str(tender.get("tenderPeriod",{}).get("endDate","Not specified")),
                    "published":str(rel.get("date","Unknown")),"url":str(rel.get("ocid",""))})
    except Exception: pass
    if not all_contracts:
        import random; random.seed(int(datetime.now().timestamp()) % 1000)
        sectors=[("IT services","72000000"),("Construction","45000000"),("Cleaning services","90600000"),
                 ("Training services","80500000"),("Health services","85100000"),("Accounting services","79200000"),
                 ("Transport services","60000000"),("Legal services","79100000"),
                 ("Repair and maintenance","50000000"),("Environmental services","90700000")]
        buyers_cf=["NHS Trust","Local Council","Ministry of Justice","HMRC","Home Office"]
        buyers_fat=["Cabinet Office","MOD","DVLA","Crown Commercial Service","UKRI"]
        regions=["London","South East","North West","Yorkshire and the Humber","East Midlands",
                 "West Midlands","East of England","South West","North East","Wales","Scotland"]
        for i in range(min(max_results,30)):
            sname,cpv=random.choice(sectors); source="Contracts Finder" if i%2==0 else "Find a Tender"
            buyers=buyers_cf if source=="Contracts Finder" else buyers_fat
            value=round(random.uniform(10000,500000),2)
            pub_dt=datetime.now()-timedelta(days=random.randint(0,days_back))
            ddl_dt=pub_dt+timedelta(days=random.randint(14,60))
            all_contracts.append({"source":source,"title":sname+" Services Contract 2026"+str(i).zfill(3),
                "buyer":random.choice(buyers),"value":value,"cpv_code":cpv,"region":random.choice(regions),
                "deadline":ddl_dt.strftime("%Y-%m-%d"),"published":pub_dt.strftime("%Y-%m-%dT%H:%M:%S"),"url":""})
    df=pd.DataFrame(all_contracts)
    df=df.drop_duplicates(subset=["title","buyer"],keep="first")
    df["value"]=pd.to_numeric(df["value"],errors="coerce").fillna(0)
    return df.sort_values("published",ascending=False).reset_index(drop=True)

def build_row(cv,am,aq,region,cpv,encoders,feature_cols,rates,scaler):
    log_cv=np.log1p(cv)
    vbnum=0 if cv<10000 else 1 if cv<50000 else 2 if cv<100000 else 3 if cv<500000 else 4
    is_qe=int(aq in [1,4]); is_hv=int(cv>100000)
    cr=rates["cpv_sme_rate"].get(str(cpv),rates["global_sme_rate"])
    rr=rates["region_sme_rate"].get(str(region),rates["global_sme_rate"])
    r_enc=encoders.get("region",{}).get(str(region),0)
    c_enc=encoders.get("cpv_code",{}).get(str(cpv),0)
    d={"log_contract_value":log_cv,"value_band_num":vbnum,"award_month":am,"award_quarter":aq,
       "is_quarter_end":is_qe,"is_high_value":is_hv,"buyer_sme_rate":rates["global_sme_rate"],
       "cpv_sme_rate":cr,"region_sme_rate":rr,"value_band_enc":0,"region_enc":r_enc,"cpv_code_enc":c_enc}
    row=pd.DataFrame([{c:d.get(c,0) for c in feature_cols}])
    return scaler.transform(row.values),cr,rr

def get_ensemble(rs,rf,xgb,lr):
    return rf.predict_proba(rs)[0][1]*0.5+xgb.predict_proba(rs)[0][1]*0.35+lr.predict_proba(rs)[0][1]*0.15

def score_contracts(df,encoders,feature_cols,rates,scaler,rf,xgb,lr):
    now=datetime.now(); am,aq=now.month,(now.month-1)//3+1; probs=[]
    for _,row in df.iterrows():
        try:
            cv=float(row.get("value",50000) or 50000); cpv=str(row.get("cpv_code","Unknown")); reg=str(row.get("region","Unknown"))
            r,_,_=build_row(cv,am,aq,reg,cpv,encoders,feature_cols,rates,scaler)
            probs.append(round(get_ensemble(r,rf,xgb,lr),3))
        except Exception: probs.append(rates["global_sme_rate"])
    return probs

def explain_prediction(p,cv,cpv,region,rates):
    reasons=[]
    cr=rates["cpv_sme_rate"].get(str(cpv),rates["global_sme_rate"])
    rr=rates["region_sme_rate"].get(str(region),rates["global_sme_rate"])
    gr=rates["global_sme_rate"]
    if cv>150000:
        reasons.append(("High contract value","At £"+"{:,.0f}".format(cv)+", exceeds threshold where SMEs typically win. Large contracts favour established suppliers.","negative"))
    if cr<gr*0.8:
        reasons.append(("Sector disadvantage",cpv_to_industry(cpv)+" sector has "+"{:.1f}%".format(cr*100)+" SME award rate — below national average of "+"{:.1f}%".format(gr*100)+".","negative"))
    if rr<gr*0.8:
        reasons.append(("Regional disadvantage",region+" has "+"{:.1f}%".format(rr*100)+"% SME award rate — less favourable for SMEs.","negative"))
    if cv>100000:
        reasons.append(("Value mismatch","Contracts above £100,000 have significantly lower SME win rates.","negative"))
    if cr>gr*1.1:
        reasons.append(("Sector advantage",cpv_to_industry(cpv)+" sector has "+"{:.1f}%".format(cr*100)+"% SME award rate — above average. SME-friendly sector.","positive"))
    if rr>gr*1.1:
        reasons.append(("Regional advantage",region+" has "+"{:.1f}%".format(rr*100)+"% SME award rate — strong SME procurement culture.","positive"))
    if cv<50000:
        reasons.append(("Value advantage","At £"+"{:,.0f}".format(cv)+", within the range where SMEs are most competitive.","positive"))
    if not reasons:
        reasons.append(("Average conditions","Typical procurement characteristics. Win probability reflects baseline SME award rate.","neutral"))
    return reasons

def gap_analysis(cv,cpv,region,encoders,feature_cols,rates,scaler,rf,xgb,lr):
    now=datetime.now(); am,aq=now.month,(now.month-1)//3+1
    def gp(cv_,cpv_,reg_):
        r,_,_=build_row(cv_,am,aq,reg_,cpv_,encoders,feature_cols,rates,scaler)
        return get_ensemble(r,rf,xgb,lr)
    base=gp(cv,cpv,region); sugg=[]
    for test_cv,label in [(cv*0.5,"Target 50% smaller contract (£"+"{:,.0f}".format(cv*0.5)+")"),
                           (cv*0.25,"Target 75% smaller contract (£"+"{:,.0f}".format(cv*0.25)+")"),
                           (25000,"Target £25,000 contract")]:
        p2=gp(test_cv,cpv,region)
        if p2>base:
            sugg.append(("Contract Size",label,round(p2,3),round(p2-base,3),"Smaller contracts have higher SME win rates. Build track record first."))
    best_reg=max(rates["region_sme_rate"],key=rates["region_sme_rate"].get)
    p_br=gp(cv,cpv,best_reg)
    if p_br>base:
        sugg.append(("Region Strategy","Target "+best_reg,round(p_br,3),round(p_br-base,3),best_reg+" has highest SME award rate nationally."))
    best_cpv=max(rates["cpv_sme_rate"],key=rates["cpv_sme_rate"].get)
    p_bc=gp(cv,best_cpv,region)
    if p_bc>base:
        sugg.append(("Sector Pivot","Consider "+cpv_to_industry(best_cpv),round(p_bc,3),round(p_bc-base,3),cpv_to_industry(best_cpv)+" has highest SME win rate historically."))
    return base,sorted(sugg,key=lambda x:x[3],reverse=True)

def analyse_sme_batch(sme_df,encoders,feature_cols,rates,scaler,rf,xgb,lr):
    now=datetime.now(); am,aq=now.month,(now.month-1)//3+1; results=[]
    for idx,sme in sme_df.iterrows():
        try:
            cv=float(sme.get("contract_value",50000) or 50000)
            cpv=str(sme.get("cpv_code","Unknown")); reg=str(sme.get("region","Unknown"))
            turnover=float(sme.get("turnover",0) or 0); years=int(sme.get("years_active",0) or 0)
            employees=int(sme.get("employee_count",0) or 0); prior=int(sme.get("prior_contracts",0) or 0)
            certs=int(sme.get("certifications",0) or 0)
            r,cr,rr=build_row(cv,am,aq,reg,cpv,encoders,feature_cols,rates,scaler)
            p_ens=get_ensemble(r,rf,xgb,lr)
            cap_score,breakdown=compute_sme_capability_score(turnover,years,employees,prior,certs,cv)
            comb=combined_score(p_ens,cap_score)
            reasons=explain_prediction(p_ens,cv,cpv,reg,rates)
            market_bar=", ".join([x[0] for x in reasons if x[2]=="negative"]) or "None"
            cap_bar=", ".join([x[0] for x in breakdown if x[2] in ["Weak","Critical barrier"]]) or "None"
            results.append({
                "sme_name":sme.get("sme_name",sme.get("company_name","SME_"+str(idx))),
                "sector":sme.get("sector",cpv_to_industry(cpv)),"region":reg,
                "turnover":turnover,"years_active":years,"employee_count":employees,
                "prior_contracts":prior,"certifications":certs,"contract_value":cv,
                "ml_win_probability":round(p_ens,3),"capability_score":cap_score,
                "combined_readiness":comb,
                "prediction":"Likely to win" if p_ens>=0.5 else "Unlikely to win",
                "market_barriers":market_bar,"capability_barriers":cap_bar,
                "recommendation":"Apply" if comb>=70 else "Prepare first" if comb>=40 else "Development needed",
            })
        except Exception: pass
    return pd.DataFrame(results)

def make_policy_charts(ch_pred):
    fig=plt.figure(figsize=(20,14))
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)
    ax1=fig.add_subplot(gs[0,0])
    ax1.hist(ch_pred["ml_win_probability"],bins=20,color="#1D9E75",edgecolor="white")
    ax1.axvline(0.5,color="red",linestyle="--",linewidth=2,label="50% threshold")
    ax1.axvline(ch_pred["ml_win_probability"].mean(),color="orange",linestyle="--",linewidth=2,label="Mean: "+"{:.2f}".format(ch_pred["ml_win_probability"].mean()))
    ax1.set_title("SME Win Probability Distribution\n("+str(len(ch_pred))+" real UK SMEs)",fontweight="bold")
    ax1.set_xlabel("Predicted win probability"); ax1.set_ylabel("Number of SMEs"); ax1.legend(fontsize=8)
    ax2=fig.add_subplot(gs[0,1])
    sec_p=ch_pred.groupby("sector")["ml_win_probability"].mean().sort_values()
    colors2=["#D85A30" if p<0.5 else "#1D9E75" for p in sec_p.values]
    ax2.barh(range(len(sec_p)),sec_p.values,color=colors2,edgecolor="white")
    ax2.set_yticks(range(len(sec_p))); ax2.set_yticklabels([s[:25] for s in sec_p.index],fontsize=7)
    ax2.axvline(0.5,color="red",linestyle="--",linewidth=1.5)
    ax2.set_title("Average Win Probability by Sector",fontweight="bold"); ax2.set_xlabel("Mean predicted win probability")
    ax3=fig.add_subplot(gs[0,2])
    reg_p=ch_pred.groupby("region")["ml_win_probability"].mean().sort_values()
    colors3=["#D85A30" if p<0.5 else "#1D9E75" for p in reg_p.values]
    ax3.barh(range(len(reg_p)),reg_p.values,color=colors3,edgecolor="white")
    ax3.set_yticks(range(len(reg_p))); ax3.set_yticklabels(reg_p.index,fontsize=8)
    ax3.axvline(0.5,color="red",linestyle="--",linewidth=1.5)
    ax3.set_title("Average Win Probability by Region",fontweight="bold"); ax3.set_xlabel("Mean predicted win probability")
    ax4=fig.add_subplot(gs[1,0])
    ax4.hist(ch_pred["capability_score"],bins=20,color="#2E75B6",edgecolor="white")
    ax4.axvline(ch_pred["capability_score"].mean(),color="orange",linestyle="--",linewidth=2,label="Mean: "+"{:.1f}".format(ch_pred["capability_score"].mean()))
    ax4.set_title("SME Capability Score Distribution",fontweight="bold")
    ax4.set_xlabel("Capability score (0-100)"); ax4.set_ylabel("Number of SMEs"); ax4.legend(fontsize=8)
    ax5=fig.add_subplot(gs[1,1])
    bands=["Not ready\n(<30%)","Needs work\n(30-50%)","Moderate\n(50-70%)","Ready\n(>70%)"]
    counts=[
        (ch_pred["combined_readiness"]<30).sum(),
        ((ch_pred["combined_readiness"]>=30)&(ch_pred["combined_readiness"]<50)).sum(),
        ((ch_pred["combined_readiness"]>=50)&(ch_pred["combined_readiness"]<70)).sum(),
        (ch_pred["combined_readiness"]>=70).sum(),
    ]
    bar_colors=["#C0392B","#E67E22","#F1C40F","#1D9E75"]
    bars=ax5.bar(bands,counts,color=bar_colors,edgecolor="white")
    for bar,count in zip(bars,counts):
        ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                 str(count)+"\n("+"{:.0f}%".format(count/len(ch_pred)*100)+")",
                 ha="center",fontsize=9)
    ax5.set_title("SME Procurement Readiness Bands",fontweight="bold"); ax5.set_ylabel("Number of SMEs")
    ax6=fig.add_subplot(gs[1,2])
    scatter=ax6.scatter(ch_pred["ml_win_probability"],ch_pred["capability_score"],
                        c=ch_pred["combined_readiness"],cmap="RdYlGn",alpha=0.6,s=30)
    plt.colorbar(scatter,ax=ax6,label="Combined readiness %")
    ax6.axvline(0.5,color="red",linestyle="--",linewidth=1,alpha=0.5)
    ax6.axhline(50,color="blue",linestyle="--",linewidth=1,alpha=0.5)
    ax6.set_title("ML Win Probability vs Capability Score",fontweight="bold")
    ax6.set_xlabel("ML win probability"); ax6.set_ylabel("Capability score (0-100)")
    plt.suptitle("AI-Driven SME Procurement Accessibility Analysis\nBased on "+str(len(ch_pred))+" real UK SMEs from Companies House across "+str(ch_pred["sector"].nunique())+" sectors",
                 fontsize=13,fontweight="bold",y=1.01)
    plt.tight_layout()
    return fig

rf,xgb,lr,scaler,encoders,feature_cols,rates,best_name,best_auc=load_artefacts()
all_cpv_codes=list(encoders.get('cpv_code',{'Unknown':0}).keys())
all_industries=sorted(set(CPV_LOOKUP.values()))
MODEL_MAP={
    "Random Forest":("Random Forest",rf),"XGBoost":("XGBoost",xgb),
    "Logistic Regression":("Logistic Regression",lr),"Logistic Regression (baseline)":("Logistic Regression",lr)
}
best_label,_=MODEL_MAP.get(best_name,("Random Forest",rf))
model_options=[]
for opt in ["Random Forest","XGBoost","Logistic Regression"]:
    if opt==best_label:
        auc_str=" — AUC "+"{:.3f}".format(best_auc) if best_auc else ""
        model_options.append(opt+" Recommended"+auc_str)
    else: model_options.append(opt)
default_index=next((i for i,o in enumerate(model_options) if "Recommended" in o),0)

st.set_page_config(page_title="SME Procurement Intelligence",page_icon="trophy",layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem !important; max-width: 1380px; }
.stApp { background: #f0f2f8; }

h1 { font-weight: 700 !important; color: #1a1a2e !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 700 !important; color: #1a1a2e !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white; border-radius: 12px; padding: 6px;
    gap: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 16px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 16px; font-weight: 600;
    font-size: 0.80rem; letter-spacing: 0.01em; color: #6b7280;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #1a1a2e !important; color: white !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Metric cards */
[data-testid="stMetric"] {
    background: white; border-radius: 12px; padding: 16px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important; color: #9ca3af !important;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 800 !important; color: #1a1a2e; }

/* Buttons */
.stButton button {
    background: #1a1a2e; color: white; border: none; border-radius: 8px;
    padding: 8px 22px; font-weight: 600; font-size: 0.85rem;
    transition: all 0.15s ease;
}
.stButton button:hover { background: #2d2d4e; transform: translateY(-1px); }

/* Inputs */
.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextInput input {
    border-radius: 8px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

/* Alerts softer */
.stAlert { border-radius: 10px; }

/* Dividers tighter */
hr { margin: 1rem 0 !important; }

/* Expander */
.streamlit-expanderHeader { font-weight: 600; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


st.title("AI-Driven SME Procurement Accessibility Intelligence Platform")
st.markdown("Explainable AI revealing structural barriers affecting SME participation in UK public procurement.")
if best_auc:
    st.success("Best model: "+best_label+"  |  AUC-ROC: "+"{:.4f}".format(best_auc))
st.divider()

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8=st.tabs([
    "Win Probability","Barrier and Gap Analysis","Live Contracts",
    "CPV Lookup","Historical","SME Readiness","Barrier Dashboard","Certifications Guide"
])

with tab1:
    st.subheader("Will this SME win the contract?")
    st.markdown("Predicts win probability and provides a full explainability breakdown of why the SME is likely or unlikely to succeed.")
    c1,c2=st.columns(2)
    with c1:
        cv1=st.number_input("Contract value (£)",min_value=0.0,value=50000.0,step=1000.0,key="cv1")
        am1=st.selectbox("Award month",list(range(1,13)),format_func=lambda m:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1],key="am1")
        aq1=st.selectbox("Award quarter",[1,2,3,4],format_func=lambda q:"Q"+str(q),key="aq1")
        mc=st.selectbox("Model",model_options,index=default_index,key="mc1")
    with c2:
        r1=st.selectbox("Region",list(encoders.get("region",{"Unknown":0}).keys()),key="r1")
        inp1=st.radio("CPV input",["Select CPV code","Select by industry"],key="im1",horizontal=True)
        if inp1=="Select CPV code":
            cp1=st.selectbox("CPV code",all_cpv_codes,key="cp1a")
            st.caption("Industry: "+cpv_to_industry(cp1))
        else:
            ind1=st.selectbox("Industry",all_industries,key="ind1")
            cp1=st.selectbox("CPV code",INDUSTRY_LOOKUP.get(ind1,["Unknown"]),key="cp1b")
    if st.button("Predict win probability",type="primary",use_container_width=True,key="btn1"):
        row1,cr1,rr1=build_row(cv1,am1,aq1,r1,cp1,encoders,feature_cols,rates,scaler)
        p_rf1=rf.predict_proba(row1)[0][1]; p_xgb1=xgb.predict_proba(row1)[0][1]; p_lr1=lr.predict_proba(row1)[0][1]
        p_ens1=p_rf1*0.5+p_xgb1*0.35+p_lr1*0.15
        acc1,conf1,bid1=get_accessibility_scores(p_ens1)
        a,b,c,d=st.columns(4)
        a.metric("Win Probability","{:.1f}%".format(p_ens1*100))
        b.metric("Procurement Accessibility","{:.1f}%".format(acc1))
        c.metric("SME Confidence Index","{:.1f}%".format(conf1))
        d.metric("Bid Feasibility Score","{:.2f}".format(bid1))
        st.progress(float(p_ens1))
        if p_ens1>=0.5: st.success("This SME has a good chance. Applying is strongly recommended.")
        elif p_ens1>=0.35: st.warning("Borderline probability. With preparation this SME could compete. See Barrier and Gap Analysis tab.")
        else: st.error("Low win probability. This confirms why SMEs in this category hesitate to apply. See Barrier and Gap Analysis tab.")
        st.divider()
        st.markdown("### Explainability — Why is this the prediction?")
        for rname,rexpl,rdir in explain_prediction(p_ens1,cv1,cp1,r1,rates):
            if rdir=="negative": st.error("**"+rname+":** "+rexpl)
            elif rdir=="positive": st.success("**"+rname+":** "+rexpl)
            else: st.info("**"+rname+":** "+rexpl)
        st.divider()
        st.markdown("**Individual model contributions:**")
        col1,col2,col3,col4=st.columns(4)
        col1.metric("Random Forest (50%)","{:.1f}%".format(p_rf1*100))
        col2.metric("XGBoost (35%)","{:.1f}%".format(p_xgb1*100))
        col3.metric("Logistic Reg (15%)","{:.1f}%".format(p_lr1*100))
        col4.metric("Ensemble","{:.1f}%".format(p_ens1*100))
        st.info("Industry: "+cpv_to_industry(cp1)+"  |  Sector SME rate: "+"{:.1f}%".format(cr1*100)+"  |  Region SME rate: "+"{:.1f}%".format(rr1*100))
        st.divider()
        st.markdown("### SHAP Feature Attribution")
        if SHAP_AVAILABLE:
            try:
                explainer=shap.TreeExplainer(rf)
                sv=explainer.shap_values(row1)
                if isinstance(sv,list):
                    sv_use=sv[1][0]
                else:
                    sv_use=sv[0]
                contribs=sorted(zip(feature_cols,sv_use),key=lambda x:abs(x[1]),reverse=True)
                st.caption("SHAP values show how each feature pushed this specific prediction above or below the baseline. Positive values increase predicted SME win probability; negative values decrease it.")
                figs,axs=plt.subplots(figsize=(8,4))
                names=[c[0] for c in contribs[:8]][::-1]
                vals=[c[1] for c in contribs[:8]][::-1]
                colors=["#16a34a" if v>=0 else "#dc2626" for v in vals]
                axs.barh(names,vals,color=colors)
                axs.set_xlabel("SHAP value (impact on SME win probability)")
                axs.set_title("Top feature contributions to this prediction (Random Forest)")
                axs.axvline(0,color="#1a1a2e",linewidth=0.8)
                plt.tight_layout()
                st.pyplot(figs)
                plt.close(figs)
                st.markdown("**Plain-language reading:**")
                for fname,fval in contribs[:5]:
                    direction="increased" if fval>=0 else "decreased"
                    st.markdown("- **"+fname+"** "+direction+" the predicted win probability (SHAP "+"{:+.3f}".format(fval)+")")
            except Exception as e:
                st.warning("SHAP attribution could not be computed for this input: "+str(e))
        else:
            st.info("This deployment runs on Python 3.14, for which the SHAP library is not yet available. The domain-logic explainability above provides interpretable, plain-language reasoning for every prediction. SHAP-based attribution is discussed in the dissertation as a future extension.")

with tab2:
    st.subheader("Barrier and Capability Gap Analysis")
    st.markdown("Identifies structural barriers preventing SMEs from winning contracts and provides actionable recommendations to improve competitiveness.")
    c1,c2=st.columns(2)
    with c1:
        cv2=st.number_input("Contract value (£)",min_value=0.0,value=200000.0,step=1000.0,key="cv2")
        r2=st.selectbox("Region",list(encoders.get("region",{"Unknown":0}).keys()),key="r2")
    with c2:
        inp2=st.radio("CPV input",["Select CPV code","Select by industry"],key="im2",horizontal=True)
        if inp2=="Select CPV code":
            cp2=st.selectbox("CPV code",all_cpv_codes,key="cp2a"); st.caption("Industry: "+cpv_to_industry(cp2))
        else:
            ind2=st.selectbox("Industry",all_industries,key="ind2")
            cp2=st.selectbox("CPV code",INDUSTRY_LOOKUP.get(ind2,["Unknown"]),key="cp2b")
    if st.button("Run barrier and gap analysis",type="primary",use_container_width=True,key="btn2"):
        base_prob,suggestions=gap_analysis(cv2,cp2,r2,encoders,feature_cols,rates,scaler,rf,xgb,lr)
        acc2,conf2,bid2=get_accessibility_scores(base_prob)
        col1,col2,col3,col4=st.columns(4)
        col1.metric("Win probability","{:.1f}%".format(base_prob*100))
        col2.metric("Procurement Accessibility","{:.1f}%".format(acc2))
        col3.metric("SME Confidence Index","{:.1f}%".format(conf2))
        col4.metric("Bid Feasibility","{:.2f}".format(bid2))
        st.progress(float(base_prob))
        if base_prob<0.3: st.error("Only "+"{:.1f}%".format(base_prob*100)+"% — structural participation barrier confirmed.")
        elif base_prob<0.5: st.warning("{:.1f}%".format(base_prob*100)+"% — below threshold. Bid costs outweigh expected return.")
        else: st.success("{:.1f}%".format(base_prob*100)+"% — SMEs should be encouraged to apply.")
        st.divider()
        st.markdown("### Identified Participation Barriers")
        for rname,rexpl,rdir in explain_prediction(base_prob,cv2,cp2,r2,rates):
            if rdir=="negative": st.error("**"+rname+":** "+rexpl)
        st.divider()
        if suggestions:
            st.markdown("### Capability Gap Recommendations")
            for category,action,new_prob,improvement,advice in suggestions[:4]:
                with st.expander(action+" -> "+"{:.1f}%".format(new_prob*100)+" (+"+"{:.1f}%".format(improvement*100)+")"):
                    st.markdown("**"+category+":** "+advice); st.progress(float(new_prob))
                    ca,cb=st.columns(2)
                    ca.metric("Current","{:.1f}%".format(base_prob*100))
                    cb.metric("Improved","{:.1f}%".format(new_prob*100),delta="+"+"{:.1f}%".format(improvement*100))
        st.divider()
        st.markdown("### Why are SMEs not applying? — Model Evidence")
        st.markdown("For a **"+cpv_to_industry(cp2)+"** contract worth **£"+"{:,.0f}".format(cv2)+"** in **"+r2+"**:")
        if base_prob<0.3: st.error("With less than 30% probability, SME reluctance is rational. Bid preparation costs cannot be justified at this level.")
        elif base_prob<0.5: st.warning("At 30-50%, bid costs often outweigh expected returns. SMEs are making a rational economic calculation.")
        else: st.success("Above 50% — SMEs in this category should be actively encouraged to apply.")

with tab3:
    st.subheader("Live UK Government Contracts")
    st.markdown("Real-time contracts from **Contracts Finder** and **Find a Tender** — both sources running concurrently, each scored for SME win probability.")
    st.info("The UK government APIs restrict access by IP address. When live data is unavailable, a realistic representative sample is generated automatically.")
    c1,c2=st.columns(2)
    with c1: days_back=st.slider("Days to look back",1,30,7); max_res=st.selectbox("Max contracts per source",[25,50,100],index=1)
    with c2: filter_region=st.selectbox("Filter by region",["All"]+list(encoders.get("region",{"Unknown":0}).keys())); min_prob=st.slider("Min SME win probability",0.0,1.0,0.0,0.05)
    if st.button("Fetch and score live contracts",type="primary",use_container_width=True):
        with st.spinner("Fetching from Contracts Finder and Find a Tender..."): live_df=fetch_live_contracts(days_back=days_back,max_results=max_res)
        with st.spinner("Scoring contracts..."): live_df["sme_win_probability"]=score_contracts(live_df,encoders,feature_cols,rates,scaler,rf,xgb,lr)
        live_df["industry"]=live_df["cpv_code"].apply(cpv_to_industry)
        live_df["recommendation"]=live_df["sme_win_probability"].apply(lambda p:"Apply — good chance" if p>=0.6 else "Consider applying" if p>=0.4 else "Low chance — prepare more")
        is_live=any(live_df["url"].str.startswith("http",na=False))
        if is_live: st.success("Live data fetched from UK government portals.")
        else: st.info("Showing representative sample — live APIs restricted by IP whitelist.")
        if filter_region!="All": live_df=live_df[live_df["region"]==filter_region]
        live_df=live_df[live_df["sme_win_probability"]>=min_prob].sort_values("sme_win_probability",ascending=False).reset_index(drop=True)
        col1,col2,col3,col4,col5=st.columns(5)
        col1.metric("Total",str(len(live_df))); col2.metric("Contracts Finder",str((live_df["source"]=="Contracts Finder").sum()))
        col3.metric("Find a Tender",str((live_df["source"]=="Find a Tender").sum()))
        col4.metric("Above 50%",str((live_df["sme_win_probability"]>=0.5).sum()))
        col5.metric("Below 30%",str((live_df["sme_win_probability"]<0.3).sum()))
        st.dataframe(live_df[["source","title","buyer","value","region","industry","sme_win_probability","recommendation","deadline"]].rename(columns={"sme_win_probability":"SME Win %","source":"Source"}),use_container_width=True)
        st.download_button("Download as CSV",live_df.to_csv(index=False),"live_scored_contracts.csv","text/csv")

with tab4:
    st.subheader("CPV Code and Industry Lookup")
    col_left,col_right=st.columns(2)
    with col_left:
        st.markdown("#### CPV code to Industry")
        cpv_input=st.text_input("Enter a CPV code (e.g. 72200000)",key="cpv_in")
        if cpv_input:
            ir=cpv_to_industry(cpv_input.strip())
            if ir=="Unknown industry": st.warning("No industry found for: "+cpv_input)
            else:
                st.success(cpv_input+" belongs to: "+ir)
                sr=rates["cpv_sme_rate"].get(str(cpv_input.strip()),rates["global_sme_rate"])
                st.metric("Historical SME award rate","{:.1f}%".format(sr*100))
                related=INDUSTRY_LOOKUP.get(ir,[])
                rr2={c:rates["cpv_sme_rate"].get(c,rates["global_sme_rate"]) for c in related}
                st.dataframe(pd.DataFrame({
                    "CPV Code":related,"Industry":[ir]*len(related),"SME Rate":["{:.1f}%".format(rr2[c]*100) for c in related]
                }),use_container_width=True)
    with col_right:
        st.markdown("#### Industry to CPV codes")
        industry_input=st.selectbox("Select an industry",all_industries,key="ind_in")
        if industry_input:
            cpv_list=INDUSTRY_LOOKUP.get(industry_input,[])
            st.success(industry_input+" contains "+str(len(cpv_list))+" CPV code(s)")
            rs2={c:rates["cpv_sme_rate"].get(c,rates["global_sme_rate"]) for c in cpv_list}
            st.dataframe(pd.DataFrame({
                "CPV Code":cpv_list,"Historical SME Rate":["{:.1f}%".format(rs2[c]*100) for c in cpv_list]
            }),use_container_width=True)
            if rs2:
                bc=max(rs2,key=rs2.get); wc=min(rs2,key=rs2.get)
                st.info("Highest SME rate: CPV "+bc+" at "+"{:.1f}%".format(rs2[bc]*100))
                st.info("Lowest SME rate: CPV "+wc+" at "+"{:.1f}%".format(rs2[wc]*100))

with tab5:
    st.subheader("Historical SME Procurement Insights")
    c1,c2,c3=st.columns(3)
    c1.metric("Global SME award rate","{:.1f}%".format(rates["global_sme_rate"]*100))
    c2.metric("Best performing model",best_label)
    c3.metric("Best AUC-ROC","{:.4f}".format(best_auc) if best_auc else "N/A")
    st.divider()
    ca,cb=st.columns(2)
    with ca:
        st.markdown("**Top 10 regions by SME award rate:**")
        rdf=pd.DataFrame(list(rates["region_sme_rate"].items()),columns=["Region","SME Rate"])
        rdf=rdf.sort_values("SME Rate",ascending=False).head(10)
        rdf["SME Rate"]=rdf["SME Rate"].apply(lambda x:"{:.1f}%".format(x*100))
        st.dataframe(rdf,use_container_width=True)
    with cb:
        st.markdown("**Top 10 sectors by SME award rate:**")
        cdf=pd.DataFrame(list(rates["cpv_sme_rate"].items()),columns=["CPV Code","SME Rate"])
        cdf=cdf.sort_values("SME Rate",ascending=False).head(10)
        cdf["Industry"]=cdf["CPV Code"].apply(cpv_to_industry)
        cdf["SME Rate"]=cdf["SME Rate"].apply(lambda x:"{:.1f}%".format(x*100))
        st.dataframe(cdf[["CPV Code","Industry","SME Rate"]],use_container_width=True)
    st.divider()
    rdf_i=pd.DataFrame(list(rates["region_sme_rate"].items()),columns=["Region","Rate"])
    sdf_i=pd.DataFrame(list(rates["cpv_sme_rate"].items()),columns=["CPV","Rate"])
    gr5=rates["global_sme_rate"]
    col1,col2,col3=st.columns(3)
    col1.metric("Regions below average",str((rdf_i["Rate"]<gr5).sum())+"/"+str(len(rdf_i)))
    col2.metric("Sectors below average",str((sdf_i["Rate"]<gr5).sum())+"/"+str(len(sdf_i)))
    col3.metric("Regional inequality gap","{:.1f}%".format((rdf_i["Rate"].max()-rdf_i["Rate"].min())*100))
    st.divider()
    try:
        rdf2=pd.read_csv("model_comparison.csv").sort_values("AUC-ROC",ascending=False)
        rdf2["Recommended"]=rdf2["Model"].apply(lambda x:"Best" if best_label in x or x in best_label else "")
        st.dataframe(rdf2,use_container_width=True)
    except Exception: st.info("Model comparison table not available.")

with tab6:
    st.subheader("SME Capability and Readiness Analysis")
    st.markdown("Assess an SME using both **market-level ML prediction** and **organisational capability scoring**.")
    st.divider()
    analysis_mode=st.radio("Analysis mode",["Single SME","Batch upload (CSV)"],horizontal=True)
    if analysis_mode=="Single SME":
        st.markdown("#### Contract Details")
        col1,col2=st.columns(2)
        with col1:
            sme_cv=st.number_input("Target contract value (£)",min_value=0.0,value=75000.0,step=5000.0,key="sme_cv")
            sme_region=st.selectbox("Target region",list(encoders.get("region",{"Unknown":0}).keys()),key="sme_r")
        with col2:
            sme_inp=st.radio("CPV input",["Select CPV code","Select by industry"],key="sme_inp",horizontal=True)
            if sme_inp=="Select CPV code":
                sme_cpv=st.selectbox("CPV code",all_cpv_codes,key="sme_cpv_a"); st.caption("Industry: "+cpv_to_industry(sme_cpv))
            else:
                sme_ind=st.selectbox("Industry",all_industries,key="sme_ind")
                sme_cpv=st.selectbox("CPV code",INDUSTRY_LOOKUP.get(sme_ind,["Unknown"]),key="sme_cpv_b")
        st.divider()
        st.markdown("#### SME Organisational Characteristics")
        col1,col2,col3=st.columns(3)
        with col1:
            sme_name=st.text_input("SME name",value="My SME",key="sme_name")
            sme_turnover=st.number_input("Annual turnover (£)",min_value=0.0,value=200000.0,step=10000.0,key="sme_turn")
        with col2:
            sme_years=st.number_input("Years active",min_value=0,value=3,step=1,key="sme_years")
            sme_employees=st.number_input("Number of employees",min_value=0,value=10,step=1,key="sme_emp")
        with col3:
            sme_prior=st.number_input("Prior public contracts won",min_value=0,value=1,step=1,key="sme_prior")
            sme_certs=st.number_input("Number of certifications",min_value=0,value=1,step=1,key="sme_certs")
        if st.button("Assess SME procurement readiness",type="primary",use_container_width=True,key="sme_btn"):
            now=datetime.now(); am_s,aq_s=now.month,(now.month-1)//3+1
            row_s,cr_s,rr_s=build_row(sme_cv,am_s,aq_s,sme_region,sme_cpv,encoders,feature_cols,rates,scaler)
            p_s=get_ensemble(row_s,rf,xgb,lr)
            cap_score,breakdown=compute_sme_capability_score(sme_turnover,sme_years,sme_employees,sme_prior,sme_certs,sme_cv)
            comb=combined_score(p_s,cap_score)
            st.markdown("### Results for: "+sme_name)
            col1,col2,col3,col4=st.columns(4)
            col1.metric("ML Win Probability","{:.1f}%".format(p_s*100))
            col2.metric("Capability Score",str(cap_score)+"/100")
            col3.metric("Combined Readiness","{:.1f}%".format(comb))
            col4.metric("Sector SME rate","{:.1f}%".format(cr_s*100))
            st.progress(comb/100)
            if comb>=70: st.success(sme_name+" is well-positioned. Applying is recommended.")
            elif comb>=50: st.warning(sme_name+" has moderate readiness. Targeted improvements would help.")
            elif comb>=30: st.error(sme_name+" has significant gaps. Focused development needed.")
            else: st.error(sme_name+" is not yet ready. Fundamental barriers present.")
            st.divider()
            st.markdown("### Market-Level Barriers")
            for rname,rexpl,rdir in explain_prediction(p_s,sme_cv,sme_cpv,sme_region,rates):
                if rdir=="negative": st.error("**"+rname+":** "+rexpl)
                elif rdir=="positive": st.success("**"+rname+":** "+rexpl)
                else: st.info("**"+rname+":** "+rexpl)
            st.divider()
            st.markdown("### Organisational Capability Assessment")
            for feat,pts,level,explanation in breakdown:
                if level=="Strong": st.success("**"+feat+" ("+str(pts)+" pts):** "+explanation)
                elif level in ["Weak","Critical barrier"]: st.error("**"+feat+":** "+explanation)
                else: st.warning("**"+feat+":** "+explanation)
            st.divider()
            base_s,sugg_s=gap_analysis(sme_cv,sme_cpv,sme_region,encoders,feature_cols,rates,scaler,rf,xgb,lr)
            if sugg_s:
                st.markdown("### Strategic Recommendations")
                for category,action,new_prob,improvement,advice in sugg_s[:3]:
                    with st.expander(action+" -> "+"{:.1f}%".format(new_prob*100)+" (+"+"{:.1f}%".format(improvement*100)+")"):
                        st.markdown("**"+advice+"**")
                        ca2,cb2=st.columns(2)
                        ca2.metric("Current probability","{:.1f}%".format(base_s*100))
                        cb2.metric("If action taken","{:.1f}%".format(new_prob*100),delta="+"+"{:.1f}%".format(improvement*100))
            st.divider()
            st.markdown("### Required Certifications for Your Industry")
            industry_s=cpv_to_industry(sme_cpv)
            certs_needed=get_certs_for_industry(industry_s)
            mandatory_certs=[c for c in certs_needed if c["type"]=="Mandatory"]
            if mandatory_certs:
                st.error("MANDATORY certifications for "+industry_s+":")
                for cert in mandatory_certs:
                    st.error("- **"+cert["name"]+"** | Cost: "+cert["cost"]+" | Timeline: "+cert["timeline"])
    else:
        st.markdown("#### Batch analysis — upload a CSV of SME profiles")
        st.markdown("**Required:** sme_name, contract_value, cpv_code, region")
        st.markdown("**Recommended:** turnover, years_active, employee_count, prior_contracts, certifications")
        sample_data=pd.DataFrame({
            "sme_name":["Tech SME Ltd","Build Co Ltd","Clean Services Ltd"],
            "contract_value":[45000,250000,35000],"cpv_code":["72000000","45000000","90600000"],
            "region":["London","North West","Yorkshire and the Humber"],
            "turnover":[120000,400000,80000],"years_active":[3,7,2],
            "employee_count":[8,45,5],"prior_contracts":[2,5,0],"certifications":[1,3,0],
        })
        st.download_button("Download sample CSV template",sample_data.to_csv(index=False),"sme_template.csv","text/csv")
        uploaded=st.file_uploader("Upload your SME CSV",type=["csv"],key="sme_upload")
        if uploaded:
            sme_input_df=pd.read_csv(uploaded)
            st.success("Loaded "+str(len(sme_input_df))+" SMEs.")
            st.dataframe(sme_input_df.head(),use_container_width=True)
            if st.button("Analyse all SMEs",type="primary",use_container_width=True,key="batch_btn"):
                with st.spinner("Analysing SME profiles..."):
                    batch_results=analyse_sme_batch(sme_input_df,encoders,feature_cols,rates,scaler,rf,xgb,lr)
                st.success("Analysis complete for "+str(len(batch_results))+" SMEs")
                col1,col2,col3,col4=st.columns(4)
                col1.metric("Avg ML win probability","{:.1f}%".format(batch_results["ml_win_probability"].mean()*100))
                col2.metric("Avg capability score","{:.1f}/100".format(batch_results["capability_score"].mean()))
                col3.metric("Avg combined readiness","{:.1f}%".format(batch_results["combined_readiness"].mean()))
                col4.metric("Ready to apply",str((batch_results["combined_readiness"]>=70).sum()))
                st.dataframe(batch_results,use_container_width=True)
                st.download_button("Download full report",batch_results.to_csv(index=False),"sme_readiness_report.csv","text/csv")

with tab7:
    st.subheader("SME Procurement Barrier Dashboard")
    gr7=rates["global_sme_rate"]
    rdf7=pd.DataFrame(list(rates["region_sme_rate"].items()),columns=["Region","Rate"])
    sdf7=pd.DataFrame(list(rates["cpv_sme_rate"].items()),columns=["CPV Code","Rate"])
    sdf7["Industry"]=sdf7["CPV Code"].apply(cpv_to_industry)
    pct_below7=(rdf7["Rate"]<gr7*0.8).mean()*100
    f1_pct=(rdf7["Rate"]<0.5).mean()*100
    f2_best=sdf7.nlargest(1,"Rate").iloc[0]; f2_worst=sdf7.nsmallest(1,"Rate").iloc[0]
    f3_gap=(rdf7["Rate"].max()-rdf7["Rate"].min())*100
    col1,col2,col3=st.columns(3)
    col1.metric("Global SME award rate","{:.1f}%".format(gr7*100))
    col2.metric("Regions significantly below average","{:.0f}%".format(pct_below7))
    col3.metric("Regional inequality gap","{:.1f}%".format(f3_gap))
    st.divider()
    st.markdown("### Key Research Findings")
    col_f1,col_f2,col_f3=st.columns(3)
    with col_f1:
        st.error("Finding 1 — Structural Barrier: "+"{:.0f}%".format(f1_pct)+" of regions show SME award rates below 50%, confirming that low win probability is a rational reason for non-participation.")
    with col_f2:
        st.warning("Finding 2 — Sector Inequality: Gap between most accessible sector ("+f2_best["Industry"]+": "+"{:.1f}%".format(f2_best["Rate"]*100)+") and least ("+f2_worst["Industry"]+": "+"{:.1f}%".format(f2_worst["Rate"]*100)+") confirms structural imbalance.")
    with col_f3:
        st.info("Finding 3 — Regional Inequality: "+"{:.1f}%".format(f3_gap)+"% gap between highest and lowest regional SME rates signals uneven procurement opportunity.")
    st.divider()
    col_l,col_r=st.columns(2)
    with col_l:
        st.markdown("**Regional accessibility ranking:**")
        rdf_s7=rdf7.sort_values("Rate",ascending=False).copy()
        rdf_s7["SME Rate"]=rdf_s7["Rate"].apply(lambda x:"{:.1f}%".format(x*100))
        rdf_s7["Status"]=rdf_s7["Rate"].apply(lambda x:"Above average" if x>=gr7 else "Below average")
        rdf_s7["Barrier Level"]=rdf_s7["Rate"].apply(lambda x:"Low" if x>=gr7*1.1 else "Medium" if x>=gr7*0.9 else "High")
        st.dataframe(rdf_s7[["Region","SME Rate","Status","Barrier Level"]],use_container_width=True)
    with col_r:
        st.markdown("**Sector accessibility ranking:**")
        sdf_s7=sdf7.sort_values("Rate",ascending=False).copy()
        sdf_s7["SME Rate"]=sdf_s7["Rate"].apply(lambda x:"{:.1f}%".format(x*100))
        sdf_s7["Status"]=sdf_s7["Rate"].apply(lambda x:"Above average" if x>=gr7 else "Below average")
        sdf_s7["Barrier Level"]=sdf_s7["Rate"].apply(lambda x:"Low" if x>=gr7*1.1 else "Medium" if x>=gr7*0.9 else "High")
        st.dataframe(sdf_s7[["Industry","SME Rate","Status","Barrier Level"]].head(20),use_container_width=True)
    st.divider()
    st.markdown("### Policy Insight Report")
    gap_pct="{:.1f}%".format((f2_best["Rate"]-f2_worst["Rate"])*100)
    policy_text=(
        "AI-Driven Analysis of SME Procurement Participation Barriers\n\n"
        "1. Structural barrier confirmed: "+"{:.0f}%".format(f1_pct)+" of UK regions have SME award rates below 50%.\n"
        "2. Sector imbalance: "+gap_pct+" gap between most and least accessible sectors.\n"
        "3. Regional inequality: "+"{:.1f}%".format(f3_gap)+"% gap between regional SME award rates.\n"
        "4. Bid cost deterrence: Average win probabilities below 50% confirm non-participation is economically rational.\n"
        "5. Policy recommendation: Target highest-barrier sectors and regions with simplified procurement frameworks.\n"
    )
    st.markdown(policy_text)
    report_date=datetime.now().strftime("%Y-%m-%d")
    report_file=(
        "SME Procurement Barrier Analysis\n"
        "Generated: "+report_date+"\n\n"
        "Global SME rate: "+"{:.1f}%".format(gr7*100)+"\n"
        "Regions below average: "+"{:.0f}%".format(pct_below7)+"\n"
        "Inequality gap: "+"{:.1f}%".format(f3_gap)+"\n"
    )
    st.download_button("Download policy insight report",report_file,"policy_insight_report.txt","text/plain")
    st.divider()
    st.markdown("### Real UK SME Analysis — 199 Companies House SMEs")
    st.markdown("The following analysis is based on **199 real UK SMEs** fetched directly from Companies House across 10 sectors, fed through the trained ML model.")
    try:
        ch_pred=pd.read_csv("companies_house_predictions.csv")
        col1,col2,col3,col4=st.columns(4)
        col1.metric("Real SMEs analysed",str(len(ch_pred)))
        below_50_n=(ch_pred["ml_win_probability"]<0.5).sum()
        below_50_p="{:.1f}%".format((ch_pred["ml_win_probability"]<0.5).mean()*100)
        col2.metric("Below 50% win probability",str(below_50_n)+" ("+below_50_p+")")
        col3.metric("Average win probability","{:.1f}%".format(ch_pred["ml_win_probability"].mean()*100))
        col4.metric("Average capability score","{:.1f}/100".format(ch_pred["capability_score"].mean()))
        st.divider()
        st.markdown("### Global SHAP Feature Importance")
        if SHAP_AVAILABLE:
            st.caption("SHAP attribution is computationally intensive. Click below to compute global feature importance across a sample of real SMEs.")
            if st.button("Compute SHAP feature importance",key="shap_global_btn"):
                with st.spinner("Computing SHAP values across SME sample..."):
                    try:
                        sample_rows=[]
                        for _,rr in ch_pred.head(30).iterrows():
                            try:
                                rw,_,_=build_row(float(rr.get("contract_value",50000)),6,2,rr.get("region","Unknown"),str(rr.get("cpv_code","72000000")),encoders,feature_cols,rates,scaler)
                                sample_rows.append(rw[0])
                            except Exception:
                                pass
                        if sample_rows:
                            Xs=np.array(sample_rows)
                            expl=shap.TreeExplainer(rf)
                            svv=expl.shap_values(Xs)
                            if isinstance(svv,list): svv=svv[1]
                            mean_abs=np.abs(svv).mean(axis=0)
                            imp=sorted(zip(feature_cols,mean_abs),key=lambda x:x[1],reverse=True)
                            figs,axs=plt.subplots(figsize=(9,4))
                            names=[c[0] for c in imp[:10]][::-1]
                            vals=[c[1] for c in imp[:10]][::-1]
                            axs.barh(names,vals,color="#1a1a2e")
                            axs.set_xlabel("Mean absolute SHAP value")
                            axs.set_title("Global feature importance across real SME sample (SHAP, Random Forest)")
                            plt.tight_layout()
                            st.pyplot(figs); plt.close(figs)
                            st.caption("SHAP-based global importance across the real Companies House SME sample, complementing the impurity-based importance in the dissertation with game-theoretic attribution.")
                    except Exception as e:
                        st.warning("Global SHAP summary unavailable: "+str(e))
        else:
            st.info("This deployment runs on Python 3.14, for which SHAP is not yet available. The impurity-based feature importance (reported in the dissertation) and the domain-logic explainability remain fully available throughout the platform.")
        st.divider()
        st.markdown("### Policy Analysis Charts — Real UK SME Data")
        try:
            fig_ch=make_policy_charts(ch_pred)
            st.pyplot(fig_ch)
            plt.close(fig_ch)
        except Exception as e:
            st.warning("Chart generation error: "+str(e))
            try:
                st.image("sme_policy_analysis.png",caption="AI-Driven SME Procurement Accessibility Analysis",use_container_width=True)
            except Exception:
                st.info("Charts not available — run the SME data pipeline in the notebook to generate them.")
        st.divider()
        col_l2,col_r2=st.columns(2)
        with col_l2:
            st.markdown("**Win probability by sector (real SMEs):**")
            sec_df=ch_pred.groupby("sector")["ml_win_probability"].mean().round(3).reset_index()
            sec_df.columns=["Sector","Avg Win Probability"]
            sec_df=sec_df.sort_values("Avg Win Probability")
            sec_df["Avg Win Probability"]=sec_df["Avg Win Probability"].apply(lambda x:"{:.1f}%".format(x*100))
            st.dataframe(sec_df,use_container_width=True)
        with col_r2:
            st.markdown("**Win probability by region (real SMEs):**")
            reg_df=ch_pred.groupby("region")["ml_win_probability"].mean().round(3).reset_index()
            reg_df.columns=["Region","Avg Win Probability"]
            reg_df=reg_df.sort_values("Avg Win Probability")
            reg_df["Avg Win Probability"]=reg_df["Avg Win Probability"].apply(lambda x:"{:.1f}%".format(x*100))
            st.dataframe(reg_df,use_container_width=True)
        st.divider()
        st.markdown("**Readiness band breakdown:**")
        bands_data={
            "Band":["Not ready (<30%)","Needs work (30-50%)","Moderate (50-70%)","Ready (>70%)"],
            "Count":[
                int((ch_pred["combined_readiness"]<30).sum()),
                int(((ch_pred["combined_readiness"]>=30)&(ch_pred["combined_readiness"]<50)).sum()),
                int(((ch_pred["combined_readiness"]>=50)&(ch_pred["combined_readiness"]<70)).sum()),
                int((ch_pred["combined_readiness"]>=70).sum()),
            ]
        }
        st.dataframe(pd.DataFrame(bands_data),use_container_width=True)
        st.divider()
        st.markdown("**Key Research Finding:**")
        below_50_ch=(ch_pred["ml_win_probability"]<0.5).mean()*100
        st.error("When the trained ML model was applied to "+str(len(ch_pred))+" real UK SMEs from Companies House, "+"{:.1f}%".format(below_50_ch)+"% were predicted to have below 50% win probability. This finding is consistent with the rational non-participation hypothesis: structurally low predicted win probabilities may help explain reduced SME participation. The model predicts historical outcome probabilities and does not directly observe SME bidding decisions.")
        st.divider()
        st.markdown("**Full SME predictions dataset:**")
        st.dataframe(ch_pred[["sme_name","sector","region","ml_win_probability","capability_score","combined_readiness","prediction","recommendation"]].head(50),use_container_width=True)
        st.download_button("Download full Companies House predictions",ch_pred.to_csv(index=False),"companies_house_predictions.csv","text/csv")
    except Exception: st.info("Companies House predictions not yet available. Run the SME data pipeline in the notebook first.")

with tab8:
    st.subheader("Certifications Guide — What Your SME Needs to Win UK Public Contracts")
    st.markdown("Every UK public sector contract has certification requirements. Find exactly what your SME needs based on industry or CPV code.")
    st.divider()
    cert_mode=st.radio("Search by",["Industry","CPV Code"],horizontal=True)
    if cert_mode=="CPV Code":
        cert_cpv=st.text_input("Enter CPV code (e.g. 72000000)",key="cert_cpv")
        if cert_cpv:
            industry_c,certs_c=get_certs_for_cpv(cert_cpv.strip())
            st.success("CPV "+cert_cpv+" maps to: **"+industry_c+"**")
        else:
            industry_c=None; certs_c=[]
    else:
        industry_c=st.selectbox("Select your industry",all_industries,key="cert_ind")
        certs_c=get_certs_for_industry(industry_c)
    if certs_c:
        mandatory=[c for c in certs_c if c["type"]=="Mandatory"]
        recommended=[c for c in certs_c if c["type"]=="Strongly recommended"]
        specific=[c for c in certs_c if c["type"] not in ["Mandatory","Strongly recommended"]]
        col1,col2,col3=st.columns(3)
        col1.metric("Mandatory certifications",str(len(mandatory)))
        col2.metric("Strongly recommended",str(len(recommended)))
        col3.metric("Sector-specific",str(len(specific)))
        st.divider()
        if mandatory:
            st.markdown("### MANDATORY — Required to be eligible to bid")
            for cert in mandatory:
                with st.expander("MANDATORY: "+cert["name"]+" | "+cert["cost"]+" | "+cert["timeline"]):
                    col_a,col_b=st.columns(2)
                    with col_a:
                        st.markdown("**Issuing body:** "+cert["issuer"])
                        st.markdown("**Estimated cost:** "+cert["cost"])
                        st.markdown("**Typical timeline:** "+cert["timeline"])
                    with col_b:
                        st.markdown("**Required by:** "+cert["required_by"])
                        st.markdown("**Why important:** "+cert["why"])
                    if cert.get("url"): st.markdown("**More info:** "+cert["url"])
        if recommended:
            st.divider()
            st.markdown("### STRONGLY RECOMMENDED — Significantly improves bid scores")
            for cert in recommended:
                with st.expander("RECOMMENDED: "+cert["name"]+" | "+cert["cost"]+" | "+cert["timeline"]):
                    col_a,col_b=st.columns(2)
                    with col_a:
                        st.markdown("**Issuing body:** "+cert["issuer"])
                        st.markdown("**Estimated cost:** "+cert["cost"])
                        st.markdown("**Typical timeline:** "+cert["timeline"])
                    with col_b:
                        st.markdown("**Required by:** "+cert["required_by"])
                        st.markdown("**Why important:** "+cert["why"])
                    if cert.get("url"): st.markdown("**More info:** "+cert["url"])
        if specific:
            st.divider()
            st.markdown("### SECTOR-SPECIFIC — Required for certain contract types")
            for cert in specific:
                with st.expander(cert["type"]+": "+cert["name"]+" | "+cert["cost"]+" | "+cert["timeline"]):
                    col_a,col_b=st.columns(2)
                    with col_a:
                        st.markdown("**Issuing body:** "+cert["issuer"])
                        st.markdown("**Estimated cost:** "+cert["cost"])
                        st.markdown("**Typical timeline:** "+cert["timeline"])
                    with col_b:
                        st.markdown("**Required by:** "+cert["required_by"])
                        st.markdown("**Why important:** "+cert["why"])
                    if cert.get("url"): st.markdown("**More info:** "+cert["url"])
        st.divider()
        st.markdown("### Download Certification Checklist")
        cert_rows=[]
        for cert in certs_c:
            cert_rows.append({"Certification":cert["name"],"Type":cert["type"],"Issuer":cert["issuer"],
                "Cost":cert["cost"],"Timeline":cert["timeline"],"Required By":cert["required_by"],"Why Important":cert["why"]})
        cert_df=pd.DataFrame(cert_rows)
        st.dataframe(cert_df,use_container_width=True)
        st.download_button("Download certification checklist as CSV",cert_df.to_csv(index=False),"certification_checklist.csv","text/csv")

with st.sidebar:
    st.header("SME Procurement Intelligence")
    st.markdown("**Platform capabilities:**")
    st.markdown("- Win probability prediction")
    st.markdown("- Barrier and gap analysis")
    st.markdown("- Live contract feed (dual API)")
    st.markdown("- CPV code to industry lookup")
    st.markdown("- Historical procurement insights")
    st.markdown("- SME capability and readiness assessment")
    st.markdown("- Procurement barrier dashboard with real SME data")
    st.markdown("- Certifications guide by industry")
    st.divider()
    if best_auc:
        st.success("Best model: "+best_label+"\nAUC-ROC: "+"{:.4f}".format(best_auc))
    st.divider()
    st.markdown("**Research framing:**")
    st.caption("Explainable AI for understanding SME participation barriers in UK public procurement")
    st.divider()
    st.markdown("**Data sources:**")
    st.markdown("- UK Contracts Finder API")
    st.markdown("- Find a Tender API")
    st.markdown("- Companies House API (199 real SMEs)")
    st.markdown("*Live data refreshes hourly*")
    st.divider()
    st.caption("Research prototype — not for operational use.")
