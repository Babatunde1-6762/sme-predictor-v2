import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import requests
from datetime import datetime, timedelta

CPV_LOOKUP = {
    '45000000':'Construction','45100000':'Site preparation work',
    '45200000':'Building construction','45300000':'Building installation works',
    '45400000':'Building completion work','70000000':'Real estate services',
    '71000000':'Architectural and engineering services',
    '30000000':'IT equipment and supplies','48000000':'Software and IT systems',
    '72000000':'IT services','72100000':'IT consultancy',
    '72200000':'Software programming services','72300000':'Data services',
    '72400000':'Internet services','72500000':'Computer-related services',
    '72600000':'IT support and consultancy','64000000':'Postal and telecommunications',
    '32000000':'Radio, television and communications equipment',
    '33000000':'Medical equipment and supplies','85000000':'Health and social work services',
    '85100000':'Health services','85110000':'Hospital services',
    '85120000':'Medical practice services','85200000':'Veterinary services',
    '85300000':'Social work services','85320000':'Social services',
    '80000000':'Education and training services','80100000':'Primary education services',
    '80200000':'Secondary education services','80300000':'Higher education services',
    '80400000':'Adult and other education services','80500000':'Training services',
    '60000000':'Transport services','60100000':'Road transport services',
    '60200000':'Rail transport services','60400000':'Air transport services',
    '63000000':'Supporting transport services',
    '66000000':'Financial and insurance services',
    '73000000':'Research and development services',
    '79100000':'Legal services','79200000':'Accounting services',
    '79400000':'Business and management consultancy',
    '79600000':'Recruitment services','79700000':'Investigation and security services',
    '50000000':'Repair and maintenance services',
    '55000000':'Hotel, restaurant and catering services',
    '90000000':'Sewage, refuse, cleaning and environmental services',
    '90600000':'Cleaning services','90700000':'Environmental services',
    '09000000':'Petroleum products, fuel and electricity',
    '09300000':'Electricity, heating, solar and nuclear energy',
    '65000000':'Public utilities',
    '03000000':'Agricultural, farming and fishing products',
    '15000000':'Food, beverages, tobacco and related products',
    '24000000':'Chemical products','39000000':'Furniture and household appliances',
    '35000000':'Security and fire-fighting equipment',
    '98000000':'Other community and social services',
}

INDUSTRY_LOOKUP = {}
for _code, _ind in CPV_LOOKUP.items():
    INDUSTRY_LOOKUP.setdefault(_ind, []).append(_code)

def cpv_to_industry(cpv):
    s = str(cpv).split('.')[0].strip()
    if s in CPV_LOOKUP: return CPV_LOOKUP[s]
    if s[:6]+'00' in CPV_LOOKUP: return CPV_LOOKUP[s[:6]+'00']
    if s[:4]+'0000' in CPV_LOOKUP: return CPV_LOOKUP[s[:4]+'0000']
    if s[:2]+'000000' in CPV_LOOKUP: return CPV_LOOKUP[s[:2]+'000000']
    return 'Unknown industry'

@st.cache_resource
def load_artefacts():
    rf     = joblib.load('sme_rf_model.pkl')
    xgb    = joblib.load('sme_xgb_model.pkl')
    lr     = joblib.load('sme_lr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    enc    = json.load(open('encoders.json'))
    feats  = json.load(open('feature_cols.json'))
    rates  = json.load(open('historical_rates.json'))
    try:
        results   = pd.read_csv('model_comparison.csv')
        best_row  = results.loc[results['AUC-ROC'].idxmax()]
        best_name = best_row['Model']
        best_auc  = float(best_row['AUC-ROC'])
    except Exception:
        best_name = 'Random Forest'
        best_auc  = None
    return rf, xgb, lr, scaler, enc, feats, rates, best_name, best_auc

@st.cache_data(ttl=3600)
def fetch_live_contracts(days_back=7, max_results=50):
    date_from     = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    date_to       = datetime.now().strftime('%Y-%m-%d')
    all_contracts = []
    # SOURCE 1: Contracts Finder
    try:
        cf_url     = 'https://www.contractsfinder.service.gov.uk/Published/Notices/PublicSearch/json'
        cf_payload = {'searchCriteria': {'page':1,'publishedFrom':date_from,'publishedTo':date_to,'size':max_results}}
        cf_resp    = requests.post(cf_url, json=cf_payload, timeout=15)
        if cf_resp.status_code == 200:
            for n in cf_resp.json().get('results', []):
                item = n.get('item', {})
                all_contracts.append({
                    'source'   : 'Contracts Finder',
                    'title'    : item.get('title', 'Unknown')[:80],
                    'buyer'    : item.get('organizationName', 'Unknown')[:50],
                    'value'    : item.get('value', {}).get('amount', 0) if item.get('value') else 0,
                    'cpv_code' : item.get('cpvCodes', [{}])[0].get('code', 'Unknown') if item.get('cpvCodes') else 'Unknown',
                    'region'   : item.get('deliveryLocations', [{}])[0].get('region', 'Unknown') if item.get('deliveryLocations') else 'Unknown',
                    'deadline' : item.get('tenderDeadline', 'Unknown'),
                    'published': item.get('publishedAt', 'Unknown'),
                    'url'      : n.get('publishedUrl', ''),
                })
    except Exception:
        pass
    # SOURCE 2: Find a Tender
    try:
        fat_url    = 'https://www.find-tender.service.gov.uk/api/1.0/ocds/notices/list'
        fat_params = {'publishedFrom':date_from,'publishedTo':date_to,'limit':max_results}
        fat_resp   = requests.get(fat_url, params=fat_params, timeout=15, headers={'Accept':'application/json'})
        if fat_resp.status_code == 200:
            records = fat_resp.json().get('records', fat_resp.json().get('notices', []))
            for record in records:
                release = record.get('compiledRelease', record.get('releases', [{}])[0] if record.get('releases') else record)
                tender  = release.get('tender', {})
                buyer   = release.get('buyer', {}).get('name', 'Unknown')[:50]
                items   = tender.get('items', [{}])
                cpv_raw = items[0].get('classification', {}).get('id', 'Unknown') if items else 'Unknown'
                delivery= tender.get('deliveryLocations', [{}])
                region  = delivery[0].get('region', 'Unknown') if delivery else 'Unknown'
                val_obj = tender.get('value', {})
                value   = val_obj.get('amount', 0) if val_obj else 0
                all_contracts.append({
                    'source'   : 'Find a Tender',
                    'title'    : tender.get('title', 'Unknown')[:80],
                    'buyer'    : buyer,
                    'value'    : value,
                    'cpv_code' : cpv_raw,
                    'region'   : region,
                    'deadline' : tender.get('tenderPeriod', {}).get('endDate', 'Unknown'),
                    'published': release.get('date', 'Unknown'),
                    'url'      : release.get('ocid', ''),
                })
    except Exception:
        pass
    if not all_contracts:
        return pd.DataFrame()
    df = pd.DataFrame(all_contracts)
    df = df.drop_duplicates(subset=['title','buyer'], keep='first')
    df = df.sort_values('published', ascending=False).reset_index(drop=True)
    return df

def build_row(cv, am, aq, region, cpv, encoders, feature_cols, rates, scaler):
    log_cv = np.log1p(cv)
    vbnum  = 0 if cv<10000 else 1 if cv<50000 else 2 if cv<100000 else 3 if cv<500000 else 4
    is_qe  = int(aq in [1,4])
    is_hv  = int(cv > 100000)
    cr     = rates['cpv_sme_rate'].get(str(cpv),       rates['global_sme_rate'])
    rr     = rates['region_sme_rate'].get(str(region), rates['global_sme_rate'])
    r_enc  = encoders.get('region',   {}).get(str(region), 0)
    c_enc  = encoders.get('cpv_code', {}).get(str(cpv), 0)
    d = {
        'log_contract_value': log_cv,
        'value_band_num':     vbnum,
        'award_month':        am,
        'award_quarter':      aq,
        'is_quarter_end':     is_qe,
        'is_high_value':      is_hv,
        'buyer_sme_rate':     rates['global_sme_rate'],
        'cpv_sme_rate':       cr,
        'region_sme_rate':    rr,
        'value_band_enc':     0,
        'region_enc':         r_enc,
        'cpv_code_enc':       c_enc,
    }
    row = pd.DataFrame([{c: d.get(c,0) for c in feature_cols}])
    return scaler.transform(row.values), cr, rr

def get_ensemble(row_scaled, rf, xgb, lr):
    return rf.predict_proba(row_scaled)[0][1]*0.5 + xgb.predict_proba(row_scaled)[0][1]*0.35 + lr.predict_proba(row_scaled)[0][1]*0.15

def score_contracts(df, encoders, feature_cols, rates, scaler, rf, xgb, lr):
    now    = datetime.now()
    am, aq = now.month, (now.month-1)//3+1
    probs  = []
    for _, row in df.iterrows():
        try:
            cv  = float(row.get('value', 50000) or 50000)
            cpv = str(row.get('cpv_code', 'Unknown'))
            reg = str(row.get('region', 'Unknown'))
            r, _, _ = build_row(cv, am, aq, reg, cpv, encoders, feature_cols, rates, scaler)
            probs.append(round(get_ensemble(r, rf, xgb, lr), 3))
        except Exception:
            probs.append(rates['global_sme_rate'])
    return probs

def gap_analysis(cv, cpv, region, encoders, feature_cols, rates, scaler, rf, xgb, lr):
    now    = datetime.now()
    am, aq = now.month, (now.month-1)//3+1
    def gp(cv_, cpv_, reg_):
        r, _, _ = build_row(cv_, am, aq, reg_, cpv_, encoders, feature_cols, rates, scaler)
        return get_ensemble(r, rf, xgb, lr)
    base = gp(cv, cpv, region)
    sugg = []
    for test_cv, label in [(cv*0.5, f'Target 50% smaller contract (£{cv*0.5:,.0f})'),
                            (cv*0.25, f'Target 75% smaller contract (£{cv*0.25:,.0f})'),
                            (25000,   f'Target £25,000 contract')]:
        p = gp(test_cv, cpv, region)
        if p > base:
            sugg.append(('Contract size', label, round(p,3), round(p-base,3), 'Smaller contracts have higher SME award rates — build track record first'))
    best_reg = max(rates['region_sme_rate'], key=rates['region_sme_rate'].get)
    p_br = gp(cv, cpv, best_reg)
    if p_br > base:
        sugg.append(('Region', f'Target {best_reg}', round(p_br,3), round(p_br-base,3), f'{best_reg} has historically higher SME award rates'))
    best_cpv = max(rates['cpv_sme_rate'], key=rates['cpv_sme_rate'].get)
    p_bc = gp(cv, best_cpv, region)
    if p_bc > base:
        sugg.append(('Sector', f'Consider {cpv_to_industry(best_cpv)}', round(p_bc,3), round(p_bc-base,3), f'{cpv_to_industry(best_cpv)} sector has higher SME win rates historically'))
    if aq not in [1,4]:
        p_q = gp(cv, cpv, region)
        if p_q > base:
            sugg.append(('Timing', 'Apply in Q1 or Q4', round(p_q,3), round(p_q-base,3), 'Quarter-end periods show different award patterns'))
    return base, sorted(sugg, key=lambda x: x[3], reverse=True)

rf, xgb, lr, scaler, encoders, feature_cols, rates, best_name, best_auc = load_artefacts()
all_cpv_codes  = list(encoders.get('cpv_code', {'Unknown':0}).keys())
all_industries = sorted(set(CPV_LOOKUP.values()))

MODEL_MAP = {
    'Random Forest':                 ('Random Forest',       rf),
    'XGBoost':                       ('XGBoost',             xgb),
    'Logistic Regression':           ('Logistic Regression', lr),
    'Logistic Regression (baseline)':('Logistic Regression', lr),
}
best_label, _ = MODEL_MAP.get(best_name, ('Random Forest', rf))
model_options = []
for opt in ['Random Forest','XGBoost','Logistic Regression']:
    if opt == best_label:
        auc_str = f' — AUC {best_auc:.3f}' if best_auc else ''
        model_options.append(f'{opt} ⭐ Recommended{auc_str}')
    else:
        model_options.append(opt)
default_index = next((i for i,o in enumerate(model_options) if '⭐' in o), 0)

st.set_page_config(page_title='SME Procurement Intelligence', page_icon='🏆', layout='wide')
st.title('🏆 SME Procurement Intelligence Platform')
st.markdown('A live platform helping UK SMEs understand, predict, and improve their chances of winning public procurement contracts.')
if best_auc:
    st.success(f'🏆 Best model: **{best_label}**  |  AUC-ROC: **{best_auc:.4f}**  — set as default')
st.divider()

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    '🔮 Win Probability',
    '📈 Gap Analysis',
    '📡 Live Contracts',
    '🏭 CPV Lookup',
    '📊 Historical'
])

# ── TAB 1: WIN PROBABILITY ───────────────────────────────────────────
with tab1:
    st.subheader('🔮 Will this SME win the contract?')
    st.markdown('Enter the contract details to get an instant SME win probability prediction.')
    c1, c2 = st.columns(2)
    with c1:
        cv1 = st.number_input('Contract value (£)', min_value=0.0, value=50000.0, step=1000.0, key='cv1')
        am1 = st.selectbox('Award month', list(range(1,13)),
                           format_func=lambda m:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1],
                           key='am1')
        aq1 = st.selectbox('Award quarter', [1,2,3,4], format_func=lambda q:f'Q{q}', key='aq1')
        mc  = st.selectbox('Model', model_options, index=default_index, key='mc1')
    with c2:
        r1   = st.selectbox('Region', list(encoders.get('region',{'Unknown':0}).keys()), key='r1')
        inp1 = st.radio('CPV input method', ['Select CPV code','Select by industry'], key='im1', horizontal=True)
        if inp1 == 'Select CPV code':
            cp1 = st.selectbox('CPV code', all_cpv_codes, key='cp1a')
            st.caption(f'Industry: **{cpv_to_industry(cp1)}**')
        else:
            ind1 = st.selectbox('Industry', all_industries, key='ind1')
            cp1  = st.selectbox('CPV code (from industry)', INDUSTRY_LOOKUP.get(ind1,['Unknown']), key='cp1b')
    if st.button('🔮 Predict win probability', type='primary', use_container_width=True, key='btn1'):
        row1, cr1, rr1 = build_row(cv1, am1, aq1, r1, cp1, encoders, feature_cols, rates, scaler)
        p_rf1  = rf.predict_proba(row1)[0][1]
        p_xgb1 = xgb.predict_proba(row1)[0][1]
        p_lr1  = lr.predict_proba(row1)[0][1]
        p_ens1 = p_rf1*0.5 + p_xgb1*0.35 + p_lr1*0.15
        pred1  = int(p_ens1 >= 0.5)
        a,b,c,d = st.columns(4)
        a.metric('Prediction',      'SME likely ✅' if pred1 else 'SME unlikely ⚠️')
        b.metric('Win Probability', f'{p_ens1*100:.1f}%')
        c.metric('Sector SME rate', f'{cr1*100:.1f}%')
        d.metric('Region SME rate', f'{rr1*100:.1f}%')
        st.progress(float(p_ens1))
        if pred1:
            st.success('✅ This SME has a good chance. Applying is strongly recommended.')
        elif p_ens1 >= 0.35:
            st.warning('⚠️ Borderline probability. Strengthening the bid could tip the balance. Check the Gap Analysis tab.')
        else:
            st.error('❌ Low win probability. This is why many SMEs choose not to apply. See Gap Analysis tab for what to improve.')
        st.divider()
        st.markdown('**Individual model contributions:**')
        col1,col2,col3,col4 = st.columns(4)
        col1.metric('Random Forest (50%)',   f'{p_rf1*100:.1f}%')
        col2.metric('XGBoost (35%)',          f'{p_xgb1*100:.1f}%')
        col3.metric('Logistic Reg (15%)',     f'{p_lr1*100:.1f}%')
        col4.metric('Ensemble (final)',       f'{p_ens1*100:.1f}%')
        st.info(f'Industry: **{cpv_to_industry(cp1)}**  |  Sector SME rate: {cr1*100:.1f}%  |  Region SME rate: {rr1*100:.1f}%')

# ── TAB 2: GAP ANALYSIS ─────────────────────────────────────────────
with tab2:
    st.subheader('📈 What does this SME need to improve to win?')
    st.markdown('The model identifies specific changes that would most improve this SME\'s win probability — and explains why SMEs are reluctant to apply.')
    c1, c2 = st.columns(2)
    with c1:
        cv2 = st.number_input('Contract value (£)', min_value=0.0, value=200000.0, step=1000.0, key='cv2')
        r2  = st.selectbox('Region', list(encoders.get('region',{'Unknown':0}).keys()), key='r2')
    with c2:
        inp2 = st.radio('CPV input method', ['Select CPV code','Select by industry'], key='im2', horizontal=True)
        if inp2 == 'Select CPV code':
            cp2 = st.selectbox('CPV code', all_cpv_codes, key='cp2a')
            st.caption(f'Industry: **{cpv_to_industry(cp2)}**')
        else:
            ind2 = st.selectbox('Industry', all_industries, key='ind2')
            cp2  = st.selectbox('CPV code (from industry)', INDUSTRY_LOOKUP.get(ind2,['Unknown']), key='cp2b')
    if st.button('📊 Run gap analysis', type='primary', use_container_width=True, key='btn2'):
        base_prob, suggestions = gap_analysis(cv2, cp2, r2, encoders, feature_cols, rates, scaler, rf, xgb, lr)
        st.metric('Current win probability', f'{base_prob*100:.1f}%')
        st.progress(float(base_prob))
        if base_prob < 0.3:
            st.error(f'❌ Only **{base_prob*100:.1f}%** probability. The model confirms why SMEs avoid this contract type — the odds are too low to justify bid costs.')
        elif base_prob < 0.5:
            st.warning(f'⚠️ **{base_prob*100:.1f}%** probability — below threshold. SMEs may hesitate because bid preparation costs outweigh the low win chance.')
        else:
            st.success(f'✅ **{base_prob*100:.1f}%** probability — above threshold. This SME should be encouraged to apply.')
        st.divider()
        if suggestions:
            st.markdown('### 🎯 Steps to improve win probability:')
            for category, action, new_prob, improvement, advice in suggestions[:4]:
                with st.expander(f'💡 {action}  →  {new_prob*100:.1f}%  (+{improvement*100:.1f}%)'):
                    st.markdown(f'**Category:** {category}')
                    st.markdown(f'**Action:** {action}')
                    st.markdown(f'**Why this helps:** {advice}')
                    st.progress(float(new_prob))
                    col_a, col_b = st.columns(2)
                    col_a.metric('Current probability', f'{base_prob*100:.1f}%')
                    col_b.metric('Improved probability', f'{new_prob*100:.1f}%', delta=f'+{improvement*100:.1f}%')
        else:
            st.info('No clear improvement paths identified. This SME may already be well positioned.')
        st.divider()
        st.markdown('### 📝 Why are SMEs not applying? — Model evidence')
        st.markdown(f'For a **{cpv_to_industry(cp2)}** contract worth **£{cv2:,.0f}** in **{r2}**:')
        if base_prob < 0.3:
            st.error('With less than 30% predicted probability, the model confirms that SME reluctance is rational and data-driven. The financial risk of bid preparation outweighs the expected return at this probability level.')
        elif base_prob < 0.5:
            st.warning('At 30-50% probability, the decision is borderline. SMEs are making a rational economic calculation — bid costs are high and the probability of winning does not justify the investment without strategic adjustments.')
        else:
            st.success('Above 50% probability. SMEs in this category should be actively encouraged to apply. The model suggests the market conditions are favourable.')

# ── TAB 3: LIVE CONTRACTS ───────────────────────────────────────────
with tab3:
    st.subheader('📡 Live UK Government Contracts')
    st.markdown('Real-time contracts from **Contracts Finder** and **Find a Tender** — both sources running concurrently. Each contract is scored for SME win probability. Data refreshes every hour.')
    c1, c2 = st.columns(2)
    with c1:
        days_back    = st.slider('Days to look back', 1, 30, 7)
        max_res      = st.selectbox('Max contracts per source', [25,50,100], index=1)
    with c2:
        filter_region = st.selectbox('Filter by region (optional)', ['All'] + list(encoders.get('region',{'Unknown':0}).keys()))
        min_prob      = st.slider('Minimum SME win probability', 0.0, 1.0, 0.0, 0.05)
    if st.button('📡 Fetch and score live contracts', type='primary', use_container_width=True):
        with st.spinner('Fetching from Contracts Finder and Find a Tender simultaneously...'):
            live_df = fetch_live_contracts(days_back=days_back, max_results=max_res)
        if len(live_df) == 0:
            st.error('Could not fetch contracts. Both APIs may be temporarily unavailable. Please try again in a few minutes.')
        else:
            with st.spinner('Scoring all contracts for SME win probability...'):
                live_df['sme_win_probability'] = score_contracts(live_df, encoders, feature_cols, rates, scaler, rf, xgb, lr)
            live_df['industry']       = live_df['cpv_code'].apply(cpv_to_industry)
            live_df['recommendation'] = live_df['sme_win_probability'].apply(
                lambda p: '✅ Apply — good chance' if p >= 0.6 else
                          '🤔 Consider applying'    if p >= 0.4 else
                          '❌ Low chance — prepare more'
            )
            if filter_region != 'All':
                live_df = live_df[live_df['region'] == filter_region]
            live_df = live_df[live_df['sme_win_probability'] >= min_prob]
            live_df = live_df.sort_values('sme_win_probability', ascending=False).reset_index(drop=True)
            col1,col2,col3,col4,col5 = st.columns(5)
            col1.metric('Total contracts', str(len(live_df)))
            col2.metric('Contracts Finder', str((live_df['source']=='Contracts Finder').sum()))
            col3.metric('Find a Tender',    str((live_df['source']=='Find a Tender').sum()))
            col4.metric('Above 50% chance', str((live_df['sme_win_probability']>=0.5).sum()))
            col5.metric('Below 30% chance', str((live_df['sme_win_probability']< 0.3).sum()))
            st.dataframe(
                live_df[['source','title','buyer','value','region','industry','sme_win_probability','recommendation','deadline']].rename(columns={'sme_win_probability':'SME Win %','source':'Source'}),
                use_container_width=True
            )
            csv = live_df.to_csv(index=False)
            st.download_button('📥 Download as CSV', csv, 'live_scored_contracts.csv', 'text/csv')

# ── TAB 4: CPV LOOKUP ────────────────────────────────────────────────
with tab4:
    st.subheader('🏭 CPV Code ↔ Industry Lookup')
    st.markdown('Find the industry for any CPV code, or find all CPV codes for any industry.')
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('#### CPV code → Industry')
        cpv_input = st.text_input('Enter a CPV code (e.g. 72200000)', key='cpv_in')
        if cpv_input:
            industry_result = cpv_to_industry(cpv_input.strip())
            if industry_result == 'Unknown industry':
                st.warning(f'No industry found for: {cpv_input}')
            else:
                st.success(f'**{cpv_input}** → **{industry_result}**')
                sme_rate = rates['cpv_sme_rate'].get(str(cpv_input.strip()), rates['global_sme_rate'])
                st.metric('Historical SME award rate', f'{sme_rate*100:.1f}%')
                related = INDUSTRY_LOOKUP.get(industry_result, [])
                st.markdown(f'Other CPV codes in **{industry_result}**:')
                st.dataframe(pd.DataFrame({'CPV Code':related,'Industry':[industry_result]*len(related)}), use_container_width=True)
    with col_right:
        st.markdown('#### Industry → CPV codes')
        industry_input = st.selectbox('Select an industry', all_industries, key='ind_in')
        if industry_input:
            cpv_list = INDUSTRY_LOOKUP.get(industry_input, [])
            st.success(f'**{industry_input}** — {len(cpv_list)} CPV code(s)')
            rs  = {c: rates['cpv_sme_rate'].get(c, rates['global_sme_rate']) for c in cpv_list}
            dfs = pd.DataFrame({'CPV Code':cpv_list, 'Historical SME Rate':[f"{rs[c]*100:.1f}%" for c in cpv_list]})
            st.dataframe(dfs, use_container_width=True)
            if rs:
                bc = max(rs, key=rs.get)
                wc = min(rs, key=rs.get)
                st.info(f'Highest SME rate in this sector: CPV {bc} at {rs[bc]*100:.1f}%')
                st.info(f'Lowest SME rate in this sector:  CPV {wc} at {rs[wc]*100:.1f}%')

# ── TAB 5: HISTORICAL ────────────────────────────────────────────────
with tab5:
    st.subheader('📊 Historical SME Procurement Insights')
    c1,c2,c3 = st.columns(3)
    c1.metric('Global SME award rate', f"{rates['global_sme_rate']*100:.1f}%")
    c2.metric('Best performing model', best_label)
    c3.metric('Best AUC-ROC', f'{best_auc:.4f}' if best_auc else 'N/A')
    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('**Top 10 regions by SME award rate:**')
        rdf = pd.DataFrame(list(rates['region_sme_rate'].items()), columns=['Region','SME Rate'])
        rdf = rdf.sort_values('SME Rate', ascending=False).head(10)
        rdf['SME Rate'] = rdf['SME Rate'].apply(lambda x: f'{x*100:.1f}%')
        st.dataframe(rdf, use_container_width=True)
    with cb:
        st.markdown('**Top 10 sectors by SME award rate:**')
        cdf = pd.DataFrame(list(rates['cpv_sme_rate'].items()), columns=['CPV Code','SME Rate'])
        cdf = cdf.sort_values('SME Rate', ascending=False).head(10)
        cdf['Industry'] = cdf['CPV Code'].apply(cpv_to_industry)
        cdf['SME Rate'] = cdf['SME Rate'].apply(lambda x: f'{x*100:.1f}%')
        st.dataframe(cdf[['CPV Code','Industry','SME Rate']], use_container_width=True)
    st.divider()
    st.markdown('**All model performance comparison:**')
    try:
        rdf2 = pd.read_csv('model_comparison.csv').sort_values('AUC-ROC', ascending=False)
        rdf2['Recommended'] = rdf2['Model'].apply(lambda x: '⭐' if best_label in x or x in best_label else '')
        st.dataframe(rdf2, use_container_width=True)
    except Exception:
        st.info('Model comparison table not available.')

with st.sidebar:
    st.header('🏆 SME Procurement Intelligence')
    st.markdown('**What this platform does:**')
    st.markdown('🔮 Predicts if an SME will win a contract')
    st.markdown('📈 Identifies what to improve to win')
    st.markdown('📡 Shows live contracts from both UK portals')
    st.markdown('🏭 Maps CPV codes to industries')
    st.markdown('📊 Reveals historical SME award patterns')
    st.divider()
    if best_auc:
        st.success(f'⭐ Best model: **{best_label}**\nAUC-ROC: {best_auc:.4f}')
    st.divider()
    st.markdown('**Models used:**')
    st.markdown('- Random Forest')
    st.markdown('- XGBoost')
    st.markdown('- Logistic Regression')
    st.markdown('- Weighted ensemble (deep learning proxy)')
    st.divider()
    st.markdown('**Live data sources:**')
    st.markdown('- UK Contracts Finder API')
    st.markdown('- Find a Tender API')
    st.markdown('*Refreshes every hour automatically*')
    st.divider()
    st.caption('Research prototype — not for operational use.')