import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

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
        results = pd.read_csv('model_comparison.csv')
        best_row  = results.loc[results['AUC-ROC'].idxmax()]
        best_name = best_row['Model']
        best_auc  = best_row['AUC-ROC']
    except Exception:
        best_name = 'Random Forest'
        best_auc  = None
    return rf, xgb, lr, scaler, enc, feats, rates, best_name, best_auc

@st.cache_resource
def load_tflite():
    try:
        try:
            import tflite_runtime.interpreter as tflite
            interp = tflite.Interpreter(model_path='sme_tf_lite.tflite')
        except ImportError:
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path='sme_tf_lite.tflite')
        interp.allocate_tensors()
        return interp
    except Exception:
        return None

def predict_tflite(interp, x):
    inp = interp.get_input_details()
    out = interp.get_output_details()
    interp.set_tensor(inp[0]['index'], x.astype(np.float32))
    interp.invoke()
    return float(interp.get_tensor(out[0]['index'])[0][0])

def build_row(cv, am, aq, region, cpv, vband, buyer, encoders, feature_cols, rates, scaler):
    log_cv = np.log1p(cv)
    vbnum  = 0 if cv<10000 else 1 if cv<50000 else 2 if cv<100000 else 3 if cv<500000 else 4
    is_qe  = int(aq in [1,4])
    is_hv  = int(cv > 100000)
    br     = rates['buyer_sme_rate'].get(str(buyer),   rates['global_sme_rate'])
    cr     = rates['cpv_sme_rate'].get(str(cpv),       rates['global_sme_rate'])
    rr     = rates['region_sme_rate'].get(str(region), rates['global_sme_rate'])
    vb_enc = encoders.get('value_band', {}).get(str(vband), 0)
    r_enc  = encoders.get('region',     {}).get(str(region), 0)
    c_enc  = encoders.get('cpv_code',   {}).get(str(cpv), 0)
    d = {
        'log_contract_value': log_cv,
        'value_band_num':     vbnum,
        'award_month':        am,
        'award_quarter':      aq,
        'is_quarter_end':     is_qe,
        'is_high_value':      is_hv,
        'buyer_sme_rate':     br,
        'cpv_sme_rate':       cr,
        'region_sme_rate':    rr,
        'value_band_enc':     vb_enc,
        'region_enc':         r_enc,
        'cpv_code_enc':       c_enc,
    }
    row = pd.DataFrame([{c: d.get(c, 0) for c in feature_cols}])
    return scaler.transform(row.values), br, cr, rr

rf, xgb, lr, scaler, encoders, feature_cols, rates, best_name, best_auc = load_artefacts()
tflite = load_tflite()
all_cpv_codes  = list(encoders.get('cpv_code',  {'Unknown': 0}).keys())
all_industries = sorted(set(CPV_LOOKUP.values()))

MODEL_MAP = {
    'Random Forest':                ('Random Forest',       rf),
    'XGBoost':                      ('XGBoost',             xgb),
    'Logistic Regression':          ('Logistic Regression', lr),
    'Logistic Regression (baseline)':('Logistic Regression',lr),
}
best_label, best_model_obj = MODEL_MAP.get(best_name, ('Random Forest', rf))

model_options = []
for opt in ['Random Forest', 'XGBoost', 'Logistic Regression']:
    if opt == best_label:
        auc_str = f' — AUC {best_auc:.3f}' if best_auc else ''
        model_options.append(f'{opt} ⭐ Recommended{auc_str}')
    else:
        model_options.append(opt)
default_index = next((i for i, o in enumerate(model_options) if '⭐' in o), 0)

st.set_page_config(page_title='SME Award Predictor', page_icon='🏆', layout='wide')
st.title('🏆 SME Contract Award Predictor')
st.markdown('Predict the likelihood of a UK public procurement contract being awarded to an **SME**.')

if best_auc:
    st.success(f'🏆 Best performing model: **{best_label}**  |  AUC-ROC: **{best_auc:.4f}**  — automatically set as default')
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(['🔮 ML Predict', '🧠 Deep Learning', '🏭 CPV Lookup', '📊 Historical'])

# ── TAB 1: ML PREDICTION ─────────────────────────────────────────────────────
with tab1:
    st.subheader('Traditional ML Models')
    st.caption(f'Default model is set to the best performer from training: **{best_label}**')
    c1, c2, c3 = st.columns(3)
    with c1:
        cv1 = st.number_input('Contract value (£)', min_value=0.0, value=50000.0, step=1000.0, key='cv1')
        am1 = st.selectbox('Month', list(range(1,13)),
                           format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1],
                           key='am1')
        aq1 = st.selectbox('Quarter', [1,2,3,4], format_func=lambda q: f'Q{q}', key='aq1')
    with c2:
        r1  = st.selectbox('Region',     list(encoders.get('region',     {'Unknown':0}).keys()), key='r1')
        vb1 = st.selectbox('Value band', list(encoders.get('value_band', {'Unknown':0}).keys()), key='vb1')
        inp_method1 = st.radio('CPV input method', ['Select CPV code', 'Select by industry'], key='im1', horizontal=True)
        if inp_method1 == 'Select CPV code':
            cp1 = st.selectbox('CPV code', all_cpv_codes, key='cp1a')
            st.caption(f'Industry: **{cpv_to_industry(cp1)}**')
        else:
            ind1 = st.selectbox('Industry', all_industries, key='ind1')
            cp1  = st.selectbox('CPV code (from industry)', INDUSTRY_LOOKUP.get(ind1, ['Unknown']), key='cp1b')
    with c3:
        bu1 = st.text_input('Buyer name', 'Unknown', key='bu1')
        mc  = st.selectbox('Model', model_options, index=default_index, key='mc1')
    if st.button('Predict', type='primary', use_container_width=True, key='btn1'):
        row, br, cr, rr = build_row(cv1, am1, aq1, r1, cp1, vb1, bu1, encoders, feature_cols, rates, scaler)
        mc_clean = mc.replace(' ⭐ Recommended', '').split(' — AUC')[0].strip()
        m = rf if 'Random Forest' in mc_clean else xgb if 'XGBoost' in mc_clean else lr
        p    = m.predict_proba(row)[0][1]
        pred = int(p >= 0.5)
        a, b, c, d = st.columns(4)
        a.metric('Model',          mc_clean + (' ⭐' if mc_clean == best_label else ''))
        b.metric('Prediction',     'SME' if pred else 'Non-SME')
        c.metric('SME Probability', f'{p*100:.1f}%')
        d.metric('Confidence',     'High' if abs(p-0.5) > 0.25 else 'Low')
        st.progress(float(p))
        if pred:
            st.success('✅ Likely SME award.')
        else:
            st.warning('⚠️ Unlikely SME award.')
        st.info(f'Industry: {cpv_to_industry(cp1)}  |  Buyer SME rate: {br*100:.1f}%  |  Sector: {cr*100:.1f}%  |  Region: {rr*100:.1f}%')

# ── TAB 2: DEEP LEARNING ─────────────────────────────────────────────────────
with tab2:
    st.subheader('Deep Learning — TensorFlow Neural Network')
    st.markdown('Architecture: Input → Dense(64)+BN+Dropout → Dense(32)+BN+Dropout → Dense(16) → Sigmoid')
    c1, c2, c3 = st.columns(3)
    with c1:
        cv2 = st.number_input('Contract value (£)', min_value=0.0, value=50000.0, step=1000.0, key='cv2')
        am2 = st.selectbox('Month', list(range(1,13)),
                           format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1],
                           key='am2')
        aq2 = st.selectbox('Quarter', [1,2,3,4], format_func=lambda q: f'Q{q}', key='aq2')
    with c2:
        r2  = st.selectbox('Region',     list(encoders.get('region',     {'Unknown':0}).keys()), key='r2')
        vb2 = st.selectbox('Value band', list(encoders.get('value_band', {'Unknown':0}).keys()), key='vb2')
        inp_method2 = st.radio('CPV input method', ['Select CPV code', 'Select by industry'], key='im2', horizontal=True)
        if inp_method2 == 'Select CPV code':
            cp2 = st.selectbox('CPV code', all_cpv_codes, key='cp2a')
            st.caption(f'Industry: **{cpv_to_industry(cp2)}**')
        else:
            ind2 = st.selectbox('Industry', all_industries, key='ind2')
            cp2  = st.selectbox('CPV code (from industry)', INDUSTRY_LOOKUP.get(ind2, ['Unknown']), key='cp2b')
    with c3:
        bu2 = st.text_input('Buyer name', 'Unknown', key='bu2')
    if st.button('Predict with TensorFlow', type='primary', use_container_width=True, key='btn2'):
        if tflite is None:
            st.error('TFLite model not available in this deployment.')
        else:
            row2, br2, cr2, rr2 = build_row(cv2, am2, aq2, r2, cp2, vb2, bu2, encoders, feature_cols, rates, scaler)
            p2   = predict_tflite(tflite, row2)
            pred2 = int(p2 >= 0.5)
            a, b, c, d = st.columns(4)
            a.metric('Model',          'TensorFlow NN')
            b.metric('Prediction',     'SME' if pred2 else 'Non-SME')
            c.metric('SME Probability', f'{p2*100:.1f}%')
            d.metric('Confidence',     'High' if abs(p2-0.5) > 0.25 else 'Low')
            st.progress(float(p2))
            if pred2:
                st.success('✅ Likely SME award.')
            else:
                st.warning('⚠️ Unlikely SME award.')
            st.info(f'Industry: {cpv_to_industry(cp2)}  |  Buyer SME rate: {br2*100:.1f}%  |  Sector: {cr2*100:.1f}%  |  Region: {rr2*100:.1f}%')

# ── TAB 3: CPV LOOKUP ────────────────────────────────────────────────────────
with tab3:
    st.subheader('🏭 CPV Code ↔ Industry Lookup')
    st.markdown('Find the industry for any CPV code, or find CPV codes for any industry.')
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('#### CPV code → Industry')
        cpv_input = st.text_input('Enter a CPV code (e.g. 72200000)', key='cpv_in')
        if cpv_input:
            industry_result = cpv_to_industry(cpv_input.strip())
            if industry_result == 'Unknown industry':
                st.warning(f'No industry found for CPV code: {cpv_input}')
            else:
                st.success(f'**{cpv_input}** belongs to: **{industry_result}**')
                related = INDUSTRY_LOOKUP.get(industry_result, [])
                sme_for_code = rates['cpv_sme_rate'].get(str(cpv_input.strip()), rates['global_sme_rate'])
                st.metric('Historical SME award rate for this CPV', f'{sme_for_code*100:.1f}%')
                st.markdown(f'Other CPV codes in **{industry_result}**:')
                st.dataframe(pd.DataFrame({'CPV Code': related, 'Industry': [industry_result]*len(related)}),
                             use_container_width=True)
    with col_right:
        st.markdown('#### Industry → CPV codes')
        industry_input = st.selectbox('Select an industry', all_industries, key='ind_in')
        if industry_input:
            cpv_list = INDUSTRY_LOOKUP.get(industry_input, [])
            st.success(f'**{industry_input}** contains {len(cpv_list)} CPV code(s)')
            rates_in_sector = {c: rates['cpv_sme_rate'].get(c, rates['global_sme_rate']) for c in cpv_list}
            df_sector = pd.DataFrame({
                'CPV Code': cpv_list,
                'SME Rate': [f"{rates_in_sector[c]*100:.1f}%" for c in cpv_list]
            })
            st.dataframe(df_sector, use_container_width=True)
            if rates_in_sector:
                best_cpv  = max(rates_in_sector, key=rates_in_sector.get)
                worst_cpv = min(rates_in_sector, key=rates_in_sector.get)
                st.info(f'Highest SME rate: CPV {best_cpv} at {rates_in_sector[best_cpv]*100:.1f}%')
                st.info(f'Lowest SME rate:  CPV {worst_cpv} at {rates_in_sector[worst_cpv]*100:.1f}%')

# ── TAB 4: HISTORICAL ────────────────────────────────────────────────────────
with tab4:
    st.subheader('Historical SME Procurement Insights')
    col_top1, col_top2, col_top3 = st.columns(3)
    col_top1.metric('Global SME award rate', f"{rates['global_sme_rate']*100:.1f}%")
    col_top2.metric('Best performing model', best_label)
    col_top3.metric('Best AUC-ROC', f'{best_auc:.4f}' if best_auc else 'N/A')
    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('**Top 10 regions by SME award rate:**')
        rdf = pd.DataFrame(list(rates['region_sme_rate'].items()), columns=['Region', 'SME Rate'])
        rdf = rdf.sort_values('SME Rate', ascending=False).head(10)
        rdf['SME Rate'] = rdf['SME Rate'].apply(lambda x: f'{x*100:.1f}%')
        st.dataframe(rdf, use_container_width=True)
    with cb:
        st.markdown('**Top 10 CPV sectors by SME award rate:**')
        cdf = pd.DataFrame(list(rates['cpv_sme_rate'].items()), columns=['CPV Code', 'SME Rate'])
        cdf = cdf.sort_values('SME Rate', ascending=False).head(10)
        cdf['Industry'] = cdf['CPV Code'].apply(cpv_to_industry)
        cdf['SME Rate'] = cdf['SME Rate'].apply(lambda x: f'{x*100:.1f}%')
        st.dataframe(cdf[['CPV Code', 'Industry', 'SME Rate']], use_container_width=True)
    st.divider()
    st.markdown('**All model performance comparison:**')
    try:
        results_df = pd.read_csv('model_comparison.csv')
        results_df = results_df.sort_values('AUC-ROC', ascending=False)
        results_df['Recommended'] = results_df['Model'].apply(
            lambda x: '⭐ Yes' if best_label in x or x in best_label else '')
        st.dataframe(results_df, use_container_width=True)
    except Exception:
        st.info('Model comparison table not available.')

with st.sidebar:
    st.header('About')
    st.markdown('**ML Models:** Random Forest · XGBoost · Logistic Regression')
    st.markdown('**Deep Learning:** TensorFlow Neural Network (TFLite)')
    st.divider()
    if best_auc:
        st.success(f'⭐ Recommended: **{best_label}**\nAUC-ROC: {best_auc:.4f}')
    st.divider()
    st.markdown('**Features used:**')
    st.markdown('- Log contract value')
    st.markdown('- Value band')
    st.markdown('- Award month and quarter')
    st.markdown('- Buyer historical SME rate')
    st.markdown('- Sector historical SME rate')
    st.markdown('- Regional historical SME rate')
    st.divider()
    st.caption('Research prototype — not for operational use.')