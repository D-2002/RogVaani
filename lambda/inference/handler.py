"""
RogVaani — Lambda 2: Risk Inference (Revised)
Triggered by API Gateway POST /risk

Model deficiency awareness:
- vector:      f_std=0.304, score_range=24pts → reliable, leads output
- waterborne:  f_std=0.033, score_range=2.7pts → flat, ward CSV drives differentiation
- respiratory: f_std=0.013, score_range=1.7pts → flat, ward CSV drives differentiation

Strategy:
- City-level ML score used for vector (reliable)
- Ward vulnerability scores from CSV used heavily for waterborne + respiratory
- Softmax downscaling creates spatial variation even when city score is flat
- Confidence field per disease tells frontend what's driving each score
- Fallback precautions returned instantly; Lambda 3 (Bedrock) will enhance them
"""

import json, os, io, csv, pickle, boto3
from datetime import datetime, timedelta, date
from decimal import Decimal
from math import exp

# ── AWS clients ───────────────────────────────────────────────────────
dynamodb         = boto3.resource('dynamodb', region_name='ap-south-1')
s3               = boto3.client('s3',         region_name='ap-south-1')
conditions_table = dynamodb.Table('rogvaani_current_conditions')
cache_table      = dynamodb.Table('rogvaani_risk_cache')

# ── Config from Lambda environment variables ──────────────────────────
S3_BUCKET     = os.environ.get('S3_BUCKET',     'rogvaani-data-lake')
MODEL_PREFIX  = os.environ.get('MODEL_PREFIX',  'models/ridge_v4/')
WARD_DATA_KEY = os.environ.get('WARD_DATA_KEY', 'static/ward_data/ward_static_data.csv')

# Downscaling hyperparameters
TAU    = 2.0   # softmax sharpness: higher = more concentrated hotspots on map
LAMBDA = 0.6   # spatial differentiation: 0=uniform city score, 1=full vulnerability

# Disease model reliability (from training diagnostics)
# Used to blend ML score with vulnerability-based score
MODEL_RELIABILITY = {
    'vector':      0.85,   # f_std=0.304, 24pt range — trust model
    'waterborne':  0.10,   # f_std=0.033, flat — trust CSV vulnerability
    'respiratory': 0.05,   # f_std=0.013, flat — trust CSV vulnerability
}

CORS_HEADERS = {
    'Content-Type':                'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
}

# Module-level cache — survives warm Lambda invocations
_models   = {}   # {cat: Ridge model}
_metadata = {}   # feature_metadata.json
_wards    = {}   # {ward_code: row_dict}


# ── Loaders ───────────────────────────────────────────────────────────

def load_models():
    global _models, _metadata
    if _models and _metadata:
        return
    print('Cold start: loading models from S3')
    for cat in ['vector', 'waterborne', 'respiratory']:
        local = f'/tmp/model_{cat}.pkl'
        if not os.path.exists(local):
            s3.download_file(S3_BUCKET, f'{MODEL_PREFIX}model_{cat}.pkl', local)
        with open(local, 'rb') as f:
            _models[cat] = pickle.load(f)
    meta = '/tmp/feature_metadata.json'
    if not os.path.exists(meta):
        s3.download_file(S3_BUCKET, f'{MODEL_PREFIX}feature_metadata.json', meta)
    with open(meta) as f:
        _metadata = json.load(f)
    print(f'Models loaded: {list(_models.keys())}')


def load_ward_index():
    global _wards
    if _wards:
        return
    resp    = s3.get_object(Bucket=S3_BUCKET, Key=WARD_DATA_KEY)
    content = resp['Body'].read().decode('utf-8-sig')
    for row in csv.DictReader(io.StringIO(content)):
        wid = row.get('ward_code', '').strip()
        if wid:
            _wards[wid] = row
    print(f'Ward index loaded: {len(_wards)} wards')


# ── Type helpers ──────────────────────────────────────────────────────

def safe_float(val, default=0.0):
    """
    Convert DynamoDB value to float.
    Handles Decimal, float, int, str, and strings with leading apostrophe
    which appears when DynamoDB exports negative Decimals to CSV (e.g. '-5.82).
    """
    if val is None:
        return default
    try:
        return float(str(val).strip().lstrip("'"))
    except Exception:
        return default


def sigmoid(x):
    return 1.0 / (1.0 + exp(-float(x)))


def score_to_level(score):
    if   score >= 65: return 'HIGH'
    elif score >= 45: return 'MED'
    else:             return 'LOW'


def decimal_to_float(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, dict):    return {k: decimal_to_float(v) for k,v in obj.items()}
    if isinstance(obj, list):    return [decimal_to_float(i) for i in obj]
    return obj


def ok(body):
    return {'statusCode': 200, 'headers': CORS_HEADERS,
            'body': json.dumps(decimal_to_float(body))}

def err(code, msg):
    return {'statusCode': code, 'headers': CORS_HEADERS,
            'body': json.dumps({'error': msg})}


# ── Step 1: City-level ML risk ────────────────────────────────────────

def compute_ml_city_risk(zscore_features, feature_cols):
    """
    Run Ridge models → raw ML city risk score 0-100 per disease.
    Also returns SHAP top-5 contributors per disease.
    """
    x = [safe_float(zscore_features.get(f), 0.0) for f in feature_cols]

    ml_scores    = {}
    shap_factors = {}

    for cat, model in _models.items():
        f_x = float(model.intercept_) + sum(
            float(model.coef_[i]) * x[i] for i in range(len(x))
        )
        ml_scores[cat] = round(sigmoid(f_x) * 100, 1)

        contribs = sorted(
            [(feature_cols[i], float(model.coef_[i]) * x[i]) for i in range(len(x))],
            key=lambda kv: abs(kv[1]), reverse=True
        )
        shap_factors[cat] = [
            {
                'feature':   feat,
                'shap':      round(val, 4),
                'direction': 'above_baseline' if val > 0 else 'below_baseline',
                'readable':  feature_to_label(feat),
            }
            for feat, val in contribs[:5]
        ]

    return ml_scores, shap_factors


def feature_to_label(feat):
    """Human-readable label for SHAP feature — used by Bedrock WHY engine."""
    MAP = {
        'rainfall_mm_zscore':          'Rainfall anomaly',
        'avg_temp_c_zscore':           'Temperature anomaly',
        'avg_humidity_pct_zscore':     'Humidity anomaly',
        'avg_pm25_zscore':             'PM2.5 (fine particles)',
        'avg_so2_zscore':              'SO2 (air pollution)',
        'trend_dengue_zscore':         'Dengue search interest',
        'trend_malaria_zscore':        'Malaria search interest',
        'trend_fever_zscore':          'Fever search interest',
        'trend_leptospirosis_zscore':  'Leptospirosis search interest',
        'covid_year':                  'COVID year indicator',
        'lockdown_stringency':         'Lockdown stringency',
    }
    return MAP.get(feat, feat)


# ── Step 2: Vulnerability-blended city score ──────────────────────────

def vulnerability_city_score(ward_row, disease_cat):
    """
    Compute vulnerability-based risk score from ward static data (0-100).
    Used for diseases where ML model is flat (waterborne, respiratory).
    """
    def sf(key): return safe_float(ward_row.get(key), 0.0)

    # Normalise 0-1 scores to 0-100
    if disease_cat == 'waterborne':
        # waterborne_vulnerability already 0-1
        return round(sf('waterborne_vulnerability') * 100, 1)

    elif disease_cat == 'respiratory':
        # respiratory_vulnerability already 0-1
        return round(sf('respiratory_vulnerability') * 100, 1)

    elif disease_cat == 'vector':
        # Weighted: waterlogging + composite + MBI
        wl   = min(sf('waterlogging_risk_score'), 1.0)
        comp = sf('composite_vulnerability')
        mbi  = sf('MBI_score')
        return round((0.40*wl + 0.40*comp + 0.20*mbi) * 100, 1)

    return round(sf('composite_vulnerability') * 100, 1)


def blend_city_scores(ml_scores, ward_row):
    """
    Blend ML score with vulnerability score based on MODEL_RELIABILITY.
    For flat diseases (waterborne, respiratory) vulnerability dominates.
    For vector (reliable) ML score dominates.

    blended = α × ML_score + (1-α) × vulnerability_score
    where α = MODEL_RELIABILITY[disease]
    """
    blended    = {}
    confidence = {}

    for cat in ['vector', 'waterborne', 'respiratory']:
        alpha   = MODEL_RELIABILITY.get(cat, 0.5)
        ml_s    = ml_scores.get(cat, 50.0)
        vuln_s  = vulnerability_city_score(ward_row, cat)
        blend   = round(alpha * ml_s + (1 - alpha) * vuln_s, 1)

        blended[cat]    = blend
        confidence[cat] = {
            'ml_score':    ml_s,
            'vuln_score':  vuln_s,
            'alpha':       alpha,
            'driver':      'model+data' if alpha > 0.5 else 'ward_data',
            'note':        (
                'ML model reliable for this disease'
                if alpha > 0.5
                else 'Driven by ward vulnerability index — ML signal weak for this disease'
            ),
        }

    blended['composite'] = round(
        blended['vector']      * 0.40 +
        blended['waterborne']  * 0.35 +
        blended['respiratory'] * 0.25, 1
    )
    return blended, confidence


# ── Step 3: Ward downscaling ──────────────────────────────────────────

def compute_all_ward_vulns(disease_cat):
    """Pre-compute vulnerability for all wards for a disease."""
    return {
        wid: safe_float(
            vulnerability_city_score(row, disease_cat) / 100.0
        )
        for wid, row in _wards.items()
    }


def softmax_weights(vulns):
    """Compute softmax(τ × V_w) for all wards."""
    exp_vals = {wid: exp(TAU * v) for wid, v in vulns.items()}
    total    = sum(exp_vals.values())
    return {wid: ev / total for wid, ev in exp_vals.items()}


def downscale_to_ward(city_score, ward_id, disease_cat):
    """
    R_ward(w) = R_city × (λ × S_w × N + (1 - λ))
    Ensures mean ward risk ≈ city risk.
    Provides spatial variation even when city score is flat.
    """
    if not _wards:
        return city_score

    vulns   = compute_all_ward_vulns(disease_cat)
    softmax = softmax_weights(vulns)
    N       = len(_wards)
    s_w     = softmax.get(ward_id, 1.0 / N)
    r_ward  = city_score * (LAMBDA * s_w * N + (1 - LAMBDA))
    return round(min(max(r_ward, 0.0), 100.0), 1)


# ── Step 4: Heatmap ───────────────────────────────────────────────────

def build_heatmap(city_scores):
    """
    Downscale city scores to all wards for the risk heatmap.
    Returns list sorted highest risk first.
    Ward lat/lng: if not in CSV, falls back to Mumbai centroid.
    The GeoJSON from S3 is used by frontend to draw boundaries.
    """
    # Pre-compute softmax weights once per disease (not per ward)
    softmaxes = {
        cat: softmax_weights(compute_all_ward_vulns(cat))
        for cat in ['vector', 'waterborne', 'respiratory']
    }
    N = len(_wards)

    heatmap = []
    for wid, wrow in _wards.items():
        ws = {}
        for cat in ['vector', 'waterborne', 'respiratory']:
            s_w   = softmaxes[cat].get(wid, 1.0 / N)
            r_w   = city_scores.get(cat, 50) * (LAMBDA * s_w * N + (1 - LAMBDA))
            ws[cat] = round(min(max(r_w, 0), 100), 1)

        composite = round(
            ws['vector']      * 0.40 +
            ws['waterborne']  * 0.35 +
            ws['respiratory'] * 0.25, 1
        )
        heatmap.append({
            'ward_id':    wid,
            'ward_name':  wrow.get('ward_name', wid),
            'district':   wrow.get('district', 'Mumbai'),
            # lat/lng from CSV if available; frontend uses GeoJSON for boundaries
            'lat':  safe_float(wrow.get('lat',  19.0760)),
            'lng':  safe_float(wrow.get('lng',  72.8777)),
            'risk_score':     composite,
            'risk_level':     score_to_level(composite),
            'disease_scores': ws,
            # Static context for popup
            'waterlogging_label': wrow.get('waterlogging_risk_label', 'unknown'),
            'pop_density':        safe_float(wrow.get('pop_density', 0)),
        })

    heatmap.sort(key=lambda x: x['risk_score'], reverse=True)
    return heatmap


# ── Step 5: Daily forecast ────────────────────────────────────────────

# Seasonal multipliers relative to annual mean
# Source: NVBDCP Maharashtra — same priors as model training
SEASONAL = {
    'vector': {
        1:0.40,2:0.40,3:0.40,4:0.55,5:0.65,
        6:0.85,7:1.20,8:1.30,9:1.20,10:1.05,11:0.85,12:0.75
    },
    'waterborne': {
        1:0.70,2:0.70,3:0.75,4:0.80,5:0.90,
        6:1.10,7:1.25,8:1.20,9:1.05,10:0.95,11:0.85,12:0.75
    },
    'respiratory': {
        1:1.15,2:1.10,3:1.00,4:0.90,5:0.85,
        6:0.85,7:0.85,8:0.85,9:0.95,10:1.05,11:1.15,12:1.20
    },
}

def build_daily_forecast(ward_scores, start_date, end_date):
    """
    Day-by-day risk calendar for the user's stay period.
    Today's ward scores are the anchor; seasonal multipliers shift scores
    forward/backward in time relative to today's month.
    """
    try:
        d_start = datetime.strptime(start_date, '%Y-%m-%d').date()
        d_end   = datetime.strptime(end_date,   '%Y-%m-%d').date()
    except Exception:
        d_start = date.today()
        d_end   = d_start + timedelta(days=3)

    # Clamp forecast to max 14 days to match 7-14 day product claim
    if (d_end - d_start).days > 14:
        d_end = d_start + timedelta(days=14)

    today_m  = date.today().month
    forecast = []
    current  = d_start

    while current <= d_end:
        m  = current.month
        ds = {}
        for cat in ['vector', 'waterborne', 'respiratory']:
            base    = SEASONAL[cat].get(today_m, 1.0)
            target  = SEASONAL[cat].get(m, 1.0)
            ratio   = (target / base) if base > 0 else 1.0
            # Apply ratio but keep within ±30% of today's score to avoid extremes
            ratio   = max(0.70, min(ratio, 1.30))
            modded  = ward_scores.get(cat, 50) * ratio
            ds[cat] = round(min(max(modded, 0), 100), 1)

        composite = round(
            ds['vector']      * 0.40 +
            ds['waterborne']  * 0.35 +
            ds['respiratory'] * 0.25, 1
        )
        forecast.append({
            'date':    str(current),
            'day':     current.strftime('%a').upper(),
            'day_num': current.day,
            'month':   current.strftime('%b').upper(),
            'risk':    composite,
            'level':   score_to_level(composite),
            'disease_scores': ds,
        })
        current += timedelta(days=1)

    return forecast


# ── Step 6: Precaution checklist (fallback — Bedrock enhances these) ──

PRECAUTIONS_DB = {
    'vector': {
        'HIGH': [
            {'text': 'Apply DEET mosquito repellent after 4 PM',
             'reason': 'Mosquito activity peaks at dusk in this area'},
            {'text': 'Wear full-sleeve clothing for outdoor activities',
             'reason': 'Reduces mosquito bite exposure significantly'},
            {'text': 'Avoid stagnant water areas — drains, parks, construction sites',
             'reason': 'Active mosquito breeding sites detected in ward'},
            {'text': 'Sleep under mosquito net if staying overnight',
             'reason': 'High vector activity forecast for next 48 hours'},
        ],
        'MED': [
            {'text': 'Use mosquito repellent for evening outdoor activities',
             'reason': 'Moderate vector activity this month'},
            {'text': 'Keep windows closed or screened after sunset',
             'reason': 'Reduces indoor mosquito entry risk'},
        ],
        'LOW': [
            {'text': 'Standard mosquito precautions recommended',
             'reason': 'Low but present vector activity in area'},
        ],
    },
    'waterborne': {
        'HIGH': [
            {'text': 'Drink only packaged or boiled water',
             'reason': 'Elevated waterborne infection risk in this ward'},
            {'text': 'Avoid raw street food and cut fruit',
             'reason': 'High waterlogging + humidity increases contamination risk'},
            {'text': 'Wash hands thoroughly before every meal',
             'reason': 'GI infection risk elevated in surrounding area'},
            {'text': 'Avoid wading through floodwater',
             'reason': 'Leptospirosis risk — enter through skin cuts'},
        ],
        'MED': [
            {'text': 'Prefer sealed packaged water over tap water',
             'reason': 'Moderate waterborne risk this season'},
            {'text': 'Avoid raw street food during monsoon period',
             'reason': 'Contamination risk elevated with recent rainfall'},
        ],
        'LOW': [],
    },
    'respiratory': {
        'HIGH': [
            {'text': 'Carry N95 mask for all outdoor activities',
             'reason': 'PM2.5 elevated above safe threshold in this area'},
            {'text': 'Avoid outdoor exercise between 7–10 AM and 6–9 PM',
             'reason': 'Peak pollution hours in this district'},
            {'text': 'Keep indoor spaces ventilated with filtered air',
             'reason': 'Sustained high AQI forecast for next 48 hours'},
        ],
        'MED': [
            {'text': 'Use a mask during peak traffic hours',
             'reason': 'AQI moderately elevated in this ward'},
            {'text': 'Limit duration of outdoor exercise',
             'reason': 'Respiratory risk above seasonal baseline'},
        ],
        'LOW': [],
    },
}

ACTIVITY_PRECAUTIONS = {
    'outdoor_leisure': [
        {'text':   'Plan outdoor activities on GREEN days in your forecast calendar',
         'reason': 'Risk varies day-by-day — safer windows exist in your stay',
         'category': 'general'},
    ],
    'commute': [
        {'text':   'Carry hand sanitiser for public transport',
         'reason': 'High-contact surfaces increase infection transmission risk',
         'category': 'general'},
    ],
    'exercise': [
        {'text':   'Exercise before 7 AM or after 8 PM',
         'reason': 'Pollution and mosquito activity lowest in early morning',
         'category': 'general'},
    ],
    'indoor': [],
}


def build_precautions(ward_scores, activity_type):
    """
    Build fallback precaution checklist from rule-based logic.
    Lambda 3 (Bedrock) will generate richer contextual precautions.
    This list is shown immediately while Bedrock loads.
    """
    checklist = []
    for disease in ['vector', 'waterborne', 'respiratory']:
        score = ward_scores.get(disease, 50)
        level = score_to_level(score)
        for p in PRECAUTIONS_DB.get(disease, {}).get(level, []):
            checklist.append({
                'text':     p['text'],
                'reason':   p['reason'],
                'category': disease,
                'checked':  False,
                'source':   'rule_based',   # frontend knows Bedrock may replace this
            })

    for p in ACTIVITY_PRECAUTIONS.get(activity_type, []):
        checklist.append({
            'text':     p['text'],
            'reason':   p['reason'],
            'category': p.get('category', 'general'),
            'checked':  False,
            'source':   'rule_based',
        })

    return checklist


# ── Main handler ──────────────────────────────────────────────────────

def handler(event, context):
    # CORS preflight
    if event.get('httpMethod') == 'OPTIONS':
        return {'statusCode': 200, 'headers': CORS_HEADERS, 'body': ''}

    # Parse request
    try:
        body          = json.loads(event.get('body') or '{}')
        ward_id       = body.get('ward_id', '').strip()      # e.g. "R/C"
        start_date    = body.get('start_date',    str(date.today()))
        end_date      = body.get('end_date',      str(date.today() + timedelta(days=3)))
        activity_type = body.get('activity_type', 'outdoor_leisure')
        forecast_date = body.get('forecast_date', str(date.today()))
    except Exception as e:
        return err(400, f'Invalid request body: {e}')

    if not ward_id:
        return err(400, 'ward_id is required. Example: "R/C" for Colaba')

    # ── Cache check ───────────────────────────────────────────────────
    try:
        cached = cache_table.get_item(
            Key={'ward_id': ward_id, 'forecast_date': forecast_date}
        ).get('Item')
        if cached:
            print(f'Cache HIT: {ward_id}/{forecast_date}')
            payload = json.loads(cached['risk_payload'])
            # Rebuild date-specific parts
            payload['daily_forecast'] = build_daily_forecast(
                payload['ward_scores'], start_date, end_date
            )
            payload['precautions'] = build_precautions(
                payload['ward_scores'], activity_type
            )
            return ok(payload)
    except Exception as e:
        print(f'Cache miss/error: {e}')

    # ── Load models + ward index (cached in /tmp) ─────────────────────
    try:
        load_models()
        load_ward_index()
    except Exception as e:
        return err(500, f'Model load failed: {e}')

    feature_cols = _metadata.get('feature_cols', [])

    # ── Read ward conditions from DynamoDB ────────────────────────────
    try:
        from boto3.dynamodb.conditions import Key as DKey
        item = conditions_table.get_item(
            Key={'ward_id': ward_id, 'forecast_date': forecast_date}
        ).get('Item')

        if not item:
            # Fallback: most recent available row for this ward
            rows = conditions_table.query(
                KeyConditionExpression=DKey('ward_id').eq(ward_id),
                ScanIndexForward=False,
                Limit=1
            ).get('Items', [])
            item = rows[0] if rows else None

        if not item:
            return err(404,
                f'No data for ward "{ward_id}" on {forecast_date}. '
                f'Ward IDs match your CSV ward_code column (e.g. "R/C", "R/S"). '
                f'Check Lambda 1 has run at least once.'
            )
    except Exception as e:
        return err(500, f'DynamoDB read failed: {e}')

    # ── Extract z-score features ──────────────────────────────────────
    zscore_features = {
        col: safe_float(item.get(col), 0.0)
        for col in feature_cols
    }
    print(f'Ward: {ward_id} | Features: {zscore_features}')

    # ── City-level ML risk ────────────────────────────────────────────
    ml_scores, shap_factors = compute_ml_city_risk(zscore_features, feature_cols)
    print(f'ML city scores: {ml_scores}')

    # ── Blend ML + vulnerability (compensates for model flatness) ─────
    ward_row = _wards.get(ward_id, {})
    city_scores, confidence = blend_city_scores(ml_scores, ward_row)
    print(f'Blended city scores: {city_scores}')

    # ── Ward-level downscaling ────────────────────────────────────────
    ward_scores = {}
    for cat in ['vector', 'waterborne', 'respiratory']:
        ward_scores[cat] = downscale_to_ward(city_scores[cat], ward_id, cat)

    ward_scores['composite'] = round(
        ward_scores['vector']      * 0.40 +
        ward_scores['waterborne']  * 0.35 +
        ward_scores['respiratory'] * 0.25, 1
    )
    print(f'Ward scores for {ward_id}: {ward_scores}')

    # ── Heatmap (all wards) ───────────────────────────────────────────
    heatmap = build_heatmap(city_scores)

    # ── Daily forecast ────────────────────────────────────────────────
    daily_forecast = build_daily_forecast(ward_scores, start_date, end_date)

    # ── Precaution checklist (fallback — Bedrock enhances async) ──────
    precautions = build_precautions(ward_scores, activity_type)

    # ── Compose full response ─────────────────────────────────────────
    payload = {
        # Core identity
        'ward_id':       ward_id,
        'ward_name':     ward_row.get('ward_name', ward_id),
        'district':      ward_row.get('district', 'Mumbai'),
        'forecast_date': forecast_date,
        'activity_type': activity_type,

        # Risk card (what UI shows at top)
        'overall_risk':  ward_scores['composite'],
        'risk_level':    score_to_level(ward_scores['composite']),
        'ward_scores':   ward_scores,   # breakdown by disease

        # Confidence metadata (shown as footnote, helps explain output)
        'model_confidence': confidence,

        # SHAP factors passed to Lambda 3 for WHY engine
        'top_shap_factors': shap_factors,

        # City-level anchor scores (for reference + Bedrock context)
        'city_scores':  city_scores,

        # Heatmap — all wards, sorted highest risk first
        'heatmap': heatmap,

        # Day-by-day calendar
        'daily_forecast': daily_forecast,

        # Precaution checklist — rule-based fallback
        # source=rule_based tells frontend Bedrock call may replace these
        'precautions': precautions,

        # Ward context for WHY engine + UI popup
        'ward_context': {
            'waterlogging_label':   ward_row.get('waterlogging_risk_label',  'unknown'),
            'waterlogging_spots':   ward_row.get('waterlogging_spots',       '0'),
            'waterborne_vuln':      ward_row.get('waterborne_vulnerability', 'N/A'),
            'respiratory_vuln':     ward_row.get('respiratory_vulnerability','N/A'),
            'composite_vuln':       ward_row.get('composite_vulnerability',  'N/A'),
            'pop_density':          ward_row.get('pop_density',              'N/A'),
            'slum_pct':             ward_row.get('slum_population_pct',      'N/A'),
            'mbi_score':            ward_row.get('MBI_score',                'N/A'),
            'total_dispensaries':   ward_row.get('total_dispensaries',       'N/A'),
        },

        # Live conditions (for WHY engine + audit)
        'live_conditions': {
            'rainfall_mm':       safe_float(item.get('rainfall_mm')),
            'avg_temp_c':        safe_float(item.get('avg_temp_c')),
            'avg_humidity_pct':  safe_float(item.get('avg_humidity_pct')),
            'avg_pm25':          safe_float(item.get('avg_pm25')),
            'avg_so2':           safe_float(item.get('avg_so2')),
            'fetched_at':        str(item.get('fetched_at', '')),
        },

        'computed_at': datetime.utcnow().isoformat(),
        'model_version': 'ridge_v4',
    }

    # ── Write to cache (24h TTL) ──────────────────────────────────────
    try:
        cache_table.put_item(Item={
            'ward_id':       ward_id,
            'forecast_date': forecast_date,
            'computed_at':   datetime.utcnow().isoformat(),
            'risk_payload':  json.dumps(decimal_to_float(payload)),
            'TTL':           int(datetime.utcnow().timestamp()) + 86400,
        })
    except Exception as e:
        print(f'Cache write failed (non-fatal): {e}')

    return ok(payload)
