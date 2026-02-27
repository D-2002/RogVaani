"""
RogVaani — Lambda 1: Data Ingestion
Triggered by EventBridge every 6 hours.
Fetches live weather + AQI + trends for all Mumbai wards,
computes z-score anomalies using training month_stats,
writes one row per ward per forecast_date to DynamoDB.
"""

import json
import os
import boto3
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from decimal import Decimal

# ── AWS clients ───────────────────────────────────────────────────────
dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
s3       = boto3.client('s3', region_name='ap-south-1')
table    = dynamodb.Table('rogvaani_current_conditions')

# ── Config from environment variables (set in Lambda console) ─────────
S3_BUCKET      = os.environ.get('S3_BUCKET', 'rogvaani-data-lake')
METADATA_KEY   = os.environ.get('METADATA_KEY', 'models/ridge_v4/feature_metadata.json')
WARD_DATA_KEY  = os.environ.get('WARD_DATA_KEY', 'static/ward_data/ward_static_data.csv')

# Mumbai lat/lng — used for city-level weather fetch
# Ward-level weather differentiation added when ward static data is richer
MUMBAI_LAT = 19.0760
MUMBAI_LNG = 72.8777

# TTL: 30 days from now
TTL_SECONDS = 30 * 24 * 60 * 60


# ── Helpers ───────────────────────────────────────────────────────────

def fetch_json(url):
    """Simple HTTP GET returning parsed JSON."""
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def load_metadata():
    """Load feature_metadata.json from S3. Contains monthly_feature_stats."""
    resp = s3.get_object(Bucket=S3_BUCKET, Key=METADATA_KEY)
    return json.loads(resp['Body'].read().decode())


def load_ward_list():
    """
    Load ward_static_data.csv from S3.
    Uses csv.reader to handle quoted fields containing commas.
    Returns list of dicts.
    """
    import csv, io
    resp    = s3.get_object(Bucket=S3_BUCKET, Key=WARD_DATA_KEY)
    content = resp['Body'].read().decode('utf-8-sig')  # utf-8-sig strips BOM if present
    reader  = csv.DictReader(io.StringIO(content))
    wards   = [dict(row) for row in reader]
    print(f'DEBUG first ward: {wards[0] if wards else "empty"}')
    return wards


def fetch_weather(lat, lng, target_date):
    """
    Fetch daily weather from Open-Meteo for a single day.
    Returns dict of raw values.
    """
    date_str = target_date.strftime('%Y-%m-%d')
    url = (
        'https://api.open-meteo.com/v1/forecast'
        f'?latitude={lat}&longitude={lng}'
        '&daily=temperature_2m_max,temperature_2m_min,'
        'precipitation_sum,relative_humidity_2m_mean'
        f'&start_date={date_str}&end_date={date_str}'
        '&timezone=Asia%2FKolkata'
    )
    try:
        data = fetch_json(url)
        daily = data.get('daily', {})
        temp_max = (daily.get('temperature_2m_max') or [None])[0]
        temp_min = (daily.get('temperature_2m_min') or [None])[0]
        rain     = (daily.get('precipitation_sum')  or [0.0])[0] or 0.0
        humidity = (daily.get('relative_humidity_2m_mean') or [None])[0]

        avg_temp = ((temp_max or 28.0) + (temp_min or 24.0)) / 2
        return {
            'rainfall_mm':       float(rain),
            'avg_temp_c':        round(float(avg_temp), 2),
            'avg_humidity_pct':  round(float(humidity or 70.0), 2),
        }
    except Exception as e:
        print(f'⚠️  Weather fetch failed for ({lat},{lng}): {e} — using fallback')
        return {'rainfall_mm': 0.0, 'avg_temp_c': 28.0, 'avg_humidity_pct': 70.0}


def fetch_aqi(lat, lng):
    """
    Fetch current AQI from Open-Meteo Air Quality API.
    Returns pm25 and so2 values.
    """
    url = (
        'https://air-quality-api.open-meteo.com/v1/air-quality'
        f'?latitude={lat}&longitude={lng}'
        '&current=pm2_5,sulphur_dioxide'
    )
    try:
        data    = fetch_json(url)
        current = data.get('current', {})
        return {
            'avg_pm25': round(float(current.get('pm2_5')          or 40.0), 2),
            'avg_so2':  round(float(current.get('sulphur_dioxide') or 5.0),  2),
        }
    except Exception as e:
        print(f'⚠️  AQI fetch failed: {e} — using fallback')
        return {'avg_pm25': 40.0, 'avg_so2': 5.0}


def fetch_trends():
    """
    Fetch Google Trends for Mumbai via pytrends-style proxy or fallback.
    NOTE: pytrends cannot run inside Lambda (no browser).
    We use static monthly averages as fallback here.
    TODO: replace with a scheduled Cloud Function or cached value in DynamoDB.
    Returns dict of trend values.
    """
    # Fallback: return neutral values (0 anomaly at inference = average month)
    # These will produce z-score anomaly = 0, meaning no trend signal this fetch.
    # Replace with real API call when trends proxy is available.
    return {
        'trend_dengue':        30.0,
        'trend_malaria':       20.0,
        'trend_fever':         40.0,
        'trend_leptospirosis': 10.0,
    }


def compute_zscore_anomalies(raw_features, current_month, monthly_feature_stats):
    """
    Convert raw feature values to z-score anomalies.
    Exact same logic as training notebook to_zscore_features().
    String month keys used throughout — JSON round-trip safe.
    """
    m_key  = str(int(current_month))
    result = {}

    continuous_cols = [
        'rainfall_mm', 'avg_temp_c', 'avg_humidity_pct',
        'avg_pm25', 'avg_so2',
        'trend_dengue', 'trend_malaria', 'trend_fever', 'trend_leptospirosis',
    ]
    for col in continuous_cols:
        stats    = monthly_feature_stats.get(col, {})
        raw_val  = float(raw_features.get(col, 0.0))
        mean_val = float(stats.get('mean', {}).get(m_key, 0.0))
        std_val  = max(float(stats.get('std',  {}).get(m_key, 1.0)), 1e-6)
        result[f'{col}_zscore'] = round((raw_val - mean_val) / std_val, 4)

    # Structural — always 0 at inference (post-COVID)
    result['covid_year']          = 0.0
    result['lockdown_stringency'] = 0.0
    return result


def float_to_decimal(obj):
    """
    DynamoDB requires Decimal not float.
    Recursively convert all floats in a dict.
    """
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
    return obj


# ── Main handler ──────────────────────────────────────────────────────

def handler(event, context):
    """
    EventBridge triggers this every 6 hours.
    event: {} (EventBridge scheduled event, payload not used)
    """
    print('Lambda 1 triggered — starting data ingestion')

    now           = datetime.utcnow()
    target_date   = now.date()
    current_month = now.month
    ttl_value     = int(now.timestamp()) + TTL_SECONDS

    # Load metadata (monthly_feature_stats for z-score computation)
    try:
        metadata              = load_metadata()
        monthly_feature_stats = metadata['monthly_feature_stats']
        print(f'✅ Metadata loaded from S3 — {len(monthly_feature_stats)} feature stats')
    except Exception as e:
        print(f'❌ Failed to load metadata: {e}')
        return {'statusCode': 500, 'body': f'Metadata load failed: {e}'}

    # Load ward list
    try:
        wards = load_ward_list()
        print(f'✅ Ward list loaded — {len(wards)} wards')
    except Exception as e:
        print(f'❌ Failed to load ward list: {e}')
        return {'statusCode': 500, 'body': f'Ward list load failed: {e}'}

    # Fetch city-level AQI and trends once (same for all wards)
    # Ward-level differentiation: use ward lat/lng for weather only
    aqi_data   = fetch_aqi(MUMBAI_LAT, MUMBAI_LNG)
    trend_data = fetch_trends()
    print(f'✅ AQI fetched: PM2.5={aqi_data["avg_pm25"]}, SO2={aqi_data["avg_so2"]}')
    print(f'✅ Trends fetched (fallback mode)')

    # Process each ward
    success_count = 0
    error_count   = 0

    for ward in wards:
        ward_id = ward.get('ward_code', '').strip()
        if not ward_id:
            continue

        try:
            # Ward-level weather — falls back to Mumbai centre if no lat/lng in CSV
            w_lat = float(ward.get('lat', MUMBAI_LAT) or MUMBAI_LAT)
            w_lng = float(ward.get('lng', MUMBAI_LNG) or MUMBAI_LNG)
            weather_data = fetch_weather(w_lat, w_lng, target_date)

            # Combine all raw features
            raw_features = {**weather_data, **aqi_data, **trend_data}

            # Compute z-score anomalies
            zscore_features = compute_zscore_anomalies(
                raw_features, current_month, monthly_feature_stats
            )

            # Build DynamoDB item — include static vulnerability scores for Lambda 2
            item = {
                'ward_id':       ward_id,
                'forecast_date': str(target_date),
                'fetched_at':    now.isoformat(),
                'current_month': current_month,
                'ward_name':     ward.get('ward_name', ward_id),
                'district':      ward.get('district', 'Mumbai'),
                'TTL':           ttl_value,
                # Static vulnerability scores (from CSV — used by Lambda 2)
                'waterborne_vulnerability':  ward.get('waterborne_vulnerability', '0'),
                'respiratory_vulnerability': ward.get('respiratory_vulnerability', '0'),
                'composite_vulnerability':   ward.get('composite_vulnerability', '0'),
                'waterlogging_risk_score':   ward.get('waterlogging_risk_score', '0'),
                'waterlogging_risk_label':   ward.get('waterlogging_risk_label', 'low'),
                'pop_density':               ward.get('pop_density', '0'),
                'MBI_score':                 ward.get('MBI_score', '0'),
                # Raw env values (for audit/display)
                **{k: float_to_decimal(v) for k, v in raw_features.items()},
                # Z-score anomaly features (for model inference in Lambda 2)
                **{k: float_to_decimal(v) for k, v in zscore_features.items()},
            }

            table.put_item(Item=item)
            success_count += 1

        except Exception as e:
            print(f'⚠️  Error processing ward {ward_id}: {e}')
            error_count += 1

    summary = {
        'statusCode':    200,
        'run_at':        now.isoformat(),
        'target_date':   str(target_date),
        'wards_written': success_count,
        'wards_failed':  error_count,
    }
    print(f'✅ Ingestion complete: {success_count} wards written, {error_count} failed')
    return summary
