import torch
import numpy as np
import pandas as pd
import time
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import r2_score

TARGET = 'total_spending'
EXOG_AT_T = ['key_rate', 'inflation', 'usd_rub', 'national_spending',
             'share_Здоровье', 'share_Маркетплейсы', 'share_Общественное питание',
             'share_Продовольствие', 'share_Транспорт',
             'population', 'avg_headcount', 'market_access']
STATIC_CAT = ['region_name', 'municipal_district_type']
HORIZONS = [1, 3, 6, 12]

def naive_forecast(train, horizon, target_date, **kw):
    """Последнее известное значение"""
    last = train.sort_values('date').groupby('territory_id')[TARGET].last()
    return last.rename(TARGET).reset_index()

def seasonal_naive_forecast(train, horizon, target_date, growth=True, **kw):
    """Значение того же месяца год назад (на медианный YoY-рост по стране)"""
    T = train['date'].max()
    last = train[train['date'] == T].set_index('territory_id')[TARGET]
    ref_date = target_date - pd.DateOffset(months=12)
    ref = train[train['date'] == ref_date].set_index('territory_id')[TARGET]
    pred = ref.reindex(last.index).fillna(last)   # нет года назад -> last

    if growth:
        prev = train[train['date'] == T - pd.DateOffset(months=12)].set_index('territory_id')[TARGET]
        if len(prev) > 0:
            g = (last / prev.reindex(last.index)).replace([np.inf, -np.inf], np.nan).median()
            pred = pred * float(np.clip(g, 0.8, 1.3))
    return pred.rename(TARGET).reset_index()

def prophet_forecast(train, horizon, target_date, log_transform=False):
    """На каждый мо отдельная модель"""
    train_wide = train.pivot_table(index='territory_id', columns='date', values='total_spending')
    res = []
    
    for territory_id in train_wide.index:
        series = train_wide.loc[territory_id].dropna()
        if len(series) < 3:
            res.append({
                'territory_id': territory_id,
                'total_spending': series.iloc[-1] if len(series) > 0 else 0
            })
            continue
            
        y_values = np.log(series.values + 1) if log_transform else series.values
        df_train = pd.DataFrame({'ds': series.index, 'y': y_values})
        
        model = Prophet(
            growth='linear',
            yearly_seasonality=2,         
            weekly_seasonality=False,     
            daily_seasonality=False,      
            changepoint_prior_scale=0.001, 
            seasonality_prior_scale=1.0,  
            seasonality_mode="additive"
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(df_train)

        fut = model.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
        pred = model.predict(fut)
        yhat = pred['yhat'].iloc[-1]
            
        if log_transform:
            yhat = np.expm1(yhat)
            
        res.append({
            'territory_id': territory_id,
            'total_spending': max(0, yhat)
        })
        
    return pd.DataFrame(res)


def build_direct(train, horizon, holiday):
    """Признаки на момент T -> цель на T+horizon"""
    dates = pd.to_datetime(np.sort(train['date'].unique()))
    Y = (train.pivot(index='territory_id', columns='date', values=TARGET)
              .reindex(columns=dates))
    Y.columns.name = 'date'

    feats = {'y0': Y}
    for k in [1, 2, 3, 6]:
        feats[f'r_{k}'] = Y / Y.shift(k, axis=1) # изменение
    feats['roll3'] = Y.T.rolling(3).mean().T / Y
    feats['roll6'] = Y.T.rolling(6).mean().T / Y
    # 'тот же месяц год назад' относительно текущего уровня (значение на T+h-12)
    feats['ly'] = Y.shift(12 - horizon, axis=1) / Y if horizon < 12 else Y * 0 + 1
    feats['target'] = Y.shift(-horizon, axis=1) / Y - 1 # цель: относительное изменение

    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long = long.replace([np.inf, -np.inf], np.nan).reset_index()

    # экзогенные признаки на момент T
    exog = train[['territory_id', 'date'] + EXOG_AT_T]
    long = long.merge(exog, on=['territory_id', 'date'], how='left')

    # статика территории как категории
    static = train.drop_duplicates('territory_id').set_index('territory_id')[STATIC_CAT]
    for c in STATIC_CAT:
        long[c] = long['territory_id'].map(static[c]).astype('category')

    # календарь целевого месяца месяца (известен заранее)
    tgt_dates = long['date'] + pd.DateOffset(months=horizon)
    long['m_sin'] = np.sin(2 * np.pi * tgt_dates.dt.month / 12)
    long['m_cos'] = np.cos(2 * np.pi * tgt_dates.dt.month / 12)
    long['holiday_tgt'] = tgt_dates.map(holiday)
    return long, dates[-1]

def lgbm_direct_forecast(train, horizon, target_date, holiday=None,
                         min_rows=1000, params=None, **kw):
    if horizon >= 12:   # нет обучающих пар (T, T+12) при 24 мес истории
        return seasonal_naive_forecast(train, horizon, target_date)

    long, cutoff = build_direct(train, horizon, holiday)
    tr = long[long['target'].notna() & long['y0'].notna()]
    if len(tr) < min_rows:
        return seasonal_naive_forecast(train, horizon, target_date)
    pr = long[long['date'] == cutoff].dropna(subset=['y0'])

    drop = ['territory_id', 'date', 'target']
    cols = [c for c in long.columns if c not in drop]

    p = dict(objective='l1', n_estimators=300, learning_rate=0.05, num_leaves=15,
             min_child_samples=50, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=1)
    p.update(params or {})
    model = lgb.LGBMRegressor(**p)
    model.fit(tr[cols], tr['target'], sample_weight=tr['y0'])

    pred = pr['y0'].values * (1 + model.predict(pr[cols]))
    return pd.DataFrame({'territory_id': pr['territory_id'].values,
                         TARGET: np.clip(pred, 0, None)})

def blend_forecast(train, horizon, target_date, w=0.5, **kw):
    a = lgbm_direct_forecast(train, horizon, target_date, **kw).set_index('territory_id')[TARGET]
    b = seasonal_naive_forecast(train, horizon, target_date).set_index('territory_id')[TARGET]
    b = b.reindex(a.index)
    return (w * a + (1 - w) * b).rename(TARGET).reset_index()

def yearly_growth_forecast(train, horizon, target_date, g=0.09, **kw):
    """h=12: уровень месяца T × (1+g). g — сценарный параметр (например, инфляция + надбавка)"""
    last = train.sort_values('date').groupby('territory_id')[TARGET].last()
    return (last * (1 + g)).rename(TARGET).reset_index()

def final_forecast(train, horizon, target_date, holiday=None, g=0.09, **kw):
    if horizon <= 3:
        return blend_forecast(train, horizon, target_date, holiday=holiday, w=0.25)
    if horizon <= 6:
        return seasonal_naive_forecast(train, horizon, target_date)
    return yearly_growth_forecast(train, horizon, target_date, g=g)

def lgbm_resid_forecast(train, horizon, target_date, holiday=None,
                        min_rows=1000, params=None, **kw):
    """LGBM поправляет базу 'год назад'. Цель: y_{T+h} / base - 1"""
    if horizon >= 12:
        return seasonal_naive_forecast(train, horizon, target_date)
    long, cutoff = build_direct(train, horizon, holiday)
    long['base'] = long['ly'] * long['y0'] # значение на T+h-12
    long['res'] = (1 + long['target']) * long['y0'] / long['base'] - 1
    tr = long[long['res'].notna() & long['base'].notna()]
    pr = long[(long['date'] == cutoff) & long['base'].notna()]
    if len(tr) < min_rows:
        return seasonal_naive_forecast(train, horizon, target_date)

    cols = [c for c in long.columns if c not in ['territory_id', 'date', 'target', 'res']]
    p = dict(objective='l1', n_estimators=200, learning_rate=0.03, num_leaves=7,
             min_child_samples=100, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=1)
    p.update(params or {})
    m = lgb.LGBMRegressor(**p).fit(tr[cols], tr['res'], sample_weight=tr['base'])
    pred = pr['base'].values * (1 + m.predict(pr[cols]))
    return pd.DataFrame({'territory_id': pr['territory_id'].values,
                         TARGET: np.clip(pred, 0, None)})

def collect_preds(df, horizon, fns, eval_targets):
    """Одна таблица: y + прогноз каждой модели по (target, territory_id)"""
    dates = pd.to_datetime(np.sort(df['date'].unique()))
    frames = []
    for tgt in eval_targets:
        cutoff = tgt - pd.DateOffset(months=horizon)
        if cutoff not in dates:
            continue
        train = df[df['date'] <= cutoff]
        m = (df[df['date'] == tgt][['territory_id', TARGET]]
             .rename(columns={TARGET: 'y'}))
        for k, (fn, kw) in fns.items():
            pr = fn(train, horizon, tgt, **kw).rename(columns={TARGET: k})
            m = m.merge(pr, on='territory_id', how='left')
        m['target'] = tgt
        frames.append(m)
    P = pd.concat(frames, ignore_index=True)
    P = P.dropna(subset=['sn'])
    for k in fns:
        if k != 'sn':
            print(f'h={horizon} {k}: нет прогноза у {P[k].isna().mean():.1%} строк (подставлен sn)')
            P[k] = P[k].fillna(P['sn'])
    return P

def backtest(df, horizon, forecast_fn, eval_targets, eval_ids=None, **kw):
    dates = pd.to_datetime(np.sort(df['date'].unique()))
    rows = []
    for tgt in eval_targets:
        cutoff = tgt - pd.DateOffset(months=horizon)
        if cutoff not in dates:
            continue
        train = df[df['date'] <= cutoff]
        pred = forecast_fn(train, horizon, tgt, **kw).rename(columns={TARGET: 'pred'})
        act = df[df['date'] == tgt][['territory_id', TARGET]]
        m = act.merge(pred, on='territory_id', how='inner')
        if eval_ids is not None:
            m = m[m['territory_id'].isin(eval_ids)]
        err = (m['pred'] - m[TARGET]).abs()
        rows.append({'target': tgt, 'mae': err.mean(),
                     'wape': err.sum() / m[TARGET].abs().sum(),
                     'r2': r2_score(m[TARGET], m['pred'])})
    return pd.DataFrame(rows)

def run_all(df, fn, eval_targets, horizons=(1, 3, 6, 12), eval_ids=None, **kw):
    """Таблица MAE / WAPE / R2 по горизонтам для одной модели"""
    out = []
    for h in horizons:
        r = backtest(df, h, fn, eval_targets, eval_ids=eval_ids, **kw)
        out.append({'horizon': h, 'MAE': r['mae'].mean(),
                    'WAPE': r['wape'].mean(), 'R2': r['r2'].mean()})
    return pd.DataFrame(out).set_index('horizon').round(3)
    

def summarize(models, df, eval_targets, eval_ids=None):
    """Таблица MAE: строки - модели, столбцы - горизонты"""
    out = {}
    for name, (fn, kw) in models.items():
        out[name] = {h: backtest(df, h, fn, eval_targets, eval_ids=eval_ids, **kw)['mae'].mean()
                     for h in HORIZONS}
    return pd.DataFrame(out).T.round(0)