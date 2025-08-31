import os
import pandas as pd
import numpy as np

DATA_RAW = os.path.join('data','raw')

train = pd.read_csv(os.path.join(DATA_RAW,'train.csv'))
req = {'user_id','user_session','session_value'}
missing = req - set(train.columns)
if missing:
    print({'error': f'missing columns: {missing}'})
    raise SystemExit(1)

# Session labels
labels = (
    train[['user_session','user_id','session_value']]
    .dropna(subset=['user_session','user_id','session_value'])
    .drop_duplicates(subset=['user_session'], keep='first')
    .reset_index(drop=True)
)
sv = labels['session_value'].astype(float)
# Basic stats
qs = sv.quantile([0.0,0.25,0.5,0.75,0.9,0.95,0.99,0.995,1.0]).to_dict()
mean = float(sv.mean())
std = float(sv.std())

# Count of extreme tails
p99 = float(qs[0.99])
p995 = float(qs[0.995])
maxv = float(qs[1.0])
num_p99 = int((sv > p99).sum())
num_p995 = int((sv > p995).sum())

# Per-user heavy tails
g = labels.groupby('user_id')['session_value']
user_mu = g.mean()
user_cnt = g.size()
users_heavy = int(((user_cnt>=3) & (user_mu>p95 if (p95:=float(qs[0.95])) else 0)).sum())

print({
    'n_sessions': int(len(labels)),
    'n_users': int(labels['user_id'].nunique()),
    'session_value_stats': {
        'mean': round(mean,4), 'std': round(std,4),
        **{str(k): round(v,4) for k,v in qs.items()}
    },
    'n_above_p99': num_p99,
    'n_above_p995': num_p995,
    'max': maxv,
    'users_heavy_mu_gt_p95': users_heavy,
})
