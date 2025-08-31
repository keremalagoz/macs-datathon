import pandas as pd
import os

SUB = 'submissions'
A = os.path.join(SUB, 'sk_v1_fulltrain_buy_backoff_log.csv')
B = os.path.join(SUB, 'xgb_v1_fulltrain_buy_backoff_log.csv')
OUT = os.path.join(SUB, 'blend_sk_xgb_fulltrain_v2_log_backoff.csv')

sa = pd.read_csv(A)
sb = pd.read_csv(B)
assert 'user_session' in sa.columns and 'session_value' in sa.columns
assert 'user_session' in sb.columns and 'session_value' in sb.columns

m = sa.merge(sb, on='user_session', how='inner', suffixes=('_sk','_xgb'))
m['session_value'] = 0.5*m['session_value_sk'] + 0.5*m['session_value_xgb']

out = m[['user_session','session_value']]
os.makedirs(SUB, exist_ok=True)
out.to_csv(OUT, index=False)
print({'blended': OUT, 'n': len(out)})
