"""
Створення агрегованих ознак для baseline
"""
import pandas as pd
import numpy as np

def extract_session_features(sessions, clicks, buys):
    """
    Створення ознак для кожної сесії
    """
    features = []
    
    for session_id in sessions['Session_ID']:
        session_clicks = clicks[clicks['Session_ID'] == session_id]
        session_buys = buys[buys['Session_ID'] == session_id] if len(buys) > 0 else pd.DataFrame()
        
        feat = {
            'session_id': session_id,
            'session_length': len(session_clicks),
            'unique_items': session_clicks['Item_ID'].nunique(),
            'unique_categories': session_clicks['Category'].nunique(),
            'has_purchase': 1 if len(session_buys) > 0 else 0,
            'num_purchases': len(session_buys) if len(session_buys) > 0 else 0,
            'session_duration_seconds': (
                session_clicks['Timestamp'].max() - session_clicks['Timestamp'].min()
            ).total_seconds() if len(session_clicks) > 1 else 0,
            'click_frequency': len(session_clicks) / (
                (session_clicks['Timestamp'].max() - session_clicks['Timestamp'].min()).total_seconds() / 60
            ) if len(session_clicks) > 1 else 0
        }
        
        features.append(feat)
    
    return pd.DataFrame(features)