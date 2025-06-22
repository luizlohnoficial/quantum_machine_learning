import pandas as pd
from credit_risk_quantum_ml.src import data_pipeline


def test_generate_synthetic_data_size():
    df = data_pipeline.generate_synthetic_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    assert "score_bureau" in df.columns
    assert "inadimplente" in df.columns


def test_split_data_proportions():
    df = data_pipeline.generate_synthetic_data()
    train_df, val_df, test_df = data_pipeline.split_data(df)
    total = len(df)
    assert len(train_df) + len(val_df) + len(test_df) == total
    # 70% train, 15% val, 15% test approximately
    assert abs(len(train_df) - 0.7 * total) <= 1
    assert abs(len(val_df) - 0.15 * total) <= 1
    assert abs(len(test_df) - 0.15 * total) <= 1
