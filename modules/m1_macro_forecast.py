"""
Module M1: Dự báo kinh tế vĩ mô (Cobb-Douglas + Forecasting)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def estimate_cobb_douglas(data: pd.DataFrame) -> Dict[str, float]:
    """
    Ước lượng hàm sản xuất Cobb-Douglas từ dữ liệu lịch sử.
    Trả về alpha, beta, và A (TFP)
    """
        # Lấy log của các biến
    log_Y = np.log(data['GDP'])
    log_K = np.log(data['Capital'])
    log_L = np.log(data['Labor'])

    # Tạo biến độc lập (thêm hằng số)
    X = np.column_stack([np.ones(len(log_K)), log_K, log_L])

    # Hồi quy OLS
    beta = np.linalg.lstsq(X, log_Y, rcond=None)[0]

    A = np.exp(beta[0])          # TFP
    alpha = beta[1]              # Độ co giãn của vốn
    beta_labor = beta[2]         # Độ co giãn của lao động

    return {
        'A': round(A, 4),
        'alpha': round(alpha, 4),
        'beta': round(beta_labor, 4)
    }


def forecast_macro(cobb_params: Dict[str, float],
                   scenario_params: Dict[str, Any],
                   base_year: int = 2025,
                   target_year: int = 2030) -> pd.DataFrame:
    """
    Dự báo GDP sử dụng mô hình Cobb-Douglas với tích lũy vốn và lao động.
    """
    years = list(range(base_year + 1, target_year + 1))
    
    # Lấy tham số từ Cobb-Douglas
    A = cobb_params.get('A', 1.0)
    alpha = cobb_params.get('alpha', 0.3)
    beta = cobb_params.get('beta', 0.7)
    
    # Tham số kịch bản
    tfp_growth = scenario_params.get('tfp_growth', 0.02)
    capital_growth = scenario_params.get('capital_growth', 0.05)
    labor_growth = scenario_params.get('labor_growth', 0.01)
    
    forecasts = []
    
    # Giá trị khởi tạo (bạn có thể thay sau)
    current_gdp = 450
    current_capital = 220
    current_labor = 58
    
    for year in years:
        # Tăng trưởng các yếu tố
        A *= (1 + tfp_growth)
        current_capital *= (1 + capital_growth)
        current_labor *= (1 + labor_growth)
        
        # Tính GDP theo Cobb-Douglas
        current_gdp = A * (current_capital ** alpha) * (current_labor ** beta)
        
        forecasts.append({
            'Year': year,
            'GDP': round(current_gdp, 2),
            'TFP': round(A, 4),
            'Capital': round(current_capital, 2),
            'Labor': round(current_labor, 2),
            'TFP_Growth': round(tfp_growth * 100, 2),
        })

    return pd.DataFrame(forecasts)
def run_all_scenarios(cobb_params: Dict[str, float],
                      base_year: int = 2025,
                      target_year: int = 2030) -> Dict[str, pd.DataFrame]:
    """
    Chạy dự báo cho 3 kịch bản chính: S1, S3, S5
    """
    scenarios = {
        'S1 - Truyền thống': {
            'tfp_growth': 0.015,
            'capital_growth': 0.045,
            'labor_growth': 0.012
        },
        'S3 - AI dẫn dắt': {
            'tfp_growth': 0.03,
            'capital_growth': 0.055,
            'labor_growth': 0.008
        },
        'S5 - Tối ưu cân bằng': {
            'tfp_growth': 0.022,
            'capital_growth': 0.05,
            'labor_growth': 0.01
        }
    }

    results = {}
    for name, params in scenarios.items():
        df = forecast_macro(cobb_params, params, base_year, target_year)
        df['Scenario'] = name
        results[name] = df

    return results
if __name__ == "__main__":
    print("=" * 60)
    print("MODULE M1 - DỰ BÁO KINH TẾ VĨ MÔ (AIDEOM-VN)")
    print("=" * 60)

    # Dữ liệu mẫu (thay bằng dữ liệu thật sau)
    sample_data = pd.DataFrame({
        'GDP': [300, 320, 350, 380, 410],
        'Capital': [150, 160, 175, 190, 210],
        'Labor': [50, 52, 53, 55, 57]
    })

    # 1. Ước lượng Cobb-Douglas
    print("\n[1] Ước lượng hàm Cobb-Douglas...")
    cobb_params = estimate_cobb_douglas(sample_data)
    print(f"Kết quả: {cobb_params}")

    # 2. Chạy tất cả kịch bản
    print("\n[2] Chạy dự báo cho 3 kịch bản chính...")
    all_results = run_all_scenarios(cobb_params, base_year=2025, target_year=2030)

    for scenario_name, df in all_results.items():
        print(f"\n--- {scenario_name} ---")
        print(df[['Year', 'GDP', 'TFP', 'TFP_Growth']].to_string(index=False))

    print("\n" + "=" * 60)
    print("MODULE M1 HOÀN THÀNH")
    print("=" * 60)