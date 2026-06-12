"""
Module M4: Mô hình lao động (Markov Chain + AI Impact)
"""

import pandas as pd
import numpy as np
from typing import Dict

def create_transition_matrix(automation_risk: dict, reskilling_rate: float = 0.15) -> pd.DataFrame:
    """
    Tạo ma trận chuyển dịch lao động (Markov Chain).
    automation_risk: dict {ngành: tỷ lệ tự động hóa}
    reskilling_rate: tỷ lệ đào tạo lại
    """
    industries = list(automation_risk.keys())
    n = len(industries)
    
    matrix = np.zeros((n, n))
    
    for i, ind_i in enumerate(industries):
        risk = automation_risk[ind_i]
        
        # Xác suất ở lại ngành hiện tại
        stay_prob = 1 - risk
        
        # Xác suất chuyển sang ngành khác (reskilling)
        move_prob = risk * reskilling_rate / (n - 1) if n > 1 else 0
        
        for j, ind_j in enumerate(industries):
            if i == j:
                matrix[i, j] = stay_prob
            else:
                matrix[i, j] = move_prob
    
    return pd.DataFrame(matrix, index=industries, columns=industries)

def simulate_labor_market(initial_employment: dict,
                          transition_matrix: pd.DataFrame,
                          years: int = 5) -> pd.DataFrame:
    """
    Mô phỏng thay đổi việc làm theo năm sử dụng Markov Chain.
    """
    industries = list(initial_employment.keys())
    employment = np.array([initial_employment[ind] for ind in industries])
    
    results = []
    
    for year in range(years + 1):
        results.append({
            'Year': 2025 + year,
            **{ind: round(emp, 1) for ind, emp in zip(industries, employment)}
        })
        
        if year < years:
            # Chuyển sang năm tiếp theo
            employment = employment @ transition_matrix.values
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("=== TEST MODULE M4: Mô hình lao động ===\n")

    # Giả sử 4 ngành
    automation_risk = {
        'Nông nghiệp': 0.15,
        'Công nghiệp': 0.35,
        'Dịch vụ': 0.25,
        'Công nghệ': 0.20
    }

    initial_employment = {
        'Nông nghiệp': 15000,
        'Công nghiệp': 22000,
        'Dịch vụ': 28000,
        'Công nghệ': 8000
    }

    # Tạo ma trận chuyển dịch
    trans_matrix = create_transition_matrix(automation_risk, reskilling_rate=0.20)
    print("Ma trận chuyển dịch (Markov):")
    print(trans_matrix.round(3))
    print()

    # Mô phỏng 5 năm
    result = simulate_labor_market(initial_employment, trans_matrix, years=5)
    print("Mô phỏng việc làm theo năm:")
    print(result.to_string(index=False))