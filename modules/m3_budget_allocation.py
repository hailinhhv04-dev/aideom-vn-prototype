"""
Module M3: Tối ưu phân bổ ngân sách (Linear Programming)
"""

import pandas as pd
import numpy as np
from pulp import *

def optimize_budget(total_budget: float,
                    impact_matrix: pd.DataFrame,
                    min_allocation: dict = None) -> dict:
    """
    Tối ưu phân bổ ngân sách bằng Linear Programming (PuLP).
    impact_matrix: DataFrame với index = các hạng mục, 
                   cột = ['GDP_impact', 'Job_impact', 'Digital_impact']
    """
    # Khởi tạo bài toán
    prob = LpProblem("Budget_Optimization", LpMaximize)
    
    # Biến quyết định (phân bổ cho từng hạng mục)
    categories = impact_matrix.index.tolist()
    allocation = LpVariable.dicts("Allocation", categories, lowBound=0)
    
    # Hàm mục tiêu: Tối đa hóa tác động tổng hợp
    prob += lpSum([allocation[c] * (impact_matrix.loc[c, 'GDP_impact'] * 0.5 +
                                    impact_matrix.loc[c, 'Job_impact'] * 0.3 +
                                    impact_matrix.loc[c, 'Digital_impact'] * 0.2)
                   for c in categories])
    
    # Ràng buộc tổng ngân sách
    prob += lpSum([allocation[c] for c in categories]) == total_budget, "Total_Budget"
    
    # Ràng buộc phân bổ tối thiểu (nếu có)
    if min_allocation:
        for c, min_val in min_allocation.items():
            prob += allocation[c] >= min_val, f"Min_{c}"
    
    # Giải bài toán
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # Kết quả
    result = {
        'status': LpStatus[prob.status],
        'total_budget': total_budget,
        'allocation': {c: value(allocation[c]) for c in categories},
        'total_impact': value(prob.objective)
    }
    
    return result

if __name__ == "__main__":
    print("=== TEST MODULE M3: Tối ưu phân bổ ngân sách ===\n")

    # Dữ liệu tác động mẫu (các hạng mục phân bổ)
    impact_data = pd.DataFrame({
        'GDP_impact': [1.8, 2.5, 1.2, 3.0, 2.0],
        'Job_impact': [120, 80, 200, 50, 150],
        'Digital_impact': [0.8, 1.5, 0.5, 2.2, 1.0]
    }, index=['Hạ tầng số', 'Đào tạo nhân lực', 'Hỗ trợ SME', 
              'Nghiên cứu AI', 'An toàn mạng'])

    total_budget = 1000  # Tỷ đồng

    # Chạy tối ưu
    result = optimize_budget(total_budget, impact_data)

    print(f"Trạng thái: {result['status']}")
    print(f"Tổng ngân sách: {result['total_budget']} tỷ")
    print(f"Tác động tổng hợp: {result['total_impact']:.2f}")
    print("\nPhân bổ tối ưu:")
    for category, amount in result['allocation'].items():
        print(f"  - {category}: {amount:.2f} tỷ")