"""
Module M2: Đánh giá sẵn sàng số (Entropy + TOPSIS)
"""

import pandas as pd
import numpy as np
from typing import List, Dict

def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Tính trọng số bằng phương pháp Entropy.
    matrix: ma trận dữ liệu (hàng = phương án, cột = chỉ tiêu)
    """
    # Bước 1: Chuẩn hóa ma trận (tổng theo cột = 1)
    p = matrix / matrix.sum(axis=0)
    
    # Bước 2: Tính entropy
    k = 1 / np.log(matrix.shape[0])
    entropy = -k * (p * np.log(p + 1e-10)).sum(axis=0)  # +1e-10 tránh log(0)
    
    # Bước 3: Tính trọng số
    weights = (1 - entropy) / (1 - entropy).sum()
    
    return weights

def topsis(matrix: np.ndarray, weights: np.ndarray, 
           benefit_criteria: List[bool]) -> np.ndarray:
    """
    Thực hiện thuật toán TOPSIS.
    matrix: ma trận chuẩn hóa (hoặc dữ liệu gốc)
    weights: trọng số từ entropy_weights
    benefit_criteria: list True (lợi ích) / False (chi phí)
    """
    # Bước 1: Chuẩn hóa ma trận (vector normalization)
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    
    # Bước 2: Nhân trọng số
    weighted_matrix = norm_matrix * weights
    
    # Bước 3: Xác định Ideal Best & Ideal Worst
    ideal_best = np.where(benefit_criteria, 
                          weighted_matrix.max(axis=0), 
                          weighted_matrix.min(axis=0))
    ideal_worst = np.where(benefit_criteria, 
                           weighted_matrix.min(axis=0), 
                           weighted_matrix.max(axis=0))
    
    # Bước 4: Tính khoảng cách đến Ideal Best & Worst
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    
    # Bước 5: Tính chỉ số tương đồng (closeness)
    closeness = dist_worst / (dist_best + dist_worst)
    
    return closeness

if __name__ == "__main__":
    print("=== TEST MODULE M2: Entropy + TOPSIS ===\n")

    # Dữ liệu mẫu (5 ngành, 4 chỉ tiêu)
    # Giả sử các chỉ tiêu: Internet penetration, E-commerce, Digital skills, Cloud usage
    data = np.array([
        [0.75, 0.65, 0.55, 0.40],  # Ngành 1
        [0.85, 0.80, 0.70, 0.65],  # Ngành 2
        [0.60, 0.50, 0.45, 0.30],  # Ngành 3
        [0.90, 0.85, 0.80, 0.75],  # Ngành 4
        [0.70, 0.60, 0.65, 0.50],  # Ngành 5
    ])

    # Giả sử tất cả là chỉ tiêu lợi ích (benefit)
    benefit = [True, True, True, True]

    # 1. Tính trọng số Entropy
    weights = entropy_weights(data)
    print("1. Trọng số Entropy:")
    print(weights)
    print()

    # 2. Chạy TOPSIS
    scores = topsis(data, weights, benefit)
    print("2. Chỉ số TOPSIS (càng cao càng tốt):")
    for i, score in enumerate(scores):
        print(f"   Ngành {i+1}: {score:.4f}")

    # Xếp hạng
    ranking = np.argsort(scores)[::-1] + 1
    print(f"\nXếp hạng: {ranking}")