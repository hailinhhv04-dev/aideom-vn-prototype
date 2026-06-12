"""
Module M5: Đánh giá rủi ro (Cyber, Environmental, Dependency)
"""

import pandas as pd
import numpy as np
from typing import Dict

def calculate_cyber_risk(digital_exposure: dict, vulnerability: dict) -> dict:
    """
    Tính điểm rủi ro an ninh mạng cho từng ngành.
    digital_exposure: mức độ tiếp xúc số của ngành
    vulnerability: mức độ dễ bị tấn công
    """
    risk_scores = {}
    
    for industry in digital_exposure.keys():
        exposure = digital_exposure[industry]
        vuln = vulnerability.get(industry, 0.5)
        
        # Công thức đơn giản: Rủi ro = Tiếp xúc × Dễ bị tấn công
        risk = exposure * vuln * 100
        risk_scores[industry] = round(risk, 2)
    
    return risk_scores


def calculate_environmental_risk(sector_climate_sensitivity: dict) -> dict:
    """
    Tính điểm rủi ro môi trường theo ngành.
    """
    risk_scores = {}
    for industry, sensitivity in sector_climate_sensitivity.items():
        # Rủi ro môi trường = Độ nhạy cảm với biến đổi khí hậu
        risk = sensitivity * 100
        risk_scores[industry] = round(risk, 2)
    return risk_scores


def calculate_dependency_risk(trade_dependency: dict) -> dict:
    """
    Tính điểm rủi ro phụ thuộc thương mại / chuỗi cung ứng.
    """
    risk_scores = {}
    for industry, dependency in trade_dependency.items():
        # Rủi ro = Mức độ phụ thuộc nhập khẩu
        risk = dependency * 100
        risk_scores[industry] = round(risk, 2)
    return risk_scores

if __name__ == "__main__":
    print("=== TEST MODULE M5: Đánh giá rủi ro ===\n")

    # Dữ liệu mẫu
    digital_exposure = {'Công nghiệp': 0.7, 'Dịch vụ': 0.85, 'Công nghệ': 0.95, 'Nông nghiệp': 0.4}
    vulnerability = {'Công nghiệp': 0.6, 'Dịch vụ': 0.5, 'Công nghệ': 0.7, 'Nông nghiệp': 0.4}
    
    climate_sensitivity = {'Nông nghiệp': 0.9, 'Công nghiệp': 0.5, 'Dịch vụ': 0.3, 'Công nghệ': 0.2}
    trade_dependency = {'Công nghiệp': 0.65, 'Dịch vụ': 0.4, 'Công nghệ': 0.55, 'Nông nghiệp': 0.3}

    # Tính các loại rủi ro
    cyber_risk = calculate_cyber_risk(digital_exposure, vulnerability)
    env_risk = calculate_environmental_risk(climate_sensitivity)
    dep_risk = calculate_dependency_risk(trade_dependency)

    print("1. Rủi ro An ninh mạng:")
    for ind, score in cyber_risk.items():
        print(f"   {ind}: {score}")

    print("\n2. Rủi ro Môi trường:")
    for ind, score in env_risk.items():
        print(f"   {ind}: {score}")

    print("\n3. Rủi ro Phụ thuộc chuỗi cung ứng:")
    for ind, score in dep_risk.items():
        print(f"   {ind}: {score}")