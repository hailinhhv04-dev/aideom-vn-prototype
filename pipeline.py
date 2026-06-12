"""
Pipeline tích hợp AIDEOM-VN (M1 → M5)
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('modules')

from m1_macro_forecast import estimate_cobb_douglas, forecast_macro, run_all_scenarios
from m2_digital_assessment import entropy_weights, topsis
from m3_budget_allocation import optimize_budget
from m4_labor_market import create_transition_matrix, simulate_labor_market
from m5_risk_assessment import (
    calculate_cyber_risk, 
    calculate_environmental_risk, 
    calculate_dependency_risk
)

def run_full_model(scenario_name: str = "S3 - AI dẫn dắt"):
    """
    Chạy toàn bộ mô hình AIDEOM-VN theo một kịch bản.
    """
    print(f"\n{'='*65}")
    print(f"🚀 CHẠY MÔ HÌNH AIDEOM-VN - {scenario_name}")
    print(f"{'='*65}\n")

    # ===================== M1 =====================
    print("[1] M1 - Dự báo kinh tế...")
    sample_macro_data = pd.DataFrame({
        'GDP': [300, 320, 350, 380, 410],
        'Capital': [150, 160, 175, 190, 210],
        'Labor': [50, 52, 53, 55, 57]
    })
    cobb_params = estimate_cobb_douglas(sample_macro_data)
    scenario_params = {'tfp_growth': 0.03, 'capital_growth': 0.055, 'labor_growth': 0.008}
    forecast_result = forecast_macro(cobb_params, scenario_params)
    print(f"    → GDP 2030: {forecast_result['GDP'].iloc[-1]} | TFP: {forecast_result['TFP'].iloc[-1]}")

    # ===================== M2 =====================
    print("\n[2] M2 - Đánh giá sẵn sàng số...")
    digital_data = np.array([[0.75,0.65,0.55,0.40],[0.85,0.80,0.70,0.65],[0.60,0.50,0.45,0.30],[0.90,0.85,0.80,0.75]])
    weights_m2 = entropy_weights(digital_data)
    scores_m2 = topsis(digital_data, weights_m2, [True]*4)
    print(f"    → Chỉ số TOPSIS cao nhất: {scores_m2.max():.4f}")

    # ===================== M3 =====================
    print("\n[3] M3 - Tối ưu phân bổ ngân sách...")
    impact_data = pd.DataFrame({
        'GDP_impact': [1.8, 2.5, 1.2, 3.0, 2.0],
        'Job_impact': [120, 80, 200, 50, 150],
        'Digital_impact': [0.8, 1.5, 0.5, 2.2, 1.0]
    }, index=['Hạ tầng số','Đào tạo nhân lực','Hỗ trợ SME','Nghiên cứu AI','An toàn mạng'])
    budget_result = optimize_budget(1000, impact_data)
    print(f"    → Tác động tổng hợp: {budget_result['total_impact']:.0f}")

    # ===================== M4 =====================
    print("\n[4] M4 - Mô hình lao động...")
    automation = {'Nông nghiệp': 0.15, 'Công nghiệp': 0.35, 'Dịch vụ': 0.25, 'Công nghệ': 0.20}
    initial_emp = {'Nông nghiệp': 15000, 'Công nghiệp': 22000, 'Dịch vụ': 28000, 'Công nghệ': 8000}
    trans_m4 = create_transition_matrix(automation, reskilling_rate=0.20)
    labor_result = simulate_labor_market(initial_emp, trans_m4, years=5)
    print(f"    → Việc làm Công nghệ 2030: {labor_result['Công nghệ'].iloc[-1]:.0f}")

    # ===================== M5 =====================
    print("\n[5] M5 - Đánh giá rủi ro...")
    cyber = calculate_cyber_risk({'Công nghiệp':0.7,'Dịch vụ':0.85,'Công nghệ':0.95,'Nông nghiệp':0.4},
                                 {'Công nghiệp':0.6,'Dịch vụ':0.5,'Công nghệ':0.7,'Nông nghiệp':0.4})
    env = calculate_environmental_risk({'Nông nghiệp':0.9,'Công nghiệp':0.5,'Dịch vụ':0.3,'Công nghệ':0.2})
    print(f"    → Rủi ro An ninh mạng cao nhất: {max(cyber, key=cyber.get)} ({max(cyber.values())})")
    print(f"    → Rủi ro Môi trường cao nhất: {max(env, key=env.get)} ({max(env.values())})")

    print(f"\n{'='*65}")
    print("✅ PIPELINE AIDEOM-VN HOÀN THÀNH")
    print(f"{'='*65}")
if __name__ == "__main__":
    run_full_model("S3 - AI dẫn dắt")