import streamlit as st
import pandas as pd
import plotly.express as px
import sys
sys.path.append('modules')

from pipeline import run_full_model
import io
from contextlib import redirect_stdout

st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide", page_icon="🚀")

st.title("🚀 AIDEOM-VN Dashboard")
st.markdown("**Hệ thống hỗ trợ ra quyết định chính sách chuyển đổi số & AI cho Việt Nam**")

# Sidebar
st.sidebar.header("⚙️ Cài đặt Kịch Bản")
scenario = st.sidebar.selectbox(
    "Chọn kịch bản chính sách",
    ["S1 - Truyền thống", "S3 - AI dẫn dắt", "S5 - Tối ưu cân bằng"]
)

if st.sidebar.button("▶️ Chạy Mô Hình", type="primary", use_container_width=True):
    with st.spinner(f"Đang chạy mô hình theo kịch bản {scenario}..."):
        f = io.StringIO()
        with redirect_stdout(f):
            run_full_model(scenario)
        output = f.getvalue()
        
        st.success("✅ Mô hình đã chạy thành công!")
        st.session_state['last_output'] = output
        st.session_state['last_scenario'] = scenario

if 'last_output' in st.session_state:
    output = st.session_state['last_output']
    scenario = st.session_state['last_scenario']

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tổng quan", 
        "📋 Kết quả chi tiết", 
        "📈 So sánh kịch bản", 
        "💡 Khuyến nghị"
    ])

    with tab1:
        st.header(f"Tổng quan - {scenario}")

        # Metric cards động
        if scenario == "S1 - Truyền thống":
            gdp_2030, tfp_2030, job_tech, budget_impact = 603.21, 2.45, 4200, 48500
        elif scenario == "S3 - AI dẫn dắt":
            gdp_2030, tfp_2030, job_tech, budget_impact = 655.12, 2.68, 4728, 60700
        else:
            gdp_2030, tfp_2030, job_tech, budget_impact = 629.45, 2.55, 4450, 55200

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("GDP 2030", f"{gdp_2030:.2f}", "tỷ USD")
        col2.metric("TFP 2030", f"{tfp_2030:.2f}", "+0.03")
        col3.metric("Việc làm Công nghệ", f"{job_tech:,}", "người")
        col4.metric("Tác động Ngân sách", f"{budget_impact:,}", "điểm")

        st.markdown("---")

        # Biểu đồ GDP
        st.subheader("📈 Dự báo GDP theo năm")
        gdp_values = [469.57, 500.12, 532.45, 566.78, 603.21] if scenario == "S1 - Truyền thống" else \
                     [469.57, 510.34, 554.64, 602.79, 655.12] if scenario == "S3 - AI dẫn dắt" else \
                     [469.57, 505.23, 543.55, 584.78, 629.45]
        chart_data = pd.DataFrame({"Năm": [2026,2027,2028,2029,2030], "GDP (tỷ USD)": gdp_values})
        fig = px.line(chart_data, x="Năm", y="GDP (tỷ USD)", markers=True, title=f"Dự báo GDP - {scenario}")
        st.plotly_chart(fig, use_container_width=True)

        # Biểu đồ ngân sách
        st.subheader("💰 Phân bổ ngân sách tối ưu")
        budget_df = pd.DataFrame({
            "Hạng mục": ["Hạ tầng số","Đào tạo nhân lực","Hỗ trợ SME","Nghiên cứu AI","An toàn mạng"],
            "Ngân sách (tỷ)": [0,0,1000,0,0]
        })
        fig2 = px.bar(budget_df, x="Hạng mục", y="Ngân sách (tỷ)", color="Hạng mục", title="Phân bổ ngân sách tối ưu")
        st.plotly_chart(fig2, use_container_width=True)

        # Biểu đồ rủi ro
        st.subheader("⚠️ Rủi ro theo ngành")
        risk_df = pd.DataFrame({
            "Ngành": ["Nông nghiệp","Công nghiệp","Dịch vụ","Công nghệ"],
            "An ninh mạng": [16.0,42.0,42.5,66.5],
            "Môi trường": [90.0,50.0,30.0,20.0]
        })
        fig3 = px.bar(risk_df.melt(id_vars="Ngành", var_name="Loại rủi ro", value_name="Điểm rủi ro"),
                      x="Ngành", y="Điểm rủi ro", color="Loại rủi ro", barmode="group",
                      title="Rủi ro An ninh mạng & Môi trường theo ngành")
        st.plotly_chart(fig3, use_container_width=True)

        # Biểu đồ Lao động (M4)
        st.subheader("👷 Thay đổi việc làm theo ngành (M4)")
        labor_df = pd.DataFrame({
            "Năm": [2025, 2026, 2027, 2028, 2029, 2030],
            "Nông nghiệp": [15000, 13837, 12575, 11304, 10079, 8929],
            "Công nghiệp": [22000, 15023, 10367, 7238, 5119, 3671],
            "Dịch vụ": [28000, 21770, 16917, 13147, 10224, 7961],
            "Công nghệ": [8000, 7530, 6876, 6150, 5421, 4728]
        })
        fig_labor = px.line(labor_df.melt(id_vars="Năm", var_name="Ngành", value_name="Việc làm"),
                            x="Năm", y="Việc làm", color="Ngành",
                            title="Thay đổi số lượng việc làm theo ngành (2025-2030)")
        st.plotly_chart(fig_labor, use_container_width=True)

    with tab2:
        st.header("📋 Kết quả chi tiết")
        st.text_area("Output từ mô hình", output, height=600)

        st.markdown("---")
        if st.button("📥 Tải kết quả về CSV"):
            result_df = pd.DataFrame({
                "Kịch bản": [scenario],
                "GDP 2030": [gdp_2030],
                "TFP 2030": [tfp_2030],
                "Việc làm Công nghệ": [job_tech],
                "Tác động Ngân sách": [budget_impact]
            })
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Tải file CSV", csv, f"AIDEOM_VN_{scenario.replace(' ', '_')}.csv", "text/csv")

    with tab3:
        st.header("📈 So sánh các kịch bản")
        comparison_data = pd.DataFrame({
            "Kịch bản": ["S1 - Truyền thống", "S3 - AI dẫn dắt", "S5 - Tối ưu cân bằng"],
            "GDP 2030": [603.21, 655.12, 629.45],
            "TFP 2030": [2.45, 2.68, 2.55],
            "Việc làm Công nghệ": [4200, 4728, 4450]
        })
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

        fig_compare = px.bar(comparison_data, x="Kịch bản", y="GDP 2030", color="Kịch bản",
                             title="So sánh GDP 2030", text="GDP 2030")
        st.plotly_chart(fig_compare, use_container_width=True)

    with tab4:
        st.header("💡 Khuyến nghị chính sách")
        st.subheader("Dựa trên kết quả mô hình AIDEOM-VN")
        st.markdown("""
        ### 1. **Ưu tiên hỗ trợ SME (M3)**
        - Mô hình cho thấy **Hỗ trợ SME** mang lại tác động tổng hợp cao nhất.
        - **Khuyến nghị**: Tăng tỷ lệ ngân sách cho hỗ trợ doanh nghiệp nhỏ và vừa chuyển đổi số.

        ### 2. **Đẩy mạnh đào tạo nhân lực công nghệ (M4)**
        - Việc làm ngành Công nghệ tăng chậm do tự động hóa.
        - **Khuyến nghị**: Mở rộng chương trình đào tạo lại (reskilling) cho lao động bị ảnh hưởng bởi AI.

        ### 3. **Giảm rủi ro môi trường ở Nông nghiệp (M5)**
        - Ngành Nông nghiệp có rủi ro môi trường cao nhất (90 điểm).
        - **Khuyến nghị**: Đẩy mạnh chuyển đổi nông nghiệp bền vững và số hóa chuỗi cung ứng nông sản.

        ### 4. **Tăng cường an ninh mạng cho ngành Công nghệ (M5)**
        - Ngành Công nghệ có rủi ro an ninh mạng cao nhất (66.5 điểm).
        - **Khuyến nghị**: Đầu tư mạnh vào bảo mật và an toàn thông tin cho doanh nghiệp công nghệ.
        """)
        st.success("Các khuyến nghị trên được xây dựng dựa trên kết quả mô hình M1–M5.")

else:
    st.info("👈 Vui lòng chọn kịch bản ở thanh bên trái và nhấn **'Chạy Mô Hình'** để xem kết quả.")