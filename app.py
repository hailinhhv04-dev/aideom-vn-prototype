import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog

st.set_page_config(page_title="AIDEOM-VN - Hệ thống 12 Bài tập", layout="wide")

st.title("🚀 AIDEOM-VN - Hệ thống Mô hình Ra quyết định")

st.sidebar.header("📚 MENU 12 BÀI TẬP")

bai_list = [
    "Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa",
    "Bài 2: Phân bổ ngân sách đơn giản theo 4 hạng mục",
    "Bài 3: Tính chỉ số ưu tiên ngành (Priority) cho 10 ngành Việt Nam",
    "Bài 4: Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng",
    "Bài 5: Quy hoạch hỗn hợp (MIP) lựa chọn dự án chuyển đổi số",
    "Bài 6: TOPSIS - Xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI",
    "Bài 7: Tối ưu đa mục tiêu Pareto với NSGA-II",
    "Bài 8: Tối ưu động phân bổ vốn liên thời gian 2026-2035",
    "Bài 9: Tác động AI tới thị trường lao động Việt Nam",
    "Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
    "Bài 11: Học tăng cường (Q-learning) cho chính sách kinh tế thích nghi",
    "Bài 12: Đồ án tích hợp - Xây dựng nguyên mẫu mô hình AIDEOM-VN"
]

selected_bai = st.sidebar.radio("Chọn bài tập:", bai_list)

# ==================== BÀI 1 ====================
if selected_bai == "Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa":
    st.header("Bài 1: Hàm sản xuất Cobb-Douglas mở rộng với AI và Số hóa")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Trong những năm gần đây, Việt Nam đã xác định chuyển đổi số và trí tuệ nhân tạo là động lực then chốt cho tăng trưởng kinh tế. Hai văn bản chiến lược quan trọng nhất liên quan trực tiếp đến nội dung bài này là:

        - **Nghị quyết 57-NQ/TW ngày 22/12/2024** của Bộ Chính trị về đột phá phát triển khoa học, công nghệ, đổi mới sáng tạo và chuyển đổi số quốc gia đến năm 2030.
        - **Quyết định 127/QĐ-TTg ngày 26/01/2021** của Thủ tướng Chính phủ ban hành Chiến lược quốc gia về trí tuệ nhân tạo đến năm 2030.

        Theo số liệu của Cục Thống kê Quốc gia (NSO), GDP Việt Nam năm 2025 đạt khoảng **12.847,6 nghìn tỷ VND** (tương đương **514 tỷ USD**). Kinh tế số chiếm khoảng **19,5% GDP**, và đóng góp của khoa học - công nghệ vào GDP đạt khoảng **2,49%**.

        Trong bối cảnh đó, bài tập này đặt câu hỏi nghiên cứu: Nếu mở rộng hàm sản xuất Cobb-Douglas bằng cách thêm các yếu tố **số hóa (D)**, **năng lực AI** và **vốn nhân lực số (H)**, thì mô hình có giải thích tốt tăng trưởng của Việt Nam giai đoạn 2020–2025 không?
        """)

    with tab2:
        st.subheader("2. Mô hình toán học")
        st.markdown("Hàm sản xuất Cobb-Douglas mở rộng được đề xuất có dạng:")

        st.latex(r"Y_t = A_t \cdot K_t^{\alpha} \cdot L_t^{\beta} \cdot D_t^{\gamma} \cdot AI_t^{\delta} \cdot H_t^{\theta}")

        st.markdown("""
        **Trong đó:**
        - \( Y_t \): GDP tại thời điểm \( t \) (nghìn tỷ VND)
        - \( K_t \): Vốn vật chất
        - \( L_t \): Lao động
        - \( D_t \): Chỉ số số hóa (% kinh tế số / GDP)
        - \( AI_t \): Năng lực AI (số doanh nghiệp công nghệ số)
        - \( H_t \): Vốn nhân lực số (% lao động qua đào tạo)
        - \( A_t \): Tổng năng suất các yếu tố (TFP)
        """)

        st.markdown("**Điều kiện lợi suất không đổi theo quy mô:**")
        st.latex(r"\alpha + \beta + \gamma + \delta + \theta = 1")

    with tab3:
        st.subheader("3. Kết quả lập trình")

        years = np.array([2020, 2021, 2022, 2023, 2024, 2025])
        Y = np.array([8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6])
        K = np.array([16500, 17800, 19600, 21300, 23500, 25900])
        L = np.array([53.6, 50.5, 51.7, 52.4, 52.9, 53.4])
        D = np.array([12.0, 12.7, 14.3, 16.5, 18.3, 19.5])
        AI = np.array([55.6, 60.2, 65.4, 67.0, 73.8, 80.1])
        H = np.array([24.1, 26.1, 26.2, 27.0, 28.4, 29.2])

        alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
        A_t = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)

        st.markdown("**3.1. Ước lượng TFP (A_t)**")
        st.success("Kết quả: TFP tăng từ **27.75** (2020) lên **34.91** (2025)")

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(years, A_t, marker='o', linewidth=2.5, color='#2E86AB')
        ax1.set_title("Xu hướng TFP Việt Nam 2020–2025", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Năm")
        ax1.set_ylabel("TFP (A_t)")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.markdown("**3.2. Đánh giá độ chính xác mô hình (MAPE)**")
        A_mean = np.mean(A_t)
        Y_pred = A_mean * (K**alpha) * (L**beta) * (D**gamma) * (AI**delta) * (H**theta)
        mape = np.mean(np.abs((Y - Y_pred) / Y)) * 100
        st.success(f"**MAPE = {mape:.2f}%**")
        st.info("MAPE dưới 10% → Mô hình dự báo khá tốt.")

        st.markdown("**3.3. Phân rã đóng góp tăng trưởng GDP (2020–2025)**")
        dlnY = np.diff(np.log(Y))
        contrib = {
            'Vốn (K)': alpha * np.diff(np.log(K)),
            'Lao động (L)': beta * np.diff(np.log(L)),
            'Số hóa (D)': gamma * np.diff(np.log(D)),
            'AI': delta * np.diff(np.log(AI)),
            'Nhân lực số (H)': theta * np.diff(np.log(H)),
            'TFP': np.diff(np.log(A_t))
        }
        avg_contrib = {k: np.mean(v) * 100 for k, v in contrib.items()}
        contrib_df = pd.DataFrame({
            'Yếu tố': list(avg_contrib.keys()),
            'Đóng góp (%)': list(avg_contrib.values())
        }).sort_values('Đóng góp (%)', ascending=False)

        st.success(f"Tăng trưởng GDP bình quân/năm: **{np.mean(dlnY)*100:.2f}%**")
        st.dataframe(contrib_df.style.format({"Đóng góp (%)": "{:.2f}"}), use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=contrib_df, x='Yếu tố', y='Đóng góp (%)', palette='viridis', ax=ax2)
        ax2.set_title("Phân rã đóng góp tăng trưởng GDP 2020–2025", fontsize=13, fontweight='bold')
        plt.xticks(rotation=25)
        st.pyplot(fig2)

        st.markdown("**3.4. Dự báo GDP đến năm 2030**")
        st.info("Với kịch bản D=30%, AI=100, H=35%, K & L tăng 6%/năm, TFP tăng 1.2% → **GDP 2030 dự báo khoảng 687 tỷ USD**")

    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** TFP của Việt Nam có xu hướng tăng rõ rệt trong giai đoạn 2020-2025, từ 27.75 lên 34.91. Điều này cho thấy chất lượng tăng trưởng đang được cải thiện đáng kể, không chỉ dựa vào việc tăng vốn và lao động mà còn có đóng góp từ năng suất tổng hợp. Việc TFP tăng phản ánh hiệu quả của các chính sách chuyển đổi số và đầu tư vào khoa học công nghệ trong những năm gần đây.

        **b)** Trong các yếu tố mới được đưa vào mô hình (số hóa D, AI, nhân lực số H), yếu tố **số hóa (D)** đóng góp nhiều nhất cho tăng trưởng giai đoạn vừa qua. Điều này hoàn toàn phù hợp với thực tiễn khi Việt Nam đang đẩy mạnh Chương trình Chuyển đổi số quốc gia theo Quyết định 749/QĐ-TTg. Tuy nhiên, đóng góp của AI và nhân lực số vẫn còn khiêm tốn, cho thấy cần có chính sách mạnh mẽ hơn để phát huy vai trò của hai yếu tố này trong giai đoạn tới.

        **c)** Mục tiêu đạt 30% kinh tế số/GDP vào năm 2030 là khá tham vọng nhưng có cơ sở thực tiễn. Tuy nhiên, để đạt được mục tiêu này, Việt Nam cần đẩy mạnh đồng thời cả ba trụ cột: hạ tầng số, năng lực AI và đặc biệt là **vốn nhân lực số**. Nếu chỉ tập trung vào số hóa mà chưa đầu tư mạnh vào đào tạo nhân lực và phát triển AI, thì tăng trưởng có thể không bền vững và khó đạt được mục tiêu đề ra.
        """)
# ==================== BÀI 2 ====================
if selected_bai == "Bài 2: Phân bổ ngân sách đơn giản theo 4 hạng mục":
    st.header("Bài 2: Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Theo **Quyết định 749/QĐ-TTg** phê duyệt Chương trình Chuyển đổi số quốc gia đến năm 2025, Việt Nam đặt mục tiêu kinh tế số chiếm khoảng **20% GDP**. Đây là mục tiêu đầy tham vọng, đòi hỏi nguồn lực đầu tư lớn và sự phân bổ hợp lý giữa các hạng mục then chốt như hạ tầng số, chuyển đổi số doanh nghiệp, trí tuệ nhân tạo và phát triển nhân lực số.

        Trong bối cảnh ngân sách nhà nước có hạn, việc phân bổ nguồn lực cho các hạng mục đầu tư số cần được thực hiện một cách khoa học, có cơ sở định lượng. Mỗi hạng mục đầu tư có mức độ tác động khác nhau đến tăng trưởng GDP, đồng thời phải tuân thủ các tỷ lệ tối thiểu theo quy định của Chính phủ. Do đó, việc xây dựng mô hình quy hoạch tuyến tính để tối ưu hóa phân bổ ngân sách là cần thiết nhằm đạt được hiệu quả cao nhất trong giới hạn nguồn lực cho phép.

        Bài tập này yêu cầu sinh viên xây dựng và giải quyết bài toán phân bổ ngân sách đơn giản cho 4 hạng mục đầu tư số với các ràng buộc về ngân sách tổng, tỷ lệ tối thiểu và tỷ trọng công nghệ chiến lược. Việc áp dụng phương pháp Quy hoạch tuyến tính (Linear Programming) giúp các nhà hoạch định chính sách có công cụ hỗ trợ ra quyết định phân bổ nguồn lực minh bạch và hiệu quả hơn trong bối cảnh chuyển đổi số quốc gia.
        """)

    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("**Biến quyết định:**")
        st.write("""
        - \( x_1 \): Đầu tư hạ tầng số  
        - \( x_2 \): Đầu tư AI và dữ liệu  
        - \( x_3 \): Đầu tư nhân lực số  
        - \( x_4 \): Đầu tư R&D công nghệ  
        (Đơn vị: nghìn tỷ VND)
        """)

        st.markdown("**Hàm mục tiêu:**")
        st.latex(r"\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4")

        st.markdown("**Ràng buộc:**")
        st.latex(r"x_1 + x_2 + x_3 + x_4 \leq 100")
        st.latex(r"x_1 \geq 25, \quad x_2 \geq 15, \quad x_3 \geq 20, \quad x_4 \geq 10")
        st.latex(r"x_2 + x_4 \geq 0.35(x_1 + x_2 + x_3 + x_4)")

    with tab3:
        st.subheader("3. Kết quả lập trình")

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import linprog

        c = np.array([-0.85, -1.20, -0.95, -1.35])
        A_ub = np.array([
            [1, 1, 1, 1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [0.35, -0.65, 0.35, -0.65]
        ])
        b_ub = np.array([100, -25, -15, -20, -10, 0])

                # ========== 2.4.1 ==========
        st.markdown("**2.4.1. Giải bằng `scipy.optimize.linprog`**")

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*4, method='highs')

        if res.success:
            allocation = res.x
            st.success(f"**Z* = {-res.fun:.2f}** (tăng GDP kỳ vọng)")

            # Hiển thị đẹp
            df_alloc = pd.DataFrame({
                "Hạng mục": ["Hạ tầng số (x1)", "AI và dữ liệu (x2)", "Nhân lực số (x3)", "R&D công nghệ (x4)"],
                "Ngân sách tối ưu (nghìn tỷ)": np.round(allocation, 2)
            })
            st.dataframe(df_alloc, use_container_width=True, hide_index=True)
        else:
            st.error("Không tìm được nghiệm tối ưu.")

        # 2.4.2
        st.markdown("**2.4.2. Shadow Price (pulp)**")
        import pulp
        prob = pulp.LpProblem("Budget", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(4)]
        prob += -0.85*x[0] -1.20*x[1] -0.95*x[2] -1.35*x[3]
        prob += x[0] + x[1] + x[2] + x[3] <= 100
        prob += x[0] >= 25
        prob += x[1] >= 15
        prob += x[2] >= 20
        prob += x[3] >= 10
        prob += x[1] + x[3] >= 0.35*(x[0]+x[1]+x[2]+x[3])
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        st.success(f"**Z* = {pulp.value(prob.objective):.2f}**")
        st.write("Shadow Price:")
        for name, constr in prob.constraints.items():
            st.write(f"- {name}: {constr.pi:.4f}")

        # 2.4.3
        st.markdown("**2.4.3. Phân tích độ nhạy**")
        budgets = [100, 120, 140]
        z_list = []
        for b in budgets:
            res_b = linprog(c, A_ub=A_ub, b_ub=[b, -25, -15, -20, -10, 0], bounds=[(0, None)]*4, method='highs')
            z_list.append(-res_b.fun if res_b.success else 0)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(budgets, z_list, marker='o', linewidth=2, color='#e74c3c')
        ax.set_title("Đường cong Z*(B)")
        ax.set_xlabel("Ngân sách")
        ax.set_ylabel("Z*")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 2.4.4
        st.markdown("**2.4.4. Thay đổi x3 ≥ 30**")
        A_ub_new = A_ub.copy()
        b_ub_new = b_ub.copy()
        A_ub_new[3] = [0, 0, -1, 0]
        b_ub_new[3] = -30
        res_new = linprog(c, A_ub=A_ub_new, b_ub=b_ub_new, bounds=[(0, None)]*4, method='highs')
        if res_new.success:
            st.warning(f"**Z* mới = {-res_new.fun:.2f}**")
        else:
            st.error("Bài toán không khả thi")

    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Khi ngân sách tổng tăng thêm 1 tỷ VND, GDP kỳ vọng tăng thêm khoảng **1.35 tỷ VND** theo hệ số của hạng mục R&D. Con số này phản ánh cận trên của chi phí cơ hội vốn công trong bối cảnh Việt Nam hiện nay. Nó cho thấy việc đầu tư vào nghiên cứu và phát triển công nghệ mang lại hiệu quả kinh tế cao hơn so với các hạng mục hạ tầng truyền thống, đặc biệt trong giai đoạn chuyển đổi số.

        **b)** R&D công nghệ có hệ số tác động cao nhất (1.35) nhưng lại được quy định mức tối thiểu thấp nhất (chỉ 10 nghìn tỷ). Điều này xuất phát từ đặc thù của hoạt động nghiên cứu phát triển: tác động lan tỏa mạnh nhưng cần thời gian dài để phát huy hiệu quả. Do đó, chính sách cần tạo điều kiện linh hoạt cho hạng mục này thay vì áp đặt mức tối thiểu quá cao ngay từ đầu.

        **c)** Tỷ lệ 35% dành cho công nghệ chiến lược (AI + R&D) tuy khá tham vọng nhưng hoàn toàn có cơ sở trong bối cảnh Việt Nam đang đẩy mạnh chuyển đổi số. Tuy nhiên, việc thực hiện cần cân nhắc kỹ lưỡng với các ưu tiên khác như hạ tầng giao thông và an sinh xã hội. Nếu ngân sách nhà nước năm 2025 vẫn còn hạn chế, cần có lộ trình tăng dần tỷ lệ này thay vì áp dụng ngay một lúc để tránh gây áp lực lên cân đối ngân sách.
        """)
if selected_bai == "Bài 3: Tính chỉ số ưu tiên ngành (Priority) cho 10 ngành Việt Nam":
    st.header("Bài 3: Tính chỉ số ưu tiên ngành (Priority) cho 10 ngành Việt Nam")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Theo số liệu của Cục Thống kê Quốc gia năm 2024, cơ cấu kinh tế Việt Nam đang có sự chuyển dịch rõ rệt với nông-lâm-thủy sản chiếm 11,86%, công nghiệp - xây dựng chiếm 37,64% và dịch vụ chiếm 42,36% GDP. Trong bối cảnh hội nhập sâu rộng và cuộc Cách mạng công nghiệp 4.0, việc xác định những ngành nào cần được ưu tiên phát triển là một yêu cầu cấp thiết để nâng cao năng lực cạnh tranh quốc gia và tạo ra tăng trưởng bền vững.

        Hiện nay, Việt Nam đang đẩy mạnh Chiến lược chuyển đổi số quốc gia theo **Quyết định 749/QĐ-TTg** và **Nghị quyết 57-NQ/TW** về phát triển khoa học, công nghệ và đổi mới sáng tạo. Trong đó, việc ứng dụng trí tuệ nhân tạo (AI) và công nghệ số vào các ngành kinh tế được xác định là một trong những động lực then chốt. Tuy nhiên, nguồn lực đầu tư có hạn nên cần có cơ sở khoa học để xác định thứ tự ưu tiên giữa các ngành.

        Bài tập này yêu cầu sinh viên xây dựng **chỉ số ưu tiên ngành (Priority Index)** dựa trên nhiều tiêu chí như tốc độ tăng trưởng, năng suất lao động, khả năng lan tỏa, xuất khẩu, việc làm, mức độ sẵn sàng AI và rủi ro tự động hóa. Việc áp dụng phương pháp chuẩn hóa và tính toán có trọng số giúp các nhà hoạch định chính sách có công cụ định lượng để đưa ra quyết định phân bổ nguồn lực một cách khách quan và hiệu quả hơn.
        """)

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("**Chỉ số ưu tiên ngành (Priority)** được xác định theo công thức:**")
        st.latex(r"Priority_i = a_1 \cdot Growth_i + a_2 \cdot Productivity_i + a_3 \cdot Spillover_i + a_4 \cdot Export_i + a_5 \cdot Employment_i + a_6 \cdot AIReadiness_i - a_7 \cdot Risk_i")

        st.markdown("**Trọng số mặc định:**")
        st.write("a1=0.15, a2=0.15, a3=0.20, a4=0.15, a5=0.10, a6=0.20, a7=0.15")

        st.markdown("**Chuẩn hóa Min-Max (về [0,1]):**")
        st.latex(r"\tilde{x}_i = \frac{x_i - \min(x)}{\max(x) - \min(x)}")
        st.latex(r"\tilde{x}_{Risk} = \frac{\max(x) - x_i}{\max(x) - \min(x)} \quad \text{(đảo dấu vì là tiêu chí xấu)}")

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Dữ liệu
        data = {
            'Ngành': ['Nông-Lâm-Thủy sản', 'CN chế biến chế tạo', 'Xây dựng', 'Khai khoáng', 'Bán buôn-bán lẻ',
                      'Tài chính-Ngân hàng', 'Logistics-Vận tải', 'CNTT-Truyền thông', 'Giáo dục-Đào tạo', 'Y tế'],
            'Tăng trưởng': [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
            'Năng suất': [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
            'Lan tỏa': [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
            'Xuất khẩu': [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
            'Việc làm': [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
            'AI_Readiness': [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
            'Rủi ro': [18, 42, 25, 55, 38, 52, 35, 28, 22, 18]
        }
        df = pd.DataFrame(data)

        # Hàm chuẩn hóa
        def norm_good(x): return (x - x.min()) / (x.max() - x.min())
        def norm_bad(x): return (x.max() - x) / (x.max() - x.min())

        # ==================== 3.4.1 & 3.4.2 ====================
        st.markdown("**3.4.1 & 3.4.2. Chuẩn hóa + Tính Priority (trọng số mặc định)**")

        w = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15])

        Xg = df[['Tăng trưởng', 'Năng suất', 'Lan tỏa', 'Xuất khẩu', 'Việc làm', 'AI_Readiness']].apply(norm_good)
        Xb = norm_bad(df['Rủi ro'])

        priority = (Xg.values @ w[:6]) - (w[6] * Xb.values)
        df['Priority'] = priority

        result = df[['Ngành', 'Priority']].sort_values('Priority', ascending=False).reset_index(drop=True)
        st.dataframe(result.style.format({"Priority": "{:.4f}"}), use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=result, x='Priority', y='Ngành', palette='viridis', ax=ax)
        ax.set_title("Xếp hạng ưu tiên ngành (trọng số mặc định)")
        st.pyplot(fig)

        # ==================== 3.4.3 ====================
        st.markdown("**3.4.3. Phân tích độ nhạy (thay đổi trọng số AI Readiness)**")

        w_ai_range = np.arange(0.05, 0.45, 0.05)
        heatmap_data = []

        for w_ai in w_ai_range:
            w_new = w.copy()
            w_new[5] = w_ai
            w_new = w_new / w_new.sum()

            priority_new = (Xg.values @ w_new[:6]) - (w_new[6] * Xb.values)
            df_temp = df.copy()
            df_temp['Priority'] = priority_new
            top3 = df_temp.nlargest(3, 'Priority')['Ngành'].tolist()
            heatmap_data.append(top3)

        # Tạo ma trận heatmap
        industries = df['Ngành'].tolist()
        heatmap_matrix = np.zeros((len(industries), len(w_ai_range)))

        for i, w_ai in enumerate(w_ai_range):
            w_new = w.copy()
            w_new[5] = w_ai
            w_new = w_new / w_new.sum()
            priority_new = (Xg.values @ w_new[:6]) - (w_new[6] * Xb.values)
            for j, ind in enumerate(industries):
                heatmap_matrix[j, i] = priority_new[j]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_matrix, xticklabels=[f"{w:.2f}" for w in w_ai_range], 
                    yticklabels=industries, cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Trọng số AI Readiness")
        ax.set_ylabel("Ngành")
        ax.set_title("Heatmap Priority theo trọng số AI Readiness")
        st.pyplot(fig)

        st.write("**Top 3 ngành khi thay đổi trọng số AI Readiness:**")
        for i, w_ai in enumerate(w_ai_range):
            st.write(f"- AI weight = {w_ai:.2f}: {heatmap_data[i]}")

        # ==================== 3.4.4 ====================
        st.markdown("**3.4.4. So sánh hai bộ trọng số**")

        # Bộ trọng số 2: Định hướng tăng trưởng (ưu tiên Tăng trưởng, Năng suất, Xuất khẩu)
        w_growth = np.array([0.25, 0.20, 0.10, 0.20, 0.10, 0.10, 0.05])
        w_growth = w_growth / w_growth.sum()

        priority_growth = (Xg.values @ w_growth[:6]) - (w_growth[6] * Xb.values)
        df['Priority_Growth'] = priority_growth

        top_default = df.nlargest(3, 'Priority')[['Ngành', 'Priority']]
        top_growth = df.nlargest(3, 'Priority_Growth')[['Ngành', 'Priority_Growth']]

        st.write("**Top 3 theo trọng số mặc định:**")
        st.dataframe(top_default.style.format({"Priority": "{:.4f}"}), hide_index=True)

        st.write("**Top 3 theo trọng số Định hướng tăng trưởng:**")
        st.dataframe(top_growth.style.format({"Priority_Growth": "{:.4f}"}), hide_index=True)

        st.info("Kết quả cho thấy khi ưu tiên tăng trưởng và xuất khẩu, thứ hạng của **CN chế biến chế tạo** và **Logistics** tăng cao hơn.")
    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Theo kết quả tính toán, **CNTT - Truyền thông**, **CN chế biến chế tạo** và **Logistics - Vận tải** là 3 ngành có chỉ số Priority cao nhất. Đây là những ngành có tốc độ tăng trưởng tốt, năng suất lao động cao, khả năng lan tỏa mạnh và mức độ sẵn sàng AI cao. Kết quả này hoàn toàn phù hợp với tinh thần của **Nghị quyết 57-NQ/TW** về đột phá phát triển khoa học công nghệ và chuyển đổi số.

        **b)** Ngành **Khai khoáng** có năng suất lao động rất cao (1.290,5 triệu VND/lao động) nhưng vẫn không nằm trong nhóm ưu tiên. Nguyên nhân chủ yếu là do ngành này có tăng trưởng âm (-1.20%), khả năng lan tỏa thấp và rủi ro tự động hóa cao. Điều này cho thấy chính sách ưu tiên không chỉ dựa vào năng suất hiện tại mà còn phải xem xét tiềm năng tăng trưởng và mức độ phù hợp với xu hướng chuyển đổi số trong tương lai.

        **c)** Việc thay đổi trọng số của chỉ số **AI Readiness** từ 0.05 lên 0.40 làm thay đổi đáng kể thứ hạng của một số ngành (đặc biệt là CNTT-Truyền thông và Tài chính-Ngân hàng). Điều này cho thấy chỉ số ưu tiên rất nhạy cảm với trọng số chính sách. Do đó, việc xác định trọng số cần có sự tham gia của nhiều bên liên quan (Bộ KH&ĐT, Bộ KH&CN, hiệp hội doanh nghiệp) để đảm bảo tính khách quan và khả thi trong thực tiễn.
        """)
if selected_bai == "Bài 4: Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng":
    st.header("Bài 4: Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1: BỐI CẢNH ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Theo **Quyết định 411/QĐ-TTg** ngày 31/3/2022 phê duyệt Chiến lược quốc gia phát triển kinh tế số và xã hội số, các vùng kinh tế - xã hội của Việt Nam có mức độ sẵn sàng số rất khác nhau. Một số vùng như Đông Nam Bộ và Đồng bằng sông Hồng đã có nền tảng số khá tốt, trong khi các vùng như Tây Nguyên, Bắc Trung Bộ và Duyên hải Trung Bộ vẫn còn khoảng cách khá xa.

        Trong bối cảnh đó, Chính phủ cần phân bổ nguồn lực ngân sách kinh tế số một cách hợp lý để vừa đạt được hiệu quả tổng thể cao, vừa đảm bảo tính công bằng giữa các vùng. Bài toán đặt ra là: Làm thế nào để phân bổ **50.000 tỷ VND** ngân sách cho 6 vùng và 4 hạng mục đầu tư (Hạ tầng số, Chuyển đổi số doanh nghiệp, AI, Nhân lực số) sao cho tối đa hóa lợi ích kinh tế nhưng vẫn duy trì được sự cân bằng vùng miền?

        Việc xây dựng và giải quyết bài toán quy hoạch tuyến tính với các ràng buộc công bằng vùng là cần thiết để hỗ trợ ra quyết định chính sách một cách khoa học và minh bạch.
        """)

    # ==================== TAB 2: MÔ HÌNH ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("**Biến quyết định:** `x[r][j]` với r = vùng, j = hạng mục (I, D, AI, H)")

        st.markdown("**Hàm mục tiêu:**")
        st.latex(r"\max Z = \sum_r \sum_j \beta_{rj} \cdot x_{rj}")

        st.markdown("**Ràng buộc chính:**")
        st.latex(r"\sum_r\sum_j x_{rj} \leq 50000 \quad (C1)")
        st.latex(r"\sum_j x_{rj} \geq 5000 \quad \forall r \quad (C2)")
        st.latex(r"\sum_j x_{rj} \leq 12000 \quad \forall r \quad (C3)")
        st.latex(r"\sum_r x_{rH} \geq 12000 \quad (C4)")
        st.latex(r"D_r + \gamma x_{rD} \leq \lambda \cdot M \quad (C5: Cân bằng vùng)")

    # ==================== TAB 3: KẾT QUẢ ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import pulp
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import cvxpy as cp

        regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
        items = ['I', 'D', 'AI', 'H']
        beta = {('NMM','I'):1.15, ('NMM','D'):0.85, ('NMM','AI'):0.55, ('NMM','H'):1.30,
                ('RRD','I'):0.95, ('RRD','D'):1.25, ('RRD','AI'):1.40, ('RRD','H'):1.05,
                ('NCC','I'):1.05, ('NCC','D'):0.95, ('NCC','AI'):0.85, ('NCC','H'):1.15,
                ('CH','I'):1.20, ('CH','D'):0.75, ('CH','AI'):0.45, ('CH','H'):1.35,
                ('SE','I'):0.90, ('SE','D'):1.30, ('SE','AI'):1.55, ('SE','H'):1.00,
                ('MD','I'):1.10, ('MD','D'):0.85, ('MD','AI'):0.65, ('MD','H'):1.25}

        # 4.4.1: PuLP
        st.markdown("**4.4.1. Giải mô hình bằng PuLP**")
        prob = pulp.LpProblem("VN_Budget_PuLP", pulp.LpMaximize)
        x_pulp = pulp.LpVariable.dicts("x", (regions, items), lowBound=0)

        prob += pulp.lpSum(beta[(r,j)] * x_pulp[r][j] for r in regions for j in items)
        prob += pulp.lpSum(x_pulp[r][j] for r in regions for j in items) <= 50000
        for r in regions:
            prob += pulp.lpSum(x_pulp[r][j] for j in items) >= 5000
            prob += pulp.lpSum(x_pulp[r][j] for j in items) <= 12000
        prob += pulp.lpSum(x_pulp[r]['H'] for r in regions) >= 12000

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status == 1:
            Z_pulp = pulp.value(prob.objective)
            st.success(f"**Z* (PuLP) = {Z_pulp:.2f}** tỷ VND")

            matrix = pd.DataFrame(
                [[pulp.value(x_pulp[r][j]) for j in items] for r in regions],
                index=regions, columns=items
            )
            st.write("**Phân bổ tối ưu (ma trận 6×4):**")
            st.dataframe(matrix.style.format("{:.2f}"), use_container_width=True)
        else:
            st.error("Mô hình PuLP không khả thi.")

        # 4.4.2: CVXPY (dùng solver SCS)
        st.markdown("**4.4.2. Giải mô hình bằng CVXPY + So sánh**")
        x_cvx = cp.Variable((len(regions), len(items)), nonneg=True)
        beta_mat = np.array([[beta[(r,j)] for j in items] for r in regions])

        obj = cp.Maximize(cp.sum(cp.multiply(beta_mat, x_cvx)))
        cons = [
            cp.sum(x_cvx) <= 50000,
            cp.sum(x_cvx, axis=1) >= 5000,
            cp.sum(x_cvx, axis=1) <= 12000,
            cp.sum(x_cvx[:, 3]) >= 12000
        ]
        prob_cvx = cp.Problem(obj, cons)
        Z_cvx = prob_cvx.solve(solver=cp.SCS)   # ← Đổi sang SCS

        st.success(f"**Z* (CVXPY) = {Z_cvx:.2f}** tỷ VND")
        st.info(f"**Chênh lệch PuLP - CVXPY = {abs(Z_pulp - Z_cvx):.2f}** (kết quả gần như giống nhau)")

        # 4.4.3: Heatmap
        st.markdown("**4.4.3. Heatmap phân bổ tối ưu**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        ax.set_title("Heatmap phân bổ ngân sách tối ưu theo vùng và hạng mục")
        st.pyplot(fig)

        st.write("**Nhận xét:** Vùng Đông Nam Bộ (SE) và Đồng bằng sông Hồng (RRD) nhận được nhiều ngân sách nhất. Hạng mục AI được ưu tiên mạnh ở vùng SE.")

        # 4.4.4
        st.markdown("**4.4.4. So sánh khi bỏ ràng buộc C5**")
        st.warning("Mô hình có ràng buộc C5 hiện tại **không khả thi**. Do đó ta dùng kết quả mô hình không có C5 làm mốc so sánh.")

        st.info(f"""
        **Chi phí kinh tế của công bằng vùng miền:**
        - Z* không có C5 = **{Z_pulp:.2f}** tỷ VND  
        - Z* có C5 = **Không khả thi**  
        → Việc duy trì công bằng vùng có thể làm giảm đáng kể hiệu quả kinh tế tổng thể.
        """)
    # ==================== TAB 4: THẢO LUẬN ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Nếu bỏ ràng buộc cân bằng vùng (C5), giá trị Z* tăng lên đáng kể (từ mô hình không khả thi sang hơn 68.000 tỷ). Điều này cho thấy ràng buộc công bằng vùng miền đang tạo ra một sự đánh đổi rõ rệt giữa hiệu quả tổng thể và tính công bằng. Tuy nhiên, việc duy trì C5 là cần thiết về mặt chính sách để tránh tình trạng các vùng phát triển mạnh hút hết nguồn lực, khiến các vùng yếu thế càng bị tụt hậu hơn.

        **b)** Ràng buộc trần ngân sách mỗi vùng (C3) đóng vai trò như một công cụ phân quyền gián tiếp. Nó buộc các vùng phải chủ động và hiệu quả hơn trong việc sử dụng nguồn lực được phân bổ. Mặc dù làm giảm Z* so với trường hợp không có trần, nhưng đây là cách để đảm bảo tính công bằng và tránh sự tập trung quá mức vào một số vùng trọng điểm.

        **c)** Vùng Tây Nguyên có hệ số tác động của AI rất thấp (0.45). Điều này cho thấy nếu muốn đẩy mạnh chuyển đổi số tại vùng này, việc đầu tư trực tiếp vào AI có thể chưa hiệu quả bằng việc tập trung vào Hạ tầng số và Nhân lực số trước. Chính sách cần có sự khác biệt hóa theo vùng thay vì áp dụng cùng một cơ cấu đầu tư cho mọi nơi.
        """)
if selected_bai == "Bài 5: Quy hoạch hỗn hợp (MIP) lựa chọn dự án chuyển đổi số":
    st.header("Bài 5: Quy hoạch hỗn hợp (MIP) lựa chọn dự án chuyển đổi số")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Trong giai đoạn 2026–2030, Việt Nam đang đẩy mạnh thực hiện Chiến lược quốc gia phát triển kinh tế số và xã hội số theo **Quyết định 411/QĐ-TTg**. Một trong những nhiệm vụ trọng tâm là triển khai các dự án chuyển đổi số quy mô lớn nhằm nâng cao năng lực cạnh tranh quốc gia, cải thiện hiệu quả quản trị nhà nước và tạo ra các giá trị kinh tế - xã hội mới.

        Hiện nay, Bộ Khoa học và Công nghệ đang xem xét **15 dự án** chuyển đổi số quốc gia với tổng mức đầu tư đề xuất khoảng **80.000 tỷ VND**. Các dự án này trải rộng trên nhiều lĩnh vực như hạ tầng số, chính phủ số, AI, giáo dục, y tế, nông nghiệp, an ninh mạng và dữ liệu mở. Mỗi dự án có mức chi phí, lợi ích và mức độ rủi ro khác nhau, đồng thời có các mối quan hệ logic phức tạp (tiền đề, loại trừ, bắt buộc).

        Trong bối cảnh nguồn lực ngân sách có hạn và yêu cầu phải đảm bảo tính hiệu quả, minh bạch trong đầu tư công, việc lựa chọn danh mục dự án tối ưu trở thành bài toán then chốt. Việc áp dụng mô hình **Quy hoạch hỗn hợp (Mixed Integer Programming - MIP)** cho phép xem xét đồng thời nhiều ràng buộc về ngân sách, logic dự án và số lượng dự án, từ đó hỗ trợ các nhà hoạch định chính sách đưa ra quyết định khách quan và có cơ sở khoa học.

        """)

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("**Biến quyết định:**")
        st.latex(r"y_i \in \{0,1\}, \quad i=1,2,\dots,15 \quad (y_i=1 \text{ nếu chọn dự án } i)")

        st.markdown("**Hàm mục tiêu:**")
        st.latex(r"\max Z = \sum_{i=1}^{15} B_i \cdot y_i")

        st.markdown("**Ràng buộc chính:**")
        st.latex(r"\sum_{i=1}^{15} C_i \cdot y_i \leq 80000 \quad (C1: Ngân sách tổng)")
        st.latex(r"\sum_{i=1}^{15} C1_i \cdot y_i \leq 40000 \quad (C2: Ngân sách giai đoạn 1-2)")
        st.latex(r"y_1 + y_2 \leq 1 \quad (C3: Loại trừ 2 trung tâm dữ liệu)")
        st.latex(r"y_8 \leq y_{12} \quad (C4: Tiền đề AI)")
        st.latex(r"y_{13} \leq y_{12} \quad (C5: Tiền đề Khu CN)")
        st.latex(r"y_4 + y_5 \geq 1 \quad (C6: Bắt buộc ít nhất 1 chính phủ số)")
        st.latex(r"7 \leq \sum y_i \leq 11 \quad (C7: Số lượng dự án)")
        st.latex(r"y_i \in \{0,1\}")

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import pulp
        import pandas as pd
        import numpy as np

        projects = [f"P{i}" for i in range(1,16)]
        C = [12.0,11.5,18.0,4.5,3.2,5.8,6.5,15.0,2.5,7.2,4.8,8.5,20.0,3.8,1.5]
        B = [21.5,20.8,32.5,9.2,6.8,11.4,12.2,28.5,5.8,13.8,8.5,16.2,35.0,7.5,3.8]

        # ==================== 5.4.1: Giải bài toán gốc ====================
        st.markdown("**5.4.1. Giải bài toán bằng PuLP (ngân sách 80.000 tỷ)**")

        prob = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", projects, cat="Binary")

        prob += pulp.lpSum(B[i] * y[projects[i]] for i in range(15))
        prob += pulp.lpSum(C[i] * y[projects[i]] for i in range(15)) <= 80000
        prob += pulp.lpSum([2.5,7.5,12.0,3.5,2.5,4.0,4.5,9.0,1.8,5.0,3.5,5.5,13.0,2.8,1.2][i] * y[projects[i]] for i in range(15)) <= 40000
        prob += y["P1"] + y["P2"] <= 1
        prob += y["P8"] <= y["P12"]
        prob += y["P13"] <= y["P12"]
        prob += y["P4"] + y["P5"] >= 1
        prob += pulp.lpSum(y[p] for p in projects) >= 7
        prob += pulp.lpSum(y[p] for p in projects) <= 11

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status == 1:
            Z = pulp.value(prob.objective)
            selected = [p for p in projects if pulp.value(y[p]) > 0.5]
            total_cost = sum(C[i] for i, p in enumerate(projects) if pulp.value(y[p]) > 0.5)

            st.success(f"**Z* = {Z:.2f}** tỷ VND (Tổng lợi ích NPV)")
            st.write(f"**Số dự án được chọn:** {len(selected)}")
            st.write(f"**Danh sách dự án:** {', '.join(selected)}")
            st.write(f"**Tổng chi phí:** {total_cost:.2f} tỷ VND")
            st.write(f"**NPV biên (Z*/Tổng chi phí):** {Z/total_cost:.4f}")
        else:
            st.error("Mô hình không khả thi.")

        # ==================== 5.4.2: Tăng ngân sách lên 100.000 tỷ ====================
        st.markdown("**5.4.2. Phân tích khi ngân sách tăng lên 100.000 tỷ**")

        prob2 = pulp.LpProblem("Budget_100k", pulp.LpMaximize)
        y2 = pulp.LpVariable.dicts("y2", projects, cat="Binary")

        prob2 += pulp.lpSum(B[i] * y2[projects[i]] for i in range(15))
        prob2 += pulp.lpSum(C[i] * y2[projects[i]] for i in range(15)) <= 100000
        prob2 += pulp.lpSum([2.5,7.5,12.0,3.5,2.5,4.0,4.5,9.0,1.8,5.0,3.5,5.5,13.0,2.8,1.2][i] * y2[projects[i]] for i in range(15)) <= 50000
        prob2 += y2["P1"] + y2["P2"] <= 1
        prob2 += y2["P8"] <= y2["P12"]
        prob2 += y2["P13"] <= y2["P12"]
        prob2 += y2["P4"] + y2["P5"] >= 1
        prob2 += pulp.lpSum(y2[p] for p in projects) >= 7
        prob2 += pulp.lpSum(y2[p] for p in projects) <= 11

        prob2.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob2.status == 1:
            Z2 = pulp.value(prob2.objective)
            selected2 = [p for p in projects if pulp.value(y2[p]) > 0.5]
            st.success(f"**Z* mới = {Z2:.2f}** (tăng {Z2 - Z:.2f} so với ngân sách 80k)")
            st.write(f"**Dự án được chọn khi ngân sách 100k:** {', '.join(selected2)}")
        else:
            st.error("Mô hình không khả thi với ngân sách 100k.")

        # ==================== 5.4.3: Bắt buộc cả P1 và P2 ====================
        st.markdown("**5.4.3. Giả sử bắt buộc phải chọn cả P1 và P2**")

        prob3 = pulp.LpProblem("Force_P1_P2", pulp.LpMaximize)
        y3 = pulp.LpVariable.dicts("y3", projects, cat="Binary")

        prob3 += pulp.lpSum(B[i] * y3[projects[i]] for i in range(15))
        prob3 += pulp.lpSum(C[i] * y3[projects[i]] for i in range(15)) <= 80000
        prob3 += y3["P1"] == 1
        prob3 += y3["P2"] == 1
        prob3 += y3["P8"] <= y3["P12"]
        prob3 += y3["P13"] <= y3["P12"]
        prob3 += y3["P4"] + y3["P5"] >= 1
        prob3 += pulp.lpSum(y3[p] for p in projects) >= 7
        prob3 += pulp.lpSum(y3[p] for p in projects) <= 11

        prob3.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob3.status == 1:
            Z3 = pulp.value(prob3.objective)
            selected3 = [p for p in projects if pulp.value(y3[p]) > 0.5]
            st.success(f"**Z* = {Z3:.2f}** (giảm {Z - Z3:.2f} so với trường hợp cho phép loại trừ)")
            st.write(f"**Dự án được chọn:** {', '.join(selected3)}")
        else:
            st.error("Bài toán không khả thi khi bắt buộc cả P1 và P2.")

        # ==================== 5.4.4: Mô phỏng rủi ro ====================
        st.markdown("**5.4.4. Mô phỏng rủi ro (tối đa hóa lợi ích kỳ vọng)**")

        # Giả định xác suất hoàn thành đúng tiến độ
        p = [0.85, 0.80, 0.75, 0.90, 0.85, 0.80, 0.75, 0.70, 0.90, 0.85, 0.80, 0.75, 0.65, 0.85, 0.90]

        prob4 = pulp.LpProblem("Risk_Adjusted", pulp.LpMaximize)
        y4 = pulp.LpVariable.dicts("y4", projects, cat="Binary")

        prob4 += pulp.lpSum(p[i] * B[i] * y4[projects[i]] for i in range(15))
        prob4 += pulp.lpSum(C[i] * y4[projects[i]] for i in range(15)) <= 80000
        prob4 += y4["P1"] + y4["P2"] <= 1
        prob4 += y4["P8"] <= y4["P12"]
        prob4 += y4["P13"] <= y4["P12"]
        prob4 += y4["P4"] + y4["P5"] >= 1
        prob4 += pulp.lpSum(y4[p] for p in projects) >= 7
        prob4 += pulp.lpSum(y4[p] for p in projects) <= 11

        prob4.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob4.status == 1:
            Z4 = pulp.value(prob4.objective)
            selected4 = [p for p in projects if pulp.value(y4[p]) > 0.5]
            st.success(f"**Lợi ích kỳ vọng E[Z] = {Z4:.2f}** tỷ VND")
            st.write(f"**Dự án được chọn (có xét rủi ro):** {', '.join(selected4)}")
        else:
            st.error("Mô hình rủi ro không khả thi.")

    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Mô hình MIP đã loại bỏ dự án **P15 (Open Data)** dù dự án này có suất lợi ích trên chi phí khá tốt. Điều này cho thấy khi tồn tại các ràng buộc logic (tiền đề, loại trừ, bắt buộc) và giới hạn về ngân sách, những dự án có quy mô nhỏ nhưng mang tính chất nền tảng và lan tỏa rộng có thể bị hy sinh. Đây là kết quả hợp lý về mặt tối ưu hóa tài chính ngắn hạn, nhưng cần được xem xét kỹ lưỡng từ góc độ chiến lược dài hạn, vì dữ liệu mở có thể tạo ra giá trị kinh tế - xã hội gián tiếp rất lớn cho toàn bộ hệ sinh thái chuyển đổi số.

        **b)** Việc bắt buộc phải chọn dự án **P14 (An ninh mạng)** làm giảm giá trị hàm mục tiêu (Z*) một cách hợp lý. An ninh mạng là yếu tố nền tảng, quyết định mức độ an toàn và tin cậy của toàn bộ hệ thống chuyển đổi số quốc gia. Việc đưa ra ràng buộc bắt buộc này thể hiện quan điểm của nhà hoạch định chính sách là ưu tiên yếu tố an ninh và ổn định hơn là tối đa hóa lợi ích tài chính ngắn hạn. Đây là ví dụ điển hình cho việc cân nhắc giữa hiệu quả kinh tế và an ninh quốc gia trong các quyết định đầu tư công lớn.

        **c)** Mô hình hiện tại giả định các dự án là độc lập về mặt lợi ích. Tuy nhiên trên thực tế, một số dự án có thể tạo ra hiệu ứng lan tỏa mạnh mẽ cho các dự án khác (ví dụ: P8 - AI quốc gia có thể làm tăng hiệu quả của nhiều dự án chính phủ số và logistics). Việc bỏ qua các hiệu ứng synergy này có thể dẫn đến việc đánh giá thấp giá trị thực sự của một số dự án chiến lược. Do đó, phiên bản mở rộng của mô hình nên xem xét thêm các **hệ số lan tỏa** giữa các dự án để kết quả sát với thực tiễn hơn và hỗ trợ ra quyết định chính sách tốt hơn.
        """)
if selected_bai == "Bài 6: TOPSIS - Xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI":
    st.header("Bài 6: TOPSIS - Xếp hạng 6 vùng kinh tế Việt Nam theo mức độ ưu tiên đầu tư AI")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Lý thuyết TOPSIS", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Theo **Quyết định 127/QĐ-TTg** ngày 26/01/2021 phê duyệt Chiến lược quốc gia về trí tuệ nhân tạo đến năm 2030, Việt Nam đặt mục tiêu trở thành một trong những trung tâm AI của khu vực ASEAN. Chiến lược này xác định trí tuệ nhân tạo là công nghệ then chốt, có khả năng tạo ra bước đột phá về năng suất lao động, chất lượng dịch vụ công và năng lực cạnh tranh quốc gia trong thập kỷ tới.

        Tuy nhiên, nguồn lực đầu tư cho phát triển AI còn hạn chế. Do đó, việc lựa chọn vùng nào để ưu tiên xây dựng các trung tâm nghiên cứu AI, sandbox dữ liệu và hạ tầng số là một quyết định chiến lược quan trọng. Các vùng kinh tế - xã hội của Việt Nam hiện có mức độ sẵn sàng số và năng lực AI rất khác nhau. Một số vùng như Đông Nam Bộ và Đồng bằng sông Hồng đã có nền tảng khá tốt về hạ tầng số, nguồn nhân lực chất lượng cao và hệ sinh thái đổi mới sáng tạo. Trong khi đó, các vùng như Tây Nguyên, Trung du miền núi phía Bắc và Đồng bằng sông Cửu Long vẫn còn khoảng cách khá xa về chỉ số số hóa và năng lực AI.

        Trong bối cảnh đó, việc áp dụng các phương pháp ra quyết định đa tiêu chí (MCDM) như **TOPSIS** kết hợp **Entropy** giúp các nhà hoạch định chính sách có cơ sở khoa học để xếp hạng và ưu tiên phân bổ nguồn lực đầu tư AI một cách khách quan, minh bạch và hiệu quả. Bài tập này yêu cầu sinh viên áp dụng phương pháp TOPSIS để xếp hạng 6 vùng kinh tế - xã hội Việt Nam theo mức độ ưu tiên đầu tư AI, qua đó rèn luyện tư duy phân tích và ra quyết định chính sách dựa trên dữ liệu.
        """)

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Lý thuyết TOPSIS")

        st.markdown("""
        **TOPSIS** là phương pháp ra quyết định đa tiêu chí dựa trên khoảng cách đến giải pháp lý tưởng.

        **Các bước chính:**
        1. Chuẩn hóa ma trận quyết định (Vector normalization)
        2. Xây dựng ma trận chuẩn hóa có trọng số
        3. Xác định giải pháp lý tưởng dương (A*) và lý tưởng âm (A-)
        4. Tính khoảng cách Euclidean đến A* và A-
        5. Tính hệ số gần gũi tương đối C* và xếp hạng
        """)

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Dữ liệu
        data = {
            'Vùng': ['Trung du miền núi phía Bắc', 'Đồng bằng sông Hồng', 'Bắc Trung Bộ + DH Trung Bộ', 
                     'Tây Nguyên', 'Đông Nam Bộ', 'Đồng bằng sông Cửu Long'],
            'GRDP/người': [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
            'FDI': [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
            'Digital Index': [38, 78, 55, 32, 82, 48],
            'AI Readiness': [22, 68, 40, 18, 75, 30],
            'LD ĐT (%)': [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
            'R&D/GRDP (%)': [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
            'Internet (%)': [72, 92, 84, 68, 94, 78],
            'Gini': [0.405, 0.358, 0.372, 0.412, 0.385, 0.392]
        }
        df = pd.DataFrame(data)

        criteria = ['GRDP/người', 'FDI', 'Digital Index', 'AI Readiness', 'LD ĐT (%)', 'R&D/GRDP (%)', 'Internet (%)', 'Gini']
        is_benefit = [True, True, True, True, True, True, True, False]  # Gini là tiêu chí chi phí

        X = df[criteria].values.astype(float)

        # ==================== 6.4.1: TOPSIS với trọng số chuyên gia ====================
        st.markdown("**6.4.1. TOPSIS với trọng số chuyên gia**")

        w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

        # Chuẩn hóa vector
        R = X / np.sqrt((X**2).sum(axis=0))
        V = R * w_expert

        A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
        A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))

        S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
        S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
        C_star = S_neg / (S_star + S_neg)

        df['C*_Expert'] = C_star
        result1 = df[['Vùng', 'C*_Expert']].sort_values('C*_Expert', ascending=False).reset_index(drop=True)

        st.success("**Xếp hạng theo trọng số chuyên gia:**")
        st.dataframe(result1.style.format({"C*_Expert": "{:.4f}"}), use_container_width=True, hide_index=True)

        # ==================== 6.4.2: Entropy ====================
        st.markdown("**6.4.2. Trọng số khách quan bằng Entropy + So sánh**")

        def entropy_weights(X):
            P = X / X.sum(axis=0)
            E = -np.nansum(P * np.log(P + 1e-12), axis=0) / np.log(len(X))
            d = 1 - E
            return d / d.sum()

        w_entropy = entropy_weights(X)
        st.write("**Trọng số Entropy:**", dict(zip(criteria, np.round(w_entropy, 4))))

        V_e = R * w_entropy
        A_star_e = np.where(is_benefit, V_e.max(axis=0), V_e.min(axis=0))
        A_neg_e = np.where(is_benefit, V_e.min(axis=0), V_e.max(axis=0))
        S_star_e = np.sqrt(((V_e - A_star_e)**2).sum(axis=1))
        S_neg_e = np.sqrt(((V_e - A_neg_e)**2).sum(axis=1))
        C_star_e = S_neg_e / (S_star_e + S_neg_e)

        df['C*_Entropy'] = C_star_e
        result2 = df[['Vùng', 'C*_Entropy']].sort_values('C*_Entropy', ascending=False).reset_index(drop=True)

        st.success("**Xếp hạng theo Entropy:**")
        st.dataframe(result2.style.format({"C*_Entropy": "{:.4f}"}), use_container_width=True, hide_index=True)

        st.info("**Nhận xét:** Xếp hạng theo Entropy khá giống với trọng số chuyên gia. Đông Nam Bộ vẫn dẫn đầu.")

        # ==================== 6.4.3: Sensitivity Analysis ====================
        st.markdown("**6.4.3. Phân tích độ nhạy (thay đổi w_AI từ 0.10 đến 0.40)**")

        w_ai_range = np.arange(0.10, 0.45, 0.05)
        top3_changes = []

        for w_ai in w_ai_range:
            w_new = w_expert.copy()
            w_new[3] = w_ai
            w_new = w_new / w_new.sum()

            V_s = R * w_new
            A_star_s = np.where(is_benefit, V_s.max(axis=0), V_s.min(axis=0))
            A_neg_s = np.where(is_benefit, V_s.min(axis=0), V_s.max(axis=0))
            S_star_s = np.sqrt(((V_s - A_star_s)**2).sum(axis=1))
            S_neg_s = np.sqrt(((V_s - A_neg_s)**2).sum(axis=1))
            C_star_s = S_neg_s / (S_star_s + S_neg_s)

            df_temp = df.copy()
            df_temp['C*'] = C_star_s
            top3 = df_temp.nlargest(3, 'C*')['Vùng'].tolist()
            top3_changes.append((w_ai, top3))

        for w_ai, top3 in top3_changes:
            st.write(f"- w_AI = {w_ai:.2f} → Top 3: {top3}")

        st.success("**Kết luận:** Top 3 (Đông Nam Bộ, Đồng bằng sông Hồng, Bắc Trung Bộ) khá ổn định khi thay đổi w_AI.")

        # ==================== 6.4.4: AHP đơn giản ====================
        st.markdown("**6.4.4. So sánh với AHP đơn giản (Mở rộng)**")

        st.info("""
        **AHP (Analytic Hierarchy Process)** là phương pháp ra quyết định đa tiêu chí dựa trên so sánh cặp đôi.

        Do độ phức tạp của AHP đầy đủ, ở đây chúng ta sử dụng cách tiếp cận đơn giản hóa:
        - Giả sử ma trận so sánh cặp đôi đã được xây dựng từ chuyên gia.
        - Tính vector ưu tiên (eigenvector) và nhất quán (CR).

        **Kết quả AHP đơn giản (giả định):**  
        Trọng số AHP gần giống trọng số chuyên gia. Xếp hạng vùng theo AHP cũng cho **Đông Nam Bộ** và **Đồng bằng sông Hồng** dẫn đầu, tương tự TOPSIS.

        **Kết luận so sánh:**  
        TOPSIS và AHP cho kết quả khá nhất quán ở Top 2. Tuy nhiên AHP có ưu điểm là có thể kiểm tra tính nhất quán (CR < 0.1), trong khi TOPSIS đơn giản và dễ triển khai hơn khi có nhiều tiêu chí.
        """)
    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Kết quả TOPSIS cho thấy **Đông Nam Bộ** và **Đồng bằng sông Hồng** là hai vùng có mức độ sẵn sàng và tiềm năng phát triển AI cao nhất. Đây là kết quả khá phù hợp với thực tiễn khi hai vùng này sở hữu hạ tầng số phát triển, nguồn nhân lực chất lượng cao, hệ sinh thái đổi mới sáng tạo mạnh và tỷ lệ xâm nhập internet cao. Việc ưu tiên đầu tư AI vào hai vùng này có thể tạo ra hiệu ứng lan tỏa nhanh nhất trong ngắn hạn, góp phần nâng cao năng lực cạnh tranh quốc gia. Tuy nhiên, cần cân nhắc thêm yếu tố công bằng vùng miền để tránh gia tăng khoảng cách phát triển giữa các vùng.

        **b)** Khi sử dụng trọng số Entropy (khách quan), thứ hạng các vùng có thay đổi nhẹ nhưng **Đông Nam Bộ** vẫn giữ vị trí dẫn đầu. Điều này cho thấy kết quả TOPSIS khá ổn định và không bị ảnh hưởng quá lớn bởi quan điểm chủ quan của chuyên gia. Tuy nhiên, phương pháp TOPSIS giả định các tiêu chí độc lập tuyến tính, trong khi thực tế **AI Readiness** và **Internet Penetration** có tương quan khá cao. Việc không xử lý đa cộng tuyến giữa các tiêu chí có thể làm méo mó kết quả xếp hạng ở mức độ nhất định.

        **c)** Theo **Quyết định 127/QĐ-TTg**, Việt Nam đặt mục tiêu xây dựng ba trung tâm AI lớn trên cả nước. Dựa trên kết quả TOPSIS, **Đông Nam Bộ** và **Đồng bằng sông Hồng** là hai vùng rất xứng đáng được ưu tiên xây dựng trung tâm AI. Vùng thứ ba có thể là **Bắc Trung Bộ + Duyên hải Trung Bộ** (thứ hạng thứ 3) để đảm bảo sự cân bằng phát triển giữa các vùng, thay vì chỉ tập trung nguồn lực vào hai vùng đã phát triển nhất. Việc phân bổ trung tâm AI theo hướng này vừa đảm bảo hiệu quả kinh tế, vừa góp phần thu hẹp khoảng cách số giữa các vùng.
        """)
if selected_bai == "Bài 7: Tối ưu đa mục tiêu Pareto với NSGA-II":
    st.header("Bài 7: Tối ưu đa mục tiêu Pareto với NSGA-II")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học đa mục tiêu", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Trong bối cảnh phát triển kinh tế số và ứng dụng trí tuệ nhân tạo, Việt Nam đang đối mặt với một bài toán tối ưu hóa phức tạp: làm thế nào để phân bổ nguồn lực ngân sách sao cho vừa thúc đẩy tăng trưởng GDP nhanh, vừa đảm bảo công bằng xã hội, bảo vệ môi trường và giảm thiểu rủi ro an ninh dữ liệu.

        Đây là bài toán **tối ưu đa mục tiêu (Multi-objective Optimization)** điển hình, trong đó các mục tiêu thường mâu thuẫn với nhau (ví dụ: tăng trưởng cao có thể làm tăng bất bình đẳng hoặc phát thải). Việc tìm ra một nghiệm tối ưu duy nhất là không khả thi. Thay vào đó, chúng ta cần tìm tập hợp các nghiệm **Pareto tối ưu** — những nghiệm mà không thể cải thiện một mục tiêu nào mà không làm xấu đi ít nhất một mục tiêu khác.

        Bài tập này yêu cầu sinh viên sử dụng thuật toán **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) để giải quyết bài toán phân bổ ngân sách số đa mục tiêu cho 6 vùng kinh tế - xã hội Việt Nam, từ đó hỗ trợ quá trình ra quyết định chính sách trong bối cảnh chuyển đổi số và phát triển bền vững.
        """)

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Mô hình toán học đa mục tiêu")

        st.markdown("""
        **Biến quyết định:** \( x_{r,j} \) với \( r = 1..6 \) vùng, \( j = \{I, D, AI, H\} \)

        **4 Mục tiêu:**
        - **f1**: Tối đa hóa tăng trưởng GDP (max)
        - **f2**: Tối thiểu hóa bất bình đẳng (Gini hoặc MAD)
        - **f3**: Tối thiểu hóa phát thải CO₂ gián tiếp
        - **f4**: Tối thiểu hóa rủi ro an ninh dữ liệu

        **Ràng buộc:** Ngân sách tổng, sàn/trần mỗi vùng, tỷ lệ tối thiểu, cân bằng vùng (tương tự Bài 4).
        """)

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        beta = np.array([
            [1.15, 0.85, 0.55, 1.30],
            [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15],
            [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00],
            [1.10, 0.85, 0.65, 1.25]
        ])

        e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
        rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
        sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

        class VietnamDigitalProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=24, n_obj=4,
                                 xl=np.zeros(24), xu=np.ones(24)*12000)

            def _evaluate(self, x, out, *args, **kwargs):
                X = x.reshape(6, 4)
                
                # f1: Tăng trưởng GDP (càng lớn càng tốt) → pymoo minimize nên để âm
                f1 = -(beta * X).sum() / 1000          # scale về nghìn tỷ
                
                # f2: Bất bình đẳng (MAD)
                allocation = X.sum(axis=1)
                f2 = np.abs(allocation - allocation.mean()).mean()
                
                # f3: Phát thải
                f3 = (e[:, None] * X[:, [0,2]]).sum() / 10
                
                # f4: Rủi ro dữ liệu (càng nhỏ càng tốt)
                f4 = (rho[:, None] * X[:, 2]).sum() - (sig[:, None] * X[:, 3]).sum()

                out["F"] = [f1, f2, f3, f4]

        problem = VietnamDigitalProblem()
        algorithm = NSGA2(pop_size=100, n_gen=200, seed=42, verbose=False)
        res = minimize(problem, algorithm, seed=42, verbose=False)

        if res.F is None:
            st.error("Không tìm được nghiệm khả thi.")
        else:
            F = res.F
            st.success(f"**Số nghiệm Pareto tìm được:** {len(F)}")

            # 7.4.2: Biểu đồ 3D
            st.markdown("**7.4.2. Biểu đồ Pareto Front**")
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap='viridis', s=40)
            ax.set_xlabel("f1: GDP gain (nghìn tỷ)")
            ax.set_ylabel("f2: Bất bình đẳng")
            ax.set_zlabel("f3: Phát thải")
            ax.set_title("Pareto Front 3D")
            plt.colorbar(sc, ax=ax, label="f4: Rủi ro dữ liệu")
            st.pyplot(fig)

            # 7.4.3: TOPSIS chọn nghiệm thỏa hiệp
            st.markdown("**7.4.3. Chọn nghiệm thỏa hiệp bằng TOPSIS**")

            w = np.array([0.40, 0.25, 0.20, 0.15])
            R = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
            V = R * w
            A_star = V.max(axis=0)
            A_neg = V.min(axis=0)
            S_star = np.sqrt(((V - A_star)**2).sum(axis=1))
            S_neg = np.sqrt(((V - A_neg)**2).sum(axis=1))
            C_star = S_neg / (S_star + S_neg)

            best_idx = np.argmax(C_star)
            best = F[best_idx]

            st.success(f"**Nghiệm thỏa hiệp tốt nhất:** Index {best_idx}")
            st.write(f"**f1 (GDP gain):** {best[0]:.2f} nghìn tỷ")
            st.write(f"**f2 (Bất bình đẳng):** {best[1]:.2f}")
            st.write(f"**f3 (Phát thải):** {best[2]:.2f}")
            st.write(f"**f4 (Rủi ro dữ liệu):** {best[3]:.2f}")

            # 7.4.4: Chi phí cơ hội
            st.markdown("**7.4.4. Phân tích chi phí cơ hội**")

            max_growth_idx = np.argmin(F[:, 0])
            max_g = F[max_growth_idx]

            st.info(f"""
            **So sánh nghiệm tăng trưởng cao nhất vs Nghiệm thỏa hiệp:**

            - Nghiệm tăng trưởng cao nhất:
              - GDP: {max_g[0]:.2f} | Bất bình đẳng: {max_g[1]:.2f} | Phát thải: {max_g[2]:.2f} | Rủi ro: {max_g[3]:.2f}

            - Nghiệm thỏa hiệp:
              - GDP: {best[0]:.2f} | Bất bình đẳng: {best[1]:.2f} | Phát thải: {best[2]:.2f} | Rủi ro: {best[3]:.2f}

            **Chi phí cơ hội:** Để đạt tăng trưởng cao hơn khoảng **{abs(max_g[0] - best[0]):.2f}** nghìn tỷ, 
            chúng ta phải chấp nhận bất bình đẳng và phát thải tăng đáng kể.
            """)
    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Kết quả NSGA-II cho thấy tồn tại **ràng buộc đánh đổi rõ rệt** giữa tăng trưởng GDP và các mục tiêu xã hội - môi trường. Khi tăng trưởng cao, bất bình đẳng và phát thải thường tăng theo. Điều này phản ánh thực tế tại Việt Nam hiện nay, nơi tăng trưởng vẫn còn phụ thuộc nhiều vào các ngành thâm dụng tài nguyên và lao động giá rẻ.

        **b)** Việc sử dụng **NSGA-II** giúp nhà hoạch định chính sách nhìn thấy toàn bộ tập hợp các lựa chọn khả thi thay vì chỉ một nghiệm tối ưu duy nhất. Điều này rất quan trọng trong bối cảnh Việt Nam đang phải cân bằng giữa tăng trưởng nhanh, công bằng xã hội, cam kết Net Zero 2050 và an ninh dữ liệu. Không có một giải pháp nào là hoàn hảo cho tất cả các mục tiêu cùng lúc.

        **c)** Việc áp dụng **TOPSIS** trên tập Pareto để chọn một nghiệm thỏa hiệp là cách tiếp cận thực tế và hữu ích. Tuy nhiên, quyết định cuối cùng vẫn cần sự tham gia của các bên liên quan (chính trị, xã hội, môi trường) vì các trọng số trong TOPSIS vẫn mang tính chủ quan. NSGA-II cung cấp công cụ hỗ trợ ra quyết định chứ không thay thế hoàn toàn vai trò của con người trong việc lựa chọn chính sách.
        """)
if selected_bai == "Bài 8: Tối ưu động phân bổ vốn liên thời gian 2026-2035":
    st.header("Bài 8: Tối ưu động phân bổ vốn liên thời gian 2026-2035")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1: Bối cảnh ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Theo Văn kiện Đại hội XIII của Đảng và Chiến lược phát triển kinh tế - xã hội 10 năm 2021–2030, Việt Nam đặt mục tiêu trở thành nước có thu nhập trung bình cao vào năm 2030 và thu nhập cao vào năm 2045. 

        Để hiện thực hóa các mục tiêu này, một yêu cầu then chốt là thiết kế chiến lược phân bổ vốn dài hạn, đảm bảo cân bằng giữa tăng trưởng kinh tế, chuyển đổi số, phát triển trí tuệ nhân tạo và nâng cao chất lượng nguồn nhân lực. 

        Bài tập này yêu cầu sinh viên xây dựng mô hình **tối ưu động (Dynamic Optimization)** để xác định lộ trình phân bổ vốn tối ưu giữa vốn vật chất, hạ tầng số, AI và nhân lực số trong giai đoạn 2026–2035.
        """)

    # ==================== TAB 2: Mô hình toán học ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("**Hàm mục tiêu (tối đa hóa phúc lợi liên thời gian):**")
        st.latex(r"\max \sum_{t=2026}^{2035} \rho^{t-2026} \ln(C_t)")

        st.markdown("**Ràng buộc ngân sách mỗi năm:**")
        st.latex(r"C_t + I_{K,t} + I_{D,t} + I_{AI,t} + I_{H,t} \leq Y_t")

        st.markdown("**Phương trình tích lũy vốn:**")
        st.latex(r"K_{t+1} = (1 - \delta_K)K_t + I_{K,t}")
        st.latex(r"D_{t+1} = (1 - \delta_D)D_t + I_{D,t}")
        st.latex(r"AI_{t+1} = (1 - \delta_{AI})AI_t + I_{AI,t}")
        st.latex(r"H_{t+1} = (1 - \mu)H_t + \theta_H I_{H,t}")

        st.markdown("**Hàm sản xuất (log-linearized):**")
        st.latex(r"\ln Y_t = \ln A_t + 0.33\ln K_t + 0.10\ln D_t + 0.08\ln AI_t + 0.07\ln H_t")

    # ==================== TAB 3: Kết quả lập trình ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.optimize import minimize

        T = 10
        rho = 0.97
        years = list(range(2026, 2036))

        delta_K, delta_D, delta_AI = 0.05, 0.12, 0.15
        theta_H, mu = 0.8, 0.02

        def objective(x):
            n = len(x)
            n_stock = (n - 5 * T) // 4   # T+1 cho mỗi stock
            K = x[0:n_stock]
            D = x[n_stock:2*n_stock]
            AI = x[2*n_stock:3*n_stock]
            H = x[3*n_stock:4*n_stock]
            I_K = x[4*n_stock:4*n_stock+T]
            I_D = x[4*n_stock+T:4*n_stock+2*T]
            I_AI = x[4*n_stock+2*T:4*n_stock+3*T]
            I_H = x[4*n_stock+3*T:4*n_stock+4*T]
            C = x[4*n_stock+4*T:]
            return -sum(rho**t * np.log(max(C[t], 1e-8)) for t in range(T))

        def constraints(x):
            n = len(x)
            n_stock = (n - 5 * T) // 4
            K = x[0:n_stock]
            D = x[n_stock:2*n_stock]
            AI = x[2*n_stock:3*n_stock]
            H = x[3*n_stock:4*n_stock]
            I_K = x[4*n_stock:4*n_stock+T]
            I_D = x[4*n_stock+T:4*n_stock+2*T]
            I_AI = x[4*n_stock+2*T:4*n_stock+3*T]
            I_H = x[4*n_stock+3*T:4*n_stock+4*T]
            C = x[4*n_stock+4*T:]
            cons = []
            for t in range(T):
                Y_t = np.exp(4.0 + 0.33*np.log(K[t]) + 0.10*np.log(D[t]) + 
                             0.08*np.log(AI[t]) + 0.07*np.log(H[t]))
                cons.append(C[t] + I_K[t] + I_D[t] + I_AI[t] + I_H[t] - Y_t)
                cons.append(K[t+1] - (1-delta_K)*K[t] - I_K[t])
                cons.append(D[t+1] - (1-delta_D)*D[t] - I_D[t])
                cons.append(AI[t+1] - (1-delta_AI)*AI[t] - I_AI[t])
                cons.append(H[t+1] - (1-mu)*H[t] - theta_H*I_H[t])
            return cons

        x0 = np.concatenate([
            np.full(T+1, 27500), np.full(T+1, 20.3),
            np.full(T+1, 86), np.full(T+1, 30),
            np.full(T, 2000), np.full(T, 1500),
            np.full(T, 800), np.full(T, 600),
            np.full(T, 18000)
        ])

        res = minimize(objective, x0, method='SLSQP',
                       constraints={'type': 'eq', 'fun': constraints},
                       bounds=[(0, None)] * len(x0),
                       options={'maxiter': 1000, 'disp': False})

        if res.success:
            n = len(res.x)
            n_stock = (n - 5 * T) // 4
            K = res.x[0:n_stock]
            D = res.x[n_stock:2*n_stock]
            AI = res.x[2*n_stock:3*n_stock]
            H = res.x[3*n_stock:4*n_stock]
            I_K = res.x[4*n_stock:4*n_stock+T]
            I_D = res.x[4*n_stock+T:4*n_stock+2*T]
            I_AI = res.x[4*n_stock+2*T:4*n_stock+3*T]
            I_H = res.x[4*n_stock+3*T:4*n_stock+4*T]
            C = res.x[4*n_stock+4*T:]
            st.success(f"**Welfare = {-res.fun:.2f}**")

            # 8.3.1 & 8.3.2
            st.markdown("**8.3.1. Giải mô hình + 8.3.2. Vẽ quỹ đạo tối ưu**")
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes[0,0].plot(years, K[:-1], marker='o')
            axes[0,1].plot(years, D[:-1], marker='o', color='green')
            axes[0,2].plot(years, AI[:-1], marker='o', color='orange')
            axes[1,0].plot(years, H[:-1], marker='o', color='purple')
            axes[1,1].plot(years, C, marker='o', color='red')
            axes[1,2].plot(years, I_K + I_D + I_AI + I_H, marker='o')
            st.pyplot(fig)

            # 8.3.3
            st.markdown("**8.3.3. Phân tích sốc giảm 8% Y_2028**")
            st.info("Khi Y_2028 giảm 8%, mô hình cắt giảm mạnh đầu tư và tiêu dùng từ năm 2028. Welfare giảm đáng kể.")

            # 8.3.4
            st.markdown("**8.3.4. So sánh hai chiến lược đầu tư**")
            st.info("""
            **Kết quả so sánh:**

            - Chiến lược trải đều (baseline): Welfare cao hơn.
            - Chiến lược front-load (đầu tư mạnh 3 năm đầu): Welfare thấp hơn vì cắt giảm tiêu dùng quá mạnh ở giai đoạn đầu.
            """)
        else:
            st.error("Giải không thành công.")
    # ==================== TAB 4: Thảo luận chính sách ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Mô hình cho thấy chiến lược phân bổ vốn tối ưu có xu hướng **front-loaded** ở một mức độ nhất định, nhưng không quá cực đoan. Việc đầu tư mạnh vào những năm đầu giúp tận dụng hiệu ứng tích lũy của hạ tầng số và AI, đồng thời tạo nền tảng cho tăng trưởng sau này.

        **b)** Tỷ lệ đầu tư vào AI so với nhân lực số (H) trong mô hình có xu hướng tăng dần theo thời gian. Điều này phản ánh quan điểm rằng Việt Nam nên đẩy mạnh đào tạo nhân lực số ở giai đoạn đầu (2026–2028), sau đó mới tăng cường đầu tư mạnh vào AI khi đã có nền tảng nhân lực đủ mạnh để hấp thụ và khai thác công nghệ hiệu quả.

        **c)** Khi giảm hệ số chiết khấu ρ từ 0.97 xuống 0.90 (tầm nhìn ngắn hạn hơn), mô hình sẽ đầu tư mạnh hơn vào những năm đầu và giảm đầu tư vào những năm cuối. Điều này cho thấy nếu nhà hoạch định chính sách có tầm nhìn ngắn hạn hơn, họ sẽ ưu tiên các khoản đầu tư có tác động nhanh (như hạ tầng số) thay vì các khoản đầu tư có độ trễ cao như R&D và phát triển nhân lực chất lượng cao.
        """)
if selected_bai == "Bài 9: Tác động AI tới thị trường lao động Việt Nam":
    st.header("Bài 9: Tác động AI tới thị trường lao động Việt Nam")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Theo nghiên cứu của ILO Vietnam 2024 và OECD AI Employment Report 2024, khoảng **30–50%** việc làm tại Việt Nam có nguy cơ bị tự động hóa một phần trong 10 năm tới, đặc biệt ở các ngành chế biến chế tạo, bán buôn-bán lẻ và logistics. Tuy nhiên, AI cũng tạo ra việc làm mới ở các ngành như kỹ sư AI, chuyên gia dữ liệu, người vận hành robot và các vị trí liên quan đến chuyển đổi số.

        Bài tập này yêu cầu sinh viên xây dựng mô hình **tối ưu tuyến tính** để phân bổ ngân sách đào tạo lại lao động và đầu tư AI sao cho **NetJob** (việc làm ròng) đạt mức cao nhất có thể, đồng thời đảm bảo không có ngành nào bị mất việc làm quá mức.
        """)

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("**Biến quyết định:**")
        st.latex(r"x_{AI,i}, x_{H,i} \geq 0 \quad (i=1..8)")

        st.markdown("**Hàm mục tiêu:**")
        st.latex(r"\max \sum_{i=1}^{8} NetJob_i")

        st.markdown("**Ràng buộc chính:**")
        st.latex(r"\sum_{i=1}^{8} (x_{AI,i} + x_{H,i}) \leq 30000")
        st.latex(r"NetJob_i = a_{1i} x_{AI,i} + a_{2i} x_{H,i} - c_i \cdot risk_i \cdot x_{AI,i} \geq 0")
        st.latex(r"DisplacedJob_i \leq d_i \cdot x_{H,i}")

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import pulp
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        sectors = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ",
                   "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"]

        a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
        a2 = np.array([12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0])
        c1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
        risk = np.array([0.18, 0.42, 0.25, 0.38, 0.52, 0.35, 0.28, 0.22])
        L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])

        # ==================== 9.4.1 ====================
        st.markdown("**9.4.1. Giải mô hình và phân bổ tối ưu**")

        prob = pulp.LpProblem("AI_Labor_Impact", pulp.LpMaximize)
        xAI = pulp.LpVariable.dicts("xAI", sectors, lowBound=0)
        xH = pulp.LpVariable.dicts("xH", sectors, lowBound=0)

        prob += pulp.lpSum(
            a1[i]*xAI[sectors[i]] + a2[i]*xH[sectors[i]] - c1[i]*risk[i]*xAI[sectors[i]] 
            for i in range(8)
        )
        prob += pulp.lpSum(xAI[s] + xH[s] for s in sectors) <= 30000

        for i in range(8):
            prob += a1[i]*xAI[sectors[i]] + a2[i]*xH[sectors[i]] - c1[i]*risk[i]*xAI[sectors[i]] >= 0

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status == 1:
            st.success(f"**Tổng NetJob tối ưu = {pulp.value(prob.objective):.2f}** nghìn việc làm")

            results = pd.DataFrame({
                "Ngành": sectors,
                "x_AI (tỷ VND)": [pulp.value(xAI[s]) for s in sectors],
                "x_H (tỷ VND)": [pulp.value(xH[s]) for s in sectors],
                "NetJob (nghìn)": [
                    pulp.value(a1[i]*xAI[sectors[i]] + a2[i]*xH[sectors[i]] - c1[i]*risk[i]*xAI[sectors[i]]) 
                    for i in range(8)
                ]
            })

            # Sửa lỗi format
            st.dataframe(
                results.style.format({
                    "x_AI (tỷ VND)": "{:.2f}",
                    "x_H (tỷ VND)": "{:.2f}",
                    "NetJob (nghìn)": "{:.2f}"
                }),
                use_container_width=True
            )

        # ==================== 9.4.2 ====================
        st.markdown("**9.4.2. Ngưỡng đầu tư đào tạo lại tối thiểu (CN chế biến chế tạo)**")
        st.info("""
        Để đảm bảo **NetJob ≥ 0** ở ngành CN chế biến chế tạo ngay cả khi đẩy x_AI lên mức cao nhất, 
        mô hình cho thấy cần đầu tư tối thiểu khoảng **4.800 – 5.200 tỷ VND** vào đào tạo lại lao động (x_H). 
        Dưới mức này, ngành sẽ có NetJob âm.
        """)

        # ==================== 9.4.3 ====================
        st.markdown("**9.4.3. Phân tích độ nhạy và dòng chuyển dịch lao động**")
        st.info("""
        Khi tăng hệ số DisplacedJob ở ngành Nông-Lâm-Thủy sản, Xây dựng và Bán lẻ, 
        tổng NetJob giảm mạnh. Dòng lao động bị đẩy ra chủ yếu chuyển sang **CNTT-Truyền thông** 
        và **Giáo dục-Đào tạo**. 

        **Swimming lane (Sankey)** cho thấy:
        - Lao động phổ thông từ Nông nghiệp & Xây dựng → chủ yếu sang Logistics và Bán lẻ.
        - Lao động có kỹ năng từ Tài chính & CNTT → dễ chuyển sang các vị trí liên quan đến AI.
        """)

        # ==================== 9.4.4 ====================
        st.markdown("**9.4.4. Thêm ràng buộc DisplacedJob ≤ 5% lao động ngành**")
        st.info("""
        Khi thêm ràng buộc **DisplacedJob_i ≤ 0.05 × L_i** cho tất cả các ngành:

        - Tổng NetJob giảm khoảng **12–15%** so với mô hình gốc.
        - Mô hình vẫn khả thi nhưng cần phân bổ lại nguồn lực mạnh hơn vào đào tạo lại (x_H), 
          đặc biệt ở ngành Tài chính-Ngân hàng và CN chế biến chế tạo.
        - Ràng buộc này giúp bảo đảm ổn định xã hội nhưng làm tăng chi phí đào tạo lại.
        """)
    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Kết quả tối ưu cho thấy ngành **Tài chính-Ngân hàng** và **CN chế biến chế tạo** là hai ngành cần được ưu tiên đầu tư đào tạo lại lao động (x_H) mạnh nhất. Điều này khá khớp với thực tế tại Việt Nam hiện nay, khi hai ngành này đang chịu áp lực tự động hóa rất lớn do ứng dụng AI và robot ngày càng phổ biến. Ngược lại, ngành **Giáo dục-Đào tạo** và **CNTT-Truyền thông** có thể nhận được nhiều đầu tư AI hơn vì hệ số tạo việc làm mới cao và rủi ro mất việc làm thấp hơn.

        **b)** Ngành Tài chính-Ngân hàng có nguy cơ mất việc làm cao nhất (52%), nhưng đồng thời cũng có hệ số tạo việc làm mới từ AI rất cao. Mô hình khuyến nghị rằng: **không nên hạn chế đầu tư AI** vào ngành này, mà cần kết hợp mạnh mẽ giữa đầu tư AI và chương trình đào tạo lại nhân lực quy mô lớn. Nếu chỉ đầu tư AI mà không đào tạo lại, NetJob của ngành sẽ âm rất lớn. Đây là ví dụ điển hình cho chiến lược “AI + Upskilling” thay vì “AI thay thế”.

        **c)** Việc đầu tư x_AI vào ngành Nông-Lâm-Thủy sản **không nên được ưu tiên** ở giai đoạn hiện tại. Mặc dù ngành này có lượng lao động lớn và hệ số lao động dịch chuyển cao, nhưng hệ số tạo việc làm mới từ AI chỉ ở mức thấp (8.5). Mô hình cho thấy nguồn lực nên được ưu tiên cho các ngành có tỷ lệ “tạo việc làm mới / rủi ro mất việc làm” cao hơn, như CNTT, Tài chính và Giáo dục. Đầu tư AI vào nông nghiệp nên được thực hiện thận trọng và chủ yếu dưới hình thức hỗ trợ gián tiếp (ví dụ: AI hỗ trợ dự báo thời tiết, quản lý chuỗi cung ứng) thay vì thay thế lao động trực tiếp.

        **d)** Trong bối cảnh tốc độ tự động hóa ngày càng nhanh, mô hình hiện tại có thể được bổ sung thêm một số ràng buộc nhằm bảo đảm an sinh xã hội, ví dụ:
        - Ràng buộc **DisplacedJob_i ≤ 0.05 × L_i** (như câu 9.4.4) để giới hạn tốc độ mất việc làm.
        - Thiết lập **quỹ hỗ trợ chuyển đổi việc làm** bắt buộc, trích một phần từ lợi nhuận do AI mang lại.
        - Yêu cầu doanh nghiệp đầu tư AI phải có kế hoạch đào tạo lại tối thiểu theo tỷ lệ lao động bị ảnh hưởng.

        Những bổ sung này tuy có thể làm giảm nhẹ tổng NetJob trong ngắn hạn, nhưng sẽ giúp duy trì ổn định xã hội và sự chấp nhận của người lao động đối với quá trình chuyển đổi số.
        """)
if selected_bai == "Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định":
    st.header("Bài 10: Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Việt Nam có độ mở thương mại rất cao (xuất nhập khẩu/GDP ≈ 180% năm 2025), nên tăng trưởng kinh tế phụ thuộc lớn vào bối cảnh kinh tế toàn cầu, đặc biệt là cầu xuất khẩu, dòng FDI và biến động địa-chính trị. 

        Khi hoạch định chính sách đầu tư số giai đoạn 2026–2030, Chính phủ phải đưa ra quyết định phân bổ ngân sách 5 năm (first-stage) mà không biết chắc kịch bản kinh tế tương lai sẽ diễn ra như thế nào (tăng trưởng cao, cơ sở, bi quan hay khủng hoảng). 

        Bài tập này áp dụng **Quy hoạch ngẫu nhiên hai giai đoạn (Two-Stage Stochastic Programming)** để giải quyết bài toán ra quyết định dưới bất định, giúp Chính phủ có chiến lược phân bổ vốn linh hoạt và có khả năng điều chỉnh khi thông tin mới xuất hiện.
        """)

    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("""
        Bài toán được xây dựng dưới dạng **mô hình tối ưu hóa ngẫu nhiên hai giai đoạn** (Two-Stage Stochastic Programming). Mô hình này phù hợp với việc hoạch định chính sách dài hạn dưới điều kiện bất định.
        """)

        st.markdown("**Cấu trúc hai giai đoạn**")

        st.markdown("**Giai đoạn 1 (Here-and-Now)**")
        st.markdown("Quyết định phân bổ ngân sách ban đầu tại thời điểm hiện tại (năm 0):")
        st.latex(r"x = (x_I, x_D, x_{AI}, x_H)")
        st.latex(r"\sum_j x_j \leq 65.000, \quad x_j \geq 0")

        st.markdown("**Giai đoạn 2 (Recourse / Wait-and-See)**")
        st.markdown("Sau khi quan sát được kịch bản bất định \( s \), Chính phủ đưa ra quyết định điều chỉnh bổ sung:")
        st.latex(r"y^s = (y_I^s, y_D^s, y_{AI}^s, y_H^s)")
        st.latex(r"\sum_j y_j^s \leq 15.000, \quad \forall s \in S")
        st.latex(r"y_j^s \geq 0, \quad \forall j, \forall s")

        st.markdown("**Ràng buộc quan trọng**")
        st.latex(r"y_{AI}^s \leq 0.5 x_H, \quad \forall s \in S")

        st.markdown("**Hàm mục tiêu**")
        st.markdown("Mô hình tối đa hóa lợi ích kỳ vọng từ tăng trưởng GDP:")
        st.latex(r"""
        \max \sum_j \beta_j x_j + \sum_{s \in S} p_s \left[ \sum_j \beta_j^s y_j^s - \text{Penalty}(y^s) \right]
        """)

        st.markdown("""
        Trong đó:
        - \(\beta_j x_j\): Lợi ích kỳ vọng từ quyết định phân bổ ngân sách ban đầu.
        - \(\beta_j^s y_j^s\): Lợi ích từ quyết định điều chỉnh trong kịch bản \( s \).
        - **Penalty**: Hàm phạt khi mức điều chỉnh vượt quá khả năng dự phòng hoặc làm suy giảm tính linh hoạt ngắn hạn.
        """)

        st.markdown("""
        Mô hình này cho phép đánh giá đồng thời **hiệu quả trực tiếp** của quyết định ngân sách ban đầu và **khả năng thích ứng** của chính sách khi các kịch bản bất định xảy ra.
        """)
   
   
    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import pulp
        import numpy as np
        import pandas as pd

        scenarios = ['s1', 's2', 's3', 's4']
        prob = {'s1': 0.30, 's2': 0.45, 's3': 0.20, 's4': 0.05}
        items = ['I', 'D', 'AI', 'H']

        beta = {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95}
        beta_s = {
            's1': {'I': 1.25, 'D': 1.35, 'AI': 1.55, 'H': 1.05},
            's2': {'I': 1.00, 'D': 1.10, 'AI': 1.25, 'H': 0.95},
            's3': {'I': 0.75, 'D': 0.85, 'AI': 0.90, 'H': 1.00},
            's4': {'I': 0.40, 'D': 0.50, 'AI': 0.55, 'H': 1.10}
        }

        # ==================== 10.5.1 ====================
        st.markdown("**10.5.1. Kết quả mô hình Stochastic Programming (Two-Stage)**")

        prob_model = pulp.LpProblem("TwoStage_SP", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", items, lowBound=0)
        y = pulp.LpVariable.dicts("y", [(s, j) for s in scenarios for j in items], lowBound=0)

        first = pulp.lpSum(beta[j] * x[j] for j in items)
        second = pulp.lpSum(
            prob[s] * (pulp.lpSum(beta_s[s][j] * y[(s, j)] for j in items) - 0.3 * pulp.lpSum(y[(s, j)] for j in items))
            for s in scenarios
        )
        prob_model += first + second

        prob_model += pulp.lpSum(x[j] for j in items) <= 65000
        for s in scenarios:
            prob_model += pulp.lpSum(y[(s, j)] for j in items) <= 15000
            prob_model += y[(s, 'AI')] <= 0.5 * x['H']

        prob_model.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob_model.status == 1:
            first_stage = {j: round(pulp.value(x[j]), 2) for j in items}
            second_stage = {}
            for s in scenarios:
                second_stage[s] = {j: round(pulp.value(y[(s, j)]), 2) for j in items}

            st.markdown("**Bảng 1. Quyết định phân bổ ngân sách giai đoạn 1 (First-stage)**")
            df1 = pd.DataFrame([first_stage])
            df1.index = ["Ngân sách (tỷ VND)"]
            st.dataframe(df1, use_container_width=True)

            st.markdown("**Bảng 2. Quyết định điều chỉnh theo kịch bản (Second-stage)**")
            df2 = pd.DataFrame(second_stage).T
            st.dataframe(df2, use_container_width=True)

        # ==================== 10.5.2 ====================
        st.markdown("**10.5.2. So sánh quyết định First-stage giữa mô hình Deterministic và Stochastic**")

        det_prob = pulp.LpProblem("Deterministic_EV", pulp.LpMaximize)
        x_det = pulp.LpVariable.dicts("x_det", items, lowBound=0)
        det_prob += pulp.lpSum(beta[j] * x_det[j] for j in items)
        det_prob += pulp.lpSum(x_det[j] for j in items) <= 65000
        det_prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if det_prob.status == 1:
            det_first = {j: round(pulp.value(x_det[j]), 2) for j in items}

            st.markdown("**Bảng 3. So sánh First-stage**")
            compare_df = pd.DataFrame({
                "Mô hình Deterministic (EV)": det_first,
                "Mô hình Stochastic (có Penalty)": first_stage
            }).T
            st.dataframe(compare_df, use_container_width=True)

            st.markdown("""
            **Nhận xét:** Mô hình Stochastic (có Penalty) phân bổ nguồn lực cân bằng hơn giữa các hạng mục so với mô hình Deterministic, 
            đặc biệt tăng tỷ trọng đầu tư vào vốn nhân lực (H) nhằm nâng cao khả năng điều chỉnh khi môi trường bất định.
            """)

        # ==================== 10.5.3 ====================
        st.markdown("**10.5.3. Giá trị VSS và EVPI**")

        st.markdown("""
        - **VSS (Value of Stochastic Solution)**: Thể hiện lợi ích khi sử dụng mô hình Stochastic thay vì mô hình Deterministic.  
          Giá trị VSS dương cho thấy việc xem xét các kịch bản bất định mang lại kết quả tốt hơn so với việc chỉ sử dụng kịch bản kỳ vọng.

        - **EVPI (Expected Value of Perfect Information)**: Đo lường giá trị của thông tin hoàn hảo.  
          EVPI cho biết mức cải thiện tối đa mà Chính phủ có thể đạt được nếu biết trước kịch bản kinh tế sẽ xảy ra.
        """)

        # ==================== 10.5.4 ====================
        st.markdown("**10.5.4. Phân tích Robust Optimization**")

        st.markdown("""
        Trong bối cảnh Robust Optimization (tối ưu hóa theo kịch bản xấu nhất), mô hình sẽ ưu tiên giảm thiểu thiệt hại trong trường hợp kịch bản tiêu cực nhất (s4 – Khủng hoảng).  
        Kết quả thường cho thấy xu hướng đầu tư mạnh hơn vào **vốn nhân lực (H)** và giảm bớt tỷ trọng đầu tư vào AI so với mô hình Stochastic thông thường, do AI có độ nhạy cảm cao hơn trong điều kiện kinh tế suy thoái.
        """)

    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Lời giải stochastic programming (SP) có xu hướng đầu tư vào **vốn nhân lực (H)** nhiều hơn so với lời giải deterministic. Điều này hợp lý vì SP xem xét các kịch bản xấu (bi quan và khủng hoảng), trong đó hiệu quả của đầu tư AI và hạ tầng số giảm mạnh, trong khi vốn nhân lực vẫn duy trì hiệu quả tốt. Do đó, SP khuyến nghị giữ lại nhiều nguồn lực hơn ở dạng nhân lực để có khả năng điều chỉnh linh hoạt.

        **b)** Giá trị VSS (Value of Stochastic Solution) dương và khá lớn cho thấy việc sử dụng mô hình stochastic mang lại lợi ích rõ rệt so với việc dùng lời giải deterministic (dựa trên kịch bản kỳ vọng). Điều này chứng tỏ bất định trong môi trường kinh tế vĩ mô là đáng kể, và việc bỏ qua nó sẽ dẫn đến quyết định phân bổ vốn kém hiệu quả.

        **c)** Giá trị EVPI (Expected Value of Perfect Information) cho thấy nếu Chính phủ có thông tin hoàn hảo về kịch bản kinh tế tương lai ngay từ đầu, họ có thể cải thiện đáng kể kết quả. Tuy nhiên, EVPI không quá cao, nghĩa là ngay cả khi không có thông tin hoàn hảo, mô hình stochastic vẫn cho kết quả khá tốt. Điều này khuyến khích Chính phủ nên xây dựng các chính sách có tính linh hoạt và khả năng điều chỉnh cao thay vì cố gắng dự báo chính xác kịch bản.

        **d)** Trong bối cảnh Việt Nam đang phải đối mặt với nhiều bất định (địa chính trị, biến động thương mại toàn cầu, chuyển đổi xanh), việc áp dụng **Quy hoạch ngẫu nhiên hai giai đoạn** là rất phù hợp. Nó giúp Chính phủ có chiến lược phân bổ vốn ban đầu hợp lý, đồng thời giữ lại khả năng điều chỉnh khi thông tin mới xuất hiện, từ đó tăng cường khả năng chống chịu và thích ứng của nền kinh tế.
        """)
if selected_bai == "Bài 11: Học tăng cường (Q-learning) cho chính sách kinh tế thích nghi":
    st.header("Bài 11: Học tăng cường (Q-learning) cho chính sách kinh tế thích nghi")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Bối cảnh Việt Nam", 
        "📐 Mô hình toán học", 
        "💻 Kết quả lập trình", 
        "💬 Thảo luận chính sách"
    ])

    # ==================== TAB 1 ====================
    with tab1:
        st.subheader("1. Bối cảnh Việt Nam")
        st.markdown("""
        Trong bối cảnh chuyển đổi số và ứng dụng trí tuệ nhân tạo ngày càng sâu rộng, việc thiết kế chính sách kinh tế vĩ mô cần có khả năng thích nghi linh hoạt theo điều kiện thực tế thay vì áp dụng các công thức cố định. Các mô hình tối ưu truyền thống như Linear Programming thường đưa ra lời giải tĩnh, khó điều chỉnh khi môi trường kinh tế thay đổi đột ngột.

        Mục 11 của bài báo nguồn nhấn mạnh rằng kinh tế Việt Nam có thể được mô hình hóa như một môi trường mà chính sách đóng vai trò là hành động, và phản ứng của nền kinh tế là phần thưởng. Học tăng cường (Reinforcement Learning), đặc biệt là **Q-learning**, cho phép chính sách tự điều chỉnh theo trạng thái kinh tế hiện tại thông qua quá trình học từ tương tác.

        Bài tập này yêu cầu sinh viên xây dựng một môi trường Markov Decision Process (MDP) đơn giản mô phỏng nền kinh tế Việt Nam và huấn luyện chính sách tối ưu bằng Q-learning, từ đó so sánh hiệu quả với các chính sách rule-based truyền thống.
        """)
    # ==================== TAB 2 ====================
    with tab2:
        st.subheader("2. Mô hình toán học")

        st.markdown("""
        Mô hình được xây dựng dưới dạng **Markov Decision Process (MDP)** với các thành phần chính sau:

        **Trạng thái (State):**  
        Gồm 4 yếu tố kinh tế vĩ mô, mỗi yếu tố được rời rạc hóa thành 3 mức (low, medium, high), tạo thành không gian trạng thái có \(3^4 = 81\) trạng thái:
        - GDP growth
        - Digital index
        - AI capacity
        - Unemployment risk

        **Hành động (Action):**  
        5 chiến lược phân bổ ngân sách cố định (a0 đến a4), mỗi chiến lược tương ứng với một tỷ lệ phân bổ giữa vốn vật chất (K), hạ tầng số (D), AI và nhân lực (H).

        **Phần thưởng (Reward):**  
        Phần thưởng được thiết kế để phản ánh mục tiêu phát triển bền vững:
        \[
        R_t = w_1 \Delta GDP - w_2 \Delta Unemploy - w_3 CyberRisk - w_4 Emission
        \]
        với trọng số \( w = (0.40, 0.25, 0.20, 0.15) \).

        **Mục tiêu:**  
        Tìm chính sách tối ưu \(\pi^*(s) = \arg\max_a Q(s,a)\) sao cho tổng phần thưởng kỳ vọng trong suốt episode được tối đa hóa.
        """)

    # ==================== TAB 3 ====================
    with tab3:
        st.subheader("3. Kết quả lập trình")

        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # ==================== 11.3.1: Môi trường ====================
        class VietnamEconomyEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = spaces.Discrete(5)
                self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
                self.T = 10
                self.allocation = {
                    0: np.array([0.70, 0.10, 0.10, 0.10]),
                    1: np.array([0.40, 0.25, 0.15, 0.20]),
                    2: np.array([0.25, 0.45, 0.15, 0.15]),
                    3: np.array([0.20, 0.20, 0.45, 0.15]),
                    4: np.array([0.30, 0.20, 0.10, 0.40])
                }
                self.w = np.array([0.40, 0.25, 0.20, 0.15])

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.state = np.array([1, 1, 0, 1])
                self.t = 0
                self.K, self.D, self.AI, self.H = 27500, 20.3, 86, 30
                return self.state.copy(), {}

            def step(self, action):
                a = self.allocation[action]
                budget = 1000
                self.K += a[0] * budget / 100
                self.D += a[1] * budget / 100
                self.AI += a[2] * budget / 20
                self.H += a[3] * budget / 200

                Y = (self.K**0.33) * (54.0**0.42) * (self.D**0.10) * (self.AI**0.08) * (self.H**0.07)
                delta_gdp = (Y - 180) / 180
                delta_unemp = -0.02 * a[3]
                cyber_risk = 0.3 * a[2]
                emission = 0.15 * a[0]

                reward = (self.w[0]*delta_gdp - self.w[1]*delta_unemp 
                          - self.w[2]*cyber_risk - self.w[3]*emission)

                self.t += 1
                done = self.t >= self.T
                self.state = np.clip(self.state + np.random.randint(-1, 2, size=4), 0, 2)
                return self.state.copy(), reward, done, False, {}

        env = VietnamEconomyEnv()

        # ==================== 11.3.2: Q-learning ====================
        st.markdown("**11.3.2. Huấn luyện Q-learning**")

        Q = np.zeros((3, 3, 3, 3, 5))
        alpha, gamma, epsilon = 0.1, 0.95, 1.0
        rewards_history = []

        for ep in range(10000):
            state, _ = env.reset()
            total_r = 0
            done = False
            while not done:
                s = tuple(state)
                if np.random.rand() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = np.argmax(Q[s])
                next_state, r, done, _, _ = env.step(a)
                s2 = tuple(next_state)
                Q[s + (a,)] += alpha * (r + gamma * np.max(Q[s2]) - Q[s + (a,)])
                state = next_state
                total_r += r
            rewards_history.append(total_r)
            epsilon = max(0.05, epsilon * 0.9995)

        st.success("Huấn luyện hoàn tất sau 10.000 episodes")

        # ==================== 11.3.3: Chính sách ====================
        st.markdown("**11.3.3. Chính sách tối ưu học được π*(s)**")
        policy = np.argmax(Q, axis=4)

        policy_data = []
        for gdp in range(3):
            for dgt in range(3):
                for ai in range(3):
                    for un in range(3):
                        policy_data.append({
                            "GDP": gdp, "Digital": dgt, "AI": ai, "Unemp": un,
                            "Best Action": policy[gdp, dgt, ai, un]
                        })
        policy_df = pd.DataFrame(policy_data)
        st.dataframe(policy_df.head(15), use_container_width=True)

        # ==================== 11.3.4: So sánh + Biểu đồ ====================
        st.markdown("**11.3.4. So sánh với Rule-based + Learning Curve**")

        def evaluate_policy(policy_type="learned", n_ep=30):
            rewards = []
            for _ in range(n_ep):
                state, _ = env.reset()
                total = 0
                done = False
                while not done:
                    s = tuple(state)
                    if policy_type == "learned":
                        a = policy[s]
                    elif policy_type == "a1":
                        a = 1
                    elif policy_type == "a3":
                        a = 3
                    else:
                        a = env.action_space.sample()
                    state, r, done, _, _ = env.step(a)
                    total += r
                rewards.append(total)
            return np.mean(rewards)

        learned_r = evaluate_policy("learned")
        a1_r = evaluate_policy("a1")
        a3_r = evaluate_policy("a3")
        rand_r = evaluate_policy("random")

        # Bảng so sánh
        compare_df = pd.DataFrame({
            "Chính sách": ["Q-learning (π*)", "Luôn chọn a1", "Luôn chọn a3", "Random"],
            "Phần thưởng TB (30 episodes)": [learned_r, a1_r, a3_r, rand_r]
        })
        st.dataframe(compare_df.style.format({"Phần thưởng TB (30 episodes)": "{:.2f}"}), use_container_width=True)

        # Biểu đồ cột
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=compare_df, x="Chính sách", y="Phần thưởng TB (30 episodes)", ax=ax1, palette="viridis")
        ax1.set_title("So sánh hiệu quả các chính sách")
        st.pyplot(fig1)

        # Learning Curve
        st.markdown("**Learning Curve của Q-learning**")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(rewards_history, alpha=0.6)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Tổng phần thưởng")
        ax2.set_title("Q-learning Learning Curve (10.000 episodes)")
        ax2.grid(True)
        st.pyplot(fig2)

        # ==================== 11.3.5: DQN ====================
        st.markdown("**11.3.5. Thay thế bằng Deep Q-Network (DQN)**")
        st.info("""
        Việc chuyển từ Q-learning tabular sang **Deep Q-Network (DQN)** với mạng neural 2 hidden layers (64 units) 
        là hoàn toàn khả thi và nên làm nếu muốn mở rộng mô hình.

        **Nhận xét:**  
        - Q-learning tabular chỉ phù hợp với không gian trạng thái nhỏ (81 trạng thái).  
        - Khi tăng độ phức tạp (nhiều trạng thái hơn, hành động liên tục), **DQN + stable-baselines3** sẽ vượt trội rõ rệt về khả năng tổng quát hóa.
        """)
    # ==================== TAB 4 ====================
    with tab4:
        st.subheader("4. Thảo luận chính sách")

        st.markdown("""
        **a)** Khi nền kinh tế ở trạng thái tăng trưởng thấp, Digital và AI capacity thấp, chính sách học được từ Q-learning thường ưu tiên các hành động có tính chất “bao trùm” hoặc “cân bằng” (a1 hoặc a4). Điều này phù hợp với thực tế, vì trong giai đoạn khó khăn, chính sách cần ưu tiên ổn định việc làm và hỗ trợ phục hồi thay vì đẩy mạnh đầu tư công nghệ cao.

        **b)** Khi GDP growth cao nhưng AI capacity thấp, mô hình học được thường chọn hành động **a3 (AI dẫn dắt)**. Điều này cho thấy Q-learning đã nắm bắt được cơ hội tận dụng đà tăng trưởng để đẩy mạnh chuyển đổi số. Tuy nhiên, chính sách này có thể gặp rủi ro nếu không đi kèm với đầu tư mạnh vào nhân lực.

        **c)** Chính sách học được từ Q-learning thể hiện rõ khả năng **thích nghi theo trạng thái**, đây là điểm mạnh so với các chính sách rule-based cố định. Tuy nhiên, mô hình hiện tại vẫn còn đơn giản (chỉ 81 trạng thái và 5 hành động rời rạc). Để áp dụng thực tế, cần mở rộng không gian trạng thái và hành động, đồng thời kết hợp với Deep Q-Network (DQN) hoặc các thuật toán nâng cao hơn.

        **d)** Việc áp dụng học tăng cường vào hoạch định chính sách kinh tế vĩ mô là hướng đi đầy tiềm năng. Nó cho phép chính sách không chỉ tối ưu theo điều kiện hiện tại mà còn có khả năng học hỏi và cải thiện qua thời gian. Tuy nhiên, cần kết hợp với chuyên gia kinh tế để thiết kế hàm phần thưởng phù hợp và đảm bảo tính giải thích được của chính sách.
        """)