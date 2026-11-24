# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ DỰ ÁN (PRESENTATION SCRIPT & SLIDE GUIDE)
## Đề tài: Khai Phá Dữ Liệu & Dự Báo Rủi Ro Trễ, Hủy Chuyến Bay Tại Việt Nam (FlightRisk VN)
> **Thời lượng đề xuất:** 15 – 20 phút  
> **Cấu trúc slide:** 12 slide chuẩn  
> **Phần thực nghiệm:** Live Demo trực tiếp hệ thống Web `http://127.0.0.1:5000/`  

---

## 1. PHÂN CÔNG THỜI LƯỢNG & PHÂN VAI THUYẾT TRÌNH

| Phân đoạn | Nội dung chính | Slide | Người trình bày gợi ý | Thời lượng |
| :---: | :--- | :---: | :--- | :---: |
| **Phần 1** | Giới thiệu, Bối cảnh & Vấn đề nghiên cứu | Slide 1 – 3 | Thành viên đại diện / Trưởng nhóm | 3 phút |
| **Phần 2** | Quy trình Xử lý Dữ liệu: Thu thập $\rightarrow$ Contract $\rightarrow$ Features | Slide 4 – 6 | Đại diện Nhóm Dữ liệu (TV1, TV2, TV3) | 4 phút |
| **Phần 3** | Tiền xử lý chống rò rỉ & Mô hình Học máy (Supervised & Unsupervised) | Slide 7 – 9 | Đại diện Nhóm Mô hình (TV4, TV5) | 5 phút |
| **Phần 4** | **LIVE DEMO:** Trình diễn Hệ thống Web & 3 Kịch bản UAT | Slide 10 | Đại diện Nhóm Web (TV6, TV7, TV8) | 5 phút |
| **Phần 5** | Đánh giá khoa học, Hạn chế & Kết luận | Slide 11 – 12 | Trưởng nhóm | 2 phút |
| **Q & A** | Trả lời câu hỏi chất vấn của Hội đồng | Toàn bộ | Cả 8 thành viên sẵn sàng | 5 – 10 phút |

---

## 2. KỊCH BẢN CHI TIẾT TỪNG SLIDE (SLIDE-BY-SLIDE WALKTHROUGH)

### SLIDE 1: Trang tiêu đề & Giới thiệu
- **Tiêu đề Slide:** Đề tài: Khai Phá Dữ Liệu & Dự Báo Rủi Ro Chuyến Bay Việt Nam (FlightRisk VN)
- **Thông tin phụ:** Môn học: Khai phá dữ liệu | Nhóm thực hiện: Nhóm 8 (8 thành viên).
- **Lời thoại (Script):**
  > *"Kính thưa Thầy/Cô và toàn thể các bạn, hôm nay Nhóm 8 xin phép được đại diện trình bày báo cáo tổng kết dự án Khai phá dữ liệu với đề tài: **'Nghiên cứu Khai phá Dữ liệu và Dự báo Rủi ro Chuyến bay tại Việt Nam'** mang tên hệ thống **FlightRisk VN**.*  
  > *Dự án của nhóm chúng em đi trọn vẹn vòng đời một đề tài Data Mining thực thụ: từ khâu cào dữ liệu thực tế tại 5 sân bay lớn của Việt Nam, xây dựng hợp đồng dữ liệu chuẩn, kỹ nghệ đặc trưng miền hàng không, huấn luyện học máy kết hợp khai phá luật và phân cụm, cho tới việc đóng gói thành một sản phẩm Web hoàn chỉnh phục vụ người dùng thời gian thực."*

---

### SLIDE 2: Bối cảnh thực tiễn & Câu hỏi nghiên cứu
- **Nội dung Slide:**
  - Bối cảnh: Trục bay SGN – HAN – DAD quá tải; thời tiết nhiệt đới mưa dông bất thường.
  - Vấn đề: Trễ dây chuyền gây thiệt hại lớn cho hãng bay và hành khách.
  - 3 Câu hỏi cốt lõi:
    1. Yếu tố nào (thời tiết, tải sân bay, khung giờ, hãng bay) tác động mạnh nhất đến sự chậm trễ?
    2. Các chuyến bay tại Việt Nam phân bố thành những cụm vận hành rủi ro nào?
    3. Có thể dự báo trước rủi ro và số phút trễ của một chuyến bay trước giờ khởi hành mà **không làm rò rỉ dữ liệu** hay không?
- **Lời thoại (Script):**
  > *"Thưa Thầy/Cô, hàng không Việt Nam có đặc thù là mật độ tập trung rất cao trên trục Tam giác vàng Hà Nội – Đà Nẵng – Sài Gòn. Khi xảy ra mưa dông nhiệt đới hoặc một sân bay đầu mối bị nghẽn, hiệu ứng trễ dây chuyền sẽ lan ra toàn mạng bay.*  
  > *Đứng trước thực tế đó, mục tiêu của nhóm em không chỉ dừng lại ở một file Jupyter Notebook trên dữ liệu giả lập, mà là trả lời 3 câu hỏi lớn: Đâu là quy luật kết hợp gây trễ? Các cụm rủi ro vận hành thực tế ra sao? Và làm thế nào để xây dựng một bộ máy dự báo tin cậy, không rò rỉ thông tin trước giờ cất cánh."*

---

### SLIDE 3: Kiến trúc Quy trình CRISP-DM & Dòng chảy dữ liệu (Data Lineage)
- **Nội dung Slide:** Sơ đồ quy trình 6 bước CRISP-DM tương ứng với 8 thành viên:
  - 14,000+ Snapshot thô $\rightarrow$ 13,965 chuyến bay sạch $\rightarrow$ 64 chiều đặc trưng $\rightarrow$ Mô hình Supervised + Unsupervised $\rightarrow$ REST API & Web Dashboard.
- **Lời thoại (Script):**
  > *"Để một dự án 8 thành viên phối hợp mượt mà và không giẫm chân lên nhau, nhóm em áp dụng nghiêm ngặt quy trình CRISP-DM. Chúng em thiết lập các bản giao ước dữ liệu rõ ràng giữa từng công đoạn:*  
  > *Thành viên 1 và 2 phụ trách lớp dữ liệu nền tảng; Thành viên 3, 4 và 5 phụ trách kỹ nghệ đặc trưng và thuật toán lõi; và Thành viên 6, 7, 8 chịu trách nhiệm đưa tri thức khai phá được lên giao diện Web."*

---

### SLIDE 4: Thu thập & Tích hợp Đa nguồn (Thành viên 1)
- **Nội dung Slide:**
  - Nguồn 1: Playwright FIDS Scraping tại 5 sân bay (SGN, HAN, DAD, CXR, PQC).
  - Nguồn 2: Aviationstack REST API (radar toàn cầu, bổ sung nhãn chuyến bay bị hủy).
  - Nguồn 3: Open-Meteo Weather API (khí tượng từng giờ theo tọa độ sân bay).
  - Khử trùng (Deduplication): Khóa `flight_date + flight_no + scheduled_time + origin + dest`.
- **Lời thoại (Script):**
  > *"Ở khâu thu thập, thay vì lấy các tập dữ liệu có sẵn trên mạng như Kaggle vốn không mang hơi thở hàng không Việt Nam, Thành viên 1 đã trực tiếp viết bot cào dữ liệu từ bảng điện tử FIDS của 5 sân bay lớn nhất nước. Để giải quyết việc thiếu dữ liệu các chuyến bay bị hủy, nhóm đã tích hợp thêm Aviationstack API và tự động khớp nối dữ liệu thời tiết theo giờ từ Open-Meteo theo tọa độ từng sân bay."*

---

### SLIDE 5: Hợp đồng Dữ liệu 24 Cột & Làm sạch (Thành viên 2)
- **Nội dung Slide:**
  - Hợp đồng dữ liệu `docs/schema.md` (24 cột chuẩn).
  - Ngưỡng trễ ngành hàng không: $\text{delay\_min} \ge 15$ phút $\rightarrow$ `delayed`.
  - Bộ kiểm định tự động `pipeline/check_data.py`: Kiểm tra tính duy nhất của khóa, miền giá trị của 7 hãng bay Việt Nam và logic nhãn hủy.
- **Lời thoại (Script):**
  > *"Thành viên 2 đã thiết lập Data Contract gồm 24 cột chuẩn mực. Điểm mấu chốt ở đây là chúng em tuân thủ đúng quy ước quốc tế: chuyến bay chỉ bị coi là trễ khi chênh lệch giữa giờ thực tế và giờ lịch từ 15 phút trở lên. Bộ kiểm định tự động `check_data.py` đảm bảo 100% dữ liệu chuyển giao sang bước sau không chứa bản ghi rác hay sai lệch logic."*

---

### SLIDE 6: Kỹ nghệ Đặc trưng Miền Hàng Không (Thành viên 3)
- **Nội dung Slide:**
  - Biến đổi lượng giác chu kỳ thời gian: $\sin/\cos$ của 24 giờ và 7 ngày trong tuần.
  - Chỉ số thời tiết tổng hợp (Weather Hazard Score) kết hợp mưa, gió giật và tầm nhìn.
  - Tải sân bay (`airport_load`) và biến tương tác phi tuyến `load_x_weather`.
- **Lời thoại (Script):**
  > *"Bước sang Giai đoạn 4, Thành viên 3 đã sáng tạo thêm 22 đặc trưng phản ánh bản chất vận hành. Nổi bật nhất là việc chiếu góc lượng giác cho giờ bay và ngày trong tuần, giúp mô hình hiểu được tính liên tục giữa 23h59 và 00h01. Ngoài ra, chỉ số tương tác giữa Tải sân bay và Thời tiết xấu đã mô phỏng chính xác hiện tượng: khi sân bay đang đông đúc mà có mưa dông thì nguy cơ trễ chuyến sẽ tăng theo hàm mũ."*

---

### SLIDE 7: Mã hóa, Chuẩn hóa & Chống Rò Rỉ Dữ Liệu (Thành viên 4)
- **Nội dung Slide:**
  - **Zero Data Leakage Guard:** Cách ly hoàn toàn 5 cột rò rỉ (`delay_min`, `actual_time`, `status`, `status_detail`, `is_delayed_for_analysis`).
  - `OneHotEncoder(handle_unknown="ignore")`: Chống crash khi gặp hãng bay/sân bay mới lạ.
  - `StandardScaler` & `train_medians`: Chuẩn hóa và điền khuyết thiếu CHỈ fit trên tập Train.
- **Lời thoại (Script):**
  > *"Thưa Thầy/Cô, một trong những lỗi nghiêm trọng nhất trong các đề tài dự báo chuyến bay là 'Data Leakage' – đưa các biến chỉ có sau khi máy bay cất cánh vào mô hình. Thành viên 4 đã xây dựng chốt chặn nghiêm ngặt, loại bỏ hoàn toàn 5 cột rò rỉ khỏi ma trận X.*  
  > *Đồng thời, bộ tiền xử lý chỉ được fit trên tập Train và sử dụng `handle_unknown='ignore'` để đảm bảo khi đưa vào sản xuất trên Web gặp dữ liệu mới lạ cũng không bị dừng đột ngột."*

---

### SLIDE 8: Học Máy Có Giám Sát & Lựa Chọn Mô Hình (Thành viên 5)
- **Nội dung Slide:**
  - Phân chia theo chuỗi thời gian (Chronological Split): Cắt tại `2026-10-31` (Train: 11,172 dòng, Test: 2,793 dòng).
  - Đánh giá bằng `TimeSeriesSplit` 3-folds trên Train (Test chỉ chấm 1 lần duy nhất).
  - Lựa chọn **Random Forest** (CV Macro-F1 = 0.4152 vs Baseline 0.2764, Test ROC-AUC = 0.8086).
  - Giải thích **Nghịch lý Accuracy**: Baseline đoán toàn bộ `on_time` được 80% nhưng vô dụng; Random Forest bắt được 55.4% các chuyến bay trễ.
  - Hồi quy số phút trễ: MAE = 15.13 phút.
  - Top Feature Importances: Chu kỳ giờ bay (17.7%), Hãng bay (9.3%), Phân khúc (9.2%), Tải sân bay (8.8%).
- **Lời thoại (Script):**
  > *"Tại Giai đoạn 6, Thành viên 5 áp dụng nguyên tắc cắt chuỗi thời gian: dùng quá khứ để học và kiểm thử trên tương lai, tuyệt đối không shuffle ngẫu nhiên nhằm triệt tiêu Lookahead Bias.*  
  > *Chúng em chọn mô hình bằng TimeSeriesSplit trên Train. Mặc dù Baseline đoán mù toàn bộ là đúng giờ đạt Accuracy 80%, nhưng nó có Recall bằng 0 đối với các chuyến trễ. Random Forest đạt ROC-AUC gần 0.81 và bắt được hơn 55% các chuyến trễ thực tế, chứng minh mô hình thực sự học được tri thức vận hành."*

---

### SLIDE 9: Khai Phá Tri Thức Không Giám Sát (Thành viên 5)
- **Nội dung Slide:**
  - **K-Means ($k=4$):** Tên cụm tự động suy ra từ dữ liệu (Sáng thời tiết tốt 41.8%, Chiều gió mạnh 22.0%, Tối cao điểm 34.5%, Chiều mưa dông 1.8% với tỷ lệ trễ 39%).
  - **PCA (3 chiều):** Giải thích 43.66% phương sai không gian dữ liệu.
  - **Isolation Forest:** Phát hiện 392 chuyến bay bất thường (2.81%), tỷ lệ trễ vọt lên 45.4%, lệch mạnh nhất ở cờ dông bão ($+3.88\sigma$).
  - **FP-Growth:** Luật tiêu biểu `VJ + SGN + Tối muộn -> Trễ` với độ nâng Lift = 1.97.
- **Lời thoại (Script):**
  > *"Điểm sáng trong phần Học không giám sát là tên 4 cụm rủi ro được thuật toán tự động đặt từ đặc trưng giờ và thời tiết thực tế, chứ không gán cứng chủ quan.*  
  > *Mô hình Isolation Forest đã cô lập được 392 chuyến bay cực đoan với tỷ lệ trễ lên tới 45.4%, chủ yếu do bão lớn. Các luật kết hợp khai phá bởi FP-Growth cũng chỉ ra: bay VietJet vào buổi tối tại Tân Sơn Nhất có xác suất trễ gấp đôi mức trung bình."*

---

### SLIDE 10: Trực Tiếp Live Demo Hệ Thống Web (Thành viên 6, 7, 8)
- **Nội dung Slide:** Màn hình chuyển sang trình duyệt: `http://127.0.0.1:5000/`.
- **Hành động Live Demo & Lời thoại:** *(Xem chi tiết tại Mục 3 bên dưới)*.

---

### SLIDE 11: Đánh Giá Khoa Học, Hạn Chế & Bài Học
- **Nội dung Slide:**
  - Trung thực khoa học: Lệch phân phối theo thời gian (tháng 10 trễ nhiều hơn tháng 11).
  - Hạn chế: Lớp hủy chỉ có 2 mẫu trong Test (nhiễu thống kê); Hồi quy độ trễ ($R^2 \approx 0.07$) phản ánh việc thiếu dữ liệu nội bộ như bảo dưỡng hay hành khách muộn.
  - Bài học kỹ thuật: Xây dựng cơ chế kiểm soát phả hệ mô hình (`_check_training_lineage`) tránh xung đột phiên bản thư viện.
- **Lời thoại (Script):**
  > *"Nhóm em nhìn nhận dự án với tinh thần khoa học nghiêm túc: Chúng em không cố tình tô hồng số liệu. Hồi quy độ trễ có R² khiêm tốn phản ánh đúng thực tế rằng trễ chuyến còn phụ thuộc vào nhiều yếu tố nội bộ như phi hành đoàn hay bảo trì máy bay mà dữ liệu bên ngoài không thể bao quát hết. Tuy nhiên, sai số MAE 15 phút vẫn mang giá trị tham khảo rất tốt cho hành khách."*

---

### SLIDE 12: Tổng Kết & Cảm Ơn
- **Nội dung Slide:**
  - Hoàn thành 100% mục tiêu CRISP-DM.
  - Đóng góp: Bộ pipeline tự động hóa hoàn chỉnh, 52 bài test kiểm định, và ứng dụng Web thực tế.
  - Mời Thầy/Cô và Hội đồng đặt câu hỏi.
- **Lời thoại (Script):**
  > *"Tóm lại, dự án FlightRisk VN đã hoàn thành trọn vẹn từ dữ liệu thô đến hệ thống ứng dụng với 52 bài kiểm thử tự động đạt chuẩn. Chúng em xin chân thành cảm ơn Thầy/Cô đã lắng nghe và rất mong nhận được những góp ý quý báu từ Hội đồng!"*

---

## 3. KỊCH BẢN LIVE DEMO HỆ THỐNG WEB (THAO TÁC THỰC TẾ)

Khi chuyển sang **Slide 10**, người thuyết trình mở trình duyệt tại địa chỉ `http://127.0.0.1:5000/` và thực hiện các bước sau:

### Bước 1: Giới thiệu tổng quan giao diện Web (1 phút)
- **Hành động:** Chỉ vào thanh điều hướng (4 Tabs) và dải trạng thái (Status Strip).
- **Lời thoại:**  
  > *"Đây là giao diện Web của FlightRisk VN. Phía trên là 4 phân hệ chính: Dự Báo Trực Tiếp, Mô Hình Supervised, Phân Cụm Unsupervised, và Tập Luật FP-Growth. Dải trạng thái hiển thị rõ: hệ thống đang áp dụng ngưỡng trễ chuẩn 15 phút trên 5 sân bay FIDS và 7 hãng hàng không Việt Nam."*

### Bước 2: Thử nghiệm 3 Kịch bản Kiểm thử nhanh (UAT Presets) (2 phút)
- **Thao tác 1: Bấm nút "☀️ Kịch bản 1: Sáng trời quang"**
  - **Kết quả hiển thị:** Vietnam Airlines, SGN $\rightarrow$ HAN, 08:30, trời quang, tải 15.
  - **Lời thoại:**  
    > *"Khi em bấm Kịch bản 1, form lập tức cập nhật thông số của Vietnam Airlines bay sáng sớm trời đẹp. Kết quả bên phải trả về ngay lập tức: Huy hiệu màu xanh báo **ĐÚNG GIỜ (LOW RISK)**, xác suất đúng giờ đạt trên 80%, gán vào Cụm 0 (Morning, Clear weather)."*
- **Thao tác 2: Bấm nút "🌬️ Kịch bản 2: Chiều gió mạnh"**
  - **Kết quả hiển thị:** Bamboo Airways, DAD $\rightarrow$ SGN, 15:30, gió 32 km/h.
  - **Lời thoại:**  
    > *"Ở Kịch bản 2, khi chuyến bay diễn ra vào buổi chiều kèm gió biển tại Đà Nẵng, huy hiệu lập tức chuyển sang màu vàng: **RỦI RO VỪA (MEDIUM RISK)**, xác suất trễ tăng lên xấp xỉ 35%."*
- **Thao tác 3: Bấm nút "⛈️ Kịch bản 3: Tối mưa dông & quá tải"**
  - **Kết quả hiển thị:** VietJet Air, SGN $\rightarrow$ HAN, 19:30, mưa 15mm, gió 45 km/h, tải sân bay 35.
  - **Lời thoại:**  
    > *"Đặc biệt ở Kịch bản 3, mô phỏng tình huống ngặt nghèo nhất: VietJet bay giờ tối cao điểm tại Tân Sơn Nhất giữa cơn dông lớn. Hệ thống lập tức kích hoạt cảnh báo đỏ **TRỄ CHUYẾN (HIGH RISK)** với xác suất trễ lên tới 71.9%, dự báo trễ xấp xỉ 18 phút. Đồng thời phía dưới xuất hiện ngay 3 luật FP-Growth cảnh báo hiệu ứng trễ giờ tối của VietJet."*

### Bước 3: Trình diễn các Tab Tri thức phân tích (1.5 phút)
- **Tab 2 (Mô hình Supervised):** Bấm chuyển sang tab `Mô Hình Supervised` $\rightarrow$ Cho Hội đồng thấy bảng chỉ số so sánh 3 mô hình và biểu đồ Feature Importance.
- **Tab 3 (Phân cụm Unsupervised):** Bấm sang tab `Phân Cụm Unsupervised` $\rightarrow$ Giới thiệu bảng 4 cụm K-Means, độ lệch chuẩn của Isolation Forest và thông số giải thích phương sai PCA.
- **Tab 4 (Tập luật FP-Growth):** Bấm sang tab `Tập Luật FP-Growth` $\rightarrow$ Cho thấy bảng các luật kết hợp được sắp xếp theo Lift.

---

## 4. BỘ 10 CÂU HỎI CHẤT VẤN TIỀM NĂNG (Q&A) & CÂU TRẢ LỜI "ĂN ĐIỂM"

#### Câu 1: "Tại sao nhóm không dùng train_test_split có shuffle ngẫu nhiên của scikit-learn mà lại cắt theo ngày?"
- **Trả lời:**  
  > *"Thưa Thầy/Cô, trong bài toán chuỗi thời gian của hàng không, nếu chúng em shuffle ngẫu nhiên, các chuyến bay trong cùng một ngày mưa dông sẽ bị rải đều vào cả tập Train và tập Test. Khi đó mô hình ở tập Train sẽ 'học lỏm' trước xu hướng thời tiết của ngày hôm đó để dự báo cho tập Test, gây ra hiện tượng rò rỉ dữ liệu theo thời gian (Lookahead Bias). Việc cắt theo ngày đảm bảo mô hình chỉ được học trên quá khứ và đánh giá trên tương lai độc lập, phản ánh đúng thực tế vận hành."*

#### Câu 2: "Tại sao Accuracy của Random Forest (69.89%) lại thấp hơn Baseline đoán toàn bộ đúng giờ (80.02%)?"
- **Trả lời:**  
  > *"Đây chính là hiện tượng Nghịch lý Độ chính xác (Accuracy Paradox) trong bài toán mất cân bằng lớp. Vì trong tập Test có 80% chuyến đúng giờ, một mô hình 'đoán mù' luôn nói chuyến bay đúng giờ sẽ nghiễm nhiên đạt 80% Accuracy nhưng hoàn toàn vô giá trị vì không phát hiện được bất kỳ chuyến trễ nào (Recall lớp trễ = 0%). Random Forest của nhóm em đạt ROC-AUC = 0.81 và Macro-F1 = 0.4152 (vượt trội so với baseline 0.2764) vì nó bắt được hơn 55% các chuyến trễ thực tế."*

#### Câu 3: "Nhóm làm thế nào để đảm bảo không bị rò rỉ dữ liệu (Data Leakage)?"
- **Trả lời:**  
  > *"Dự án thiết lập 3 tầng bảo vệ chống rò rỉ:  
  > Thứ nhất, loại bỏ hoàn toàn 5 cột rò rỉ (`delay_min`, `actual_time`, `status`, `status_detail`, `is_delayed_for_analysis`) khỏi ma trận X.  
  > Thứ hai, bộ tiền xử lý `FlightDataPreprocessor` chỉ fit các tham số (StandardScaler và trung vị missing values) trên tập Train, tuyệt đối không fit trên Test.  
  > Thứ ba, nhóm đã viết bài unit test tự động `test_zero_leakage_in_training_features` kiểm tra trực tiếp ma trận đặc trưng trước khi huấn luyện."*

#### Câu 4: "Tại sao hệ số $R^2$ của mô hình hồi quy số phút trễ lại khá thấp ($R^2 \approx 0.07$)?"
- **Trả lời:**  
  > *"Số phút trễ liên tục trong ngành hàng không chịu tác động của rất nhiều biến số nội bộ ngẫu nhiên mà bên ngoài không thể thu thập được, ví dụ như: máy bay chờ phụ tùng thay thế, sự cố kỹ thuật đột xuất, phi hành đoàn kẹt xe đến trễ, hoặc hành khách làm thủ tục muộn. Với dữ liệu thuần túy về lịch bay, tải sân bay và thời tiết, việc dự báo chính xác từng phút là bất khả thi. Tuy nhiên, sai số tuyệt đối MAE = 15 phút vẫn giúp hành khách và điều hành định hình được khung trễ xấp xỉ."*

#### Câu 5: "Làm thế nào nhóm chọn được số cụm $k=4$ trong K-Means?"
- **Trả lời:**  
  > *"Nhóm đã quét kiểm định tự động từ $k=2$ đến $k=8$, tính toán đồng thời giá trị Inertia (phương pháp Elbow) và Silhouette Score. Điểm $k=4$ tạo ra sự cân bằng tốt nhất: vừa có độ suy giảm quán tính rõ rệt, vừa phân tách thành 4 kịch bản vận hành thực tế có ý nghĩa kinh doanh rõ ràng: Sáng sớm trời tốt, Chiều gió mạnh, Tối cao điểm trục chính, và Chiều dông lốc cực đoan."*

#### Câu 6: "Tên của các cụm K-Means là do nhóm tự đặt hay máy tự sinh?"
- **Trả lời:**  
  > *"Tên cụm hoàn toàn do code tự động suy diễn từ giá trị trung bình của các đặc trưng sau khi phân cụm: Thuật toán kiểm tra giờ trung bình (sáng, chiều, tối), kết hợp với lượng mưa, gió giật và tầm nhìn của cụm đó để sinh ra nhãn tên tương ứng. Nhóm không gán cứng tên cụm theo index 0, 1, 2, 3 vì số thứ tự cụm của K-Means có tính ngẫu nhiên giữa các lần chạy."*

#### Câu 7: "Thuật toán Isolation Forest có ý nghĩa gì khi tham số contamination đã cố định là 3%?"
- **Trả lời:**  
  > *"Con số 3% xác định ngưỡng cắt điểm bất thường, nhưng giá trị cốt lõi của Isolation Forest nằm ở chỗ: Những chuyến bay nào bị đưa vào nhóm 3% đó và chúng khác biệt ở điểm gì? Phân tích của nhóm chỉ ra: các chuyến bay bị gắn cờ có tỷ lệ trễ lên tới 45.4% (so với 27% bình thường) và độ lệch chuẩn của các biến thời tiết cực đoan như `is_storm`, `precipitation` đều vượt ngưỡng $+3.8\sigma$."*

#### Câu 8: "Khi người dùng nhập một hãng hàng không lạ hoặc mã sân bay mới trên Web thì hệ thống có bị lỗi không?"
- **Trả lời:**  
  > *"Hoàn toàn không bị lỗi. Nhóm đã cấu hình `OneHotEncoder(handle_unknown='ignore')`. Khi gặp danh mục chưa từng xuất hiện trong tập Train, bộ tiền xử lý sẽ tự động mã hóa toàn bộ vector 0 cho biến đó, hệ thống vẫn suy luận bình thường dựa trên các đặc trưng thời tiết và thời gian còn lại."*

#### Câu 9: "Kiến trúc Web của nhóm xử lý thế nào để không phải train lại mô hình mỗi khi có yêu cầu dự báo?"
- **Trả lời:**  
  > *"Nhóm áp dụng cơ chế Model Serving chuẩn: Toàn bộ mô hình và bộ tiền xử lý được đóng gói thành các file artifact `.joblib` và `.json`. Khi máy chủ Web khởi động, hệ thống sử dụng cơ chế In-memory Caching để nạp sẵn mô hình lên RAM. Mỗi request gửi lên chỉ mất chưa đầy 50ms để biến đổi ma trận và suy luận kết quả mà không hề tốn tài nguyên huấn luyện lại."*

#### Câu 10: "Nếu được tiếp tục phát triển đề tài, nhóm sẽ cải tiến điểm gì tiếp theo?"
- **Trả lời:**  
  > *"Nhóm sẽ bổ sung thêm trường thông tin về Số hiệu đuôi tàu bay (Tail Number / Aircraft Turnaround). Trong ngành hàng không, một chiếc máy bay thường bay liên tục 4–6 chặng mỗi ngày. Nếu chúng ta theo dõi được lịch trình xoay vòng của từng chiếc tàu bay cụ thể, mô hình sẽ nắm bắt được hiệu ứng trễ tích lũy từ các chặng trước, từ đó nâng cao đáng kể độ chính xác của dự báo độ trễ."*
