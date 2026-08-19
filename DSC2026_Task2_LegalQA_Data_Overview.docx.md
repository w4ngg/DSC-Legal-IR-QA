**UIT DATA SCIENCE CHALLENGE 2026**

# **Task 2 – Legal Question Answering (LegalQA)**

*Hỏi đáp pháp luật tiếng Việt*

| Mục tiêu | Với một câu hỏi pháp luật, hệ thống truy xuất văn bản liên quan và tạo câu trả lời bằng văn xuôi dựa trên căn cứ pháp lý. |
| :---- | :---- |

# **1\. Mô tả bài toán**

* Input: một câu hỏi pháp luật tiếng Việt.  
* Output: câu trả lời tự nhiên bằng văn xuôi.

| "id": {         "question": "Trách nhiệm của tổ chức đấu thầu, bảo lãnh, đại lý phát hành",         "answer": "Theo Điều 37 Nghị định 153/2020/NĐ-CP, được sửa đổi bởi khoản 26 Điều 1 Nghị định 65/2022/NĐ-CP (Có hiệu lực từ 16/09/2022) quy định cụ thể:\\n- Tuân thủ quy định của pháp luật chứng khoán và quy định tại Điều 14 Nghị định này khi cung cấp dịch vụ đấu thầu, bảo lãnh, đại lý phát hành.\\n- Thực hiện chế độ báo cáo theo quy định tại Nghị định này.\\n- Trường hợp vi phạm quy định của pháp luật khi cung cấp dịch vụ tùy theo tính chất và mức độ vi phạm sẽ bị xử phạt vi phạm hành chính theo quy định về xử phạt hành chính trong lĩnh vực chứng khoán và thị trường chứng khoán hoặc truy cứu trách nhiệm hình sự\\nTrước đây, căn cứ Điều 37 Nghị định 153/2020/NĐ-CP quy định như sau: \\nTrách nhiệm của tổ chức đấu thầu, bảo lãnh, đại lý phát hành\\n1. Tuân thủ quy định của pháp luật khi cung cấp dịch vụ đấu thầu, bảo lãnh, đại lý phát hành.\\n2. Thực hiện đúng theo hợp đồng cung cấp dịch vụ ký kết với doanh nghiệp phát hành và nhà đầu tư mua trái phiếu.\\n3. Thực hiện chế độ báo cáo theo quy định tại Nghị định này.\\n4. Trường hợp vi phạm quy định của pháp luật khi cung cấp dịch vụ sẽ bị xử phạt vi phạm hành chính theo quy định về xử phạt hành chính trong lĩnh vực chứng khoán và thị trường chứng khoán."     } |
| :---- |

# **2\. Cấu trúc dữ liệu (Sẽ được cung cấp tương ứng theo timeline cuộc thi)**

| Tệp | Nội dung |
| :---- | :---- |
| **train.json** | Tập dữ liệu huấn luyện cho các đội phát triển phương pháp. |
| **warmup.json** | Tập dữ liệu mẫu phục vụ vòng Warm-up, giúp làm quen bài toán và quy trình submission. |
| **public-official.json** | Tập dữ liệu dùng trong giai đoạn Public Test theo cấu hình chính thức. |
| **private-official.json** | Tập dữ liệu chính thức của Private Test. |
| **selected-contexts.zip** | Kho văn bản được chọn; gồm nhiều tệp context\_\*.json.  |

# **3\. Ví dụ cấu trúc một văn bản**

**context\_\*.json**

* id: mã định danh duy nhất của văn bản  
* name: tiêu đề văn bản  
* link: đường dẫn nguồn  
* passage: nội dung văn bản được sử dụng làm ngữ cảnh/căn cứ

| {     "link": "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608.aspx",     "name": "Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608",     "passage": "BỘ Y TẾ\\r\\n\\n  -------\\n\\nCỘNG HÒA XÃ HỘI\\r\\n\\n  CHỦ NGHĨA VIỆT NAM\\r\\n\\n  Độc lập \- Tự do \- Hạnh phúc \\r\\n\\n  ---------------\\n\\nSố: 5868/QĐ-BYT\\n\\nHà Nội, ngày 28\\r\\n\\n  tháng 9 năm 2018\\n\\n\\n\\nQUYẾT ĐỊNH\\n\\nQUY\\r\\n\\nĐỊNH CHỨC NĂNG, NHIỆM VỤ, QUYỀN HẠN VÀ CƠ CẤU TỔ CHỨC CỦA VỤ TRANG THIẾT BỊ VÀ CÔNG\\r\\n\\nTRÌNH Y TẾ THUỘC BỘ Y TẾ\\n\\nBỘ TRƯỞNG BỘ Y TẾ\\n\\nCăn cứ Nghị định số 75/2017/NĐ-CP … \- Lưu: VT, TCCB, TTB, PC.\\n\\nBỘ TRƯỞNG\\n\\n\\r\\n\\n  Nguyễn Thị Kim Tiến\\n\\n",     "id": 740 }  |
| :---- |

# **4\. Đánh giá tác vụ**

Câu trả lời do hệ thống sinh ra được so sánh với câu trả lời tham chiếu do các chuyên gia pháp lý xây dựng. Kết quả được đánh giá bằng hai độ đo: METEOR (độ đo chính) và ROUGE-L (độ đo phụ). Giá trị càng cao càng tốt.

**METEOR — Độ đo chính**

METEOR (Metric for Evaluation of Translation with Explicit ORdering) đánh giá mức độ tương đồng giữa câu trả lời dự đoán và câu trả lời tham chiếu dựa trên mức độ khớp của các token, đồng thời xem xét Precision, Recall và mức độ liên tục của các token được khớp. METEOR càng cao cho thấy câu trả lời dự đoán càng tương đồng với đáp án tham chiếu.

**ROUGE-L — Độ đo phụ**

ROUGE-L đánh giá mức độ tương đồng giữa câu trả lời dự đoán và câu trả lời tham chiếu dựa trên Longest Common Subsequence (LCS), qua đó phản ánh mức độ bảo toàn nội dung và thứ tự thông tin. ROUGE-L càng cao càng tốt.

Quy tắc xếp hạng: METEOR là độ đo chính để xếp hạng các đội; ROUGE-L là độ đo phụ dùng để tham khảo và đánh giá bổ sung chất lượng câu trả lời.

| Độ đo | Vai trò | Mô tả |
| :---- | :---- | :---- |
| **METEOR** | Độ đo chính | Đánh giá mức độ tương đồng giữa câu trả lời dự đoán và câu trả lời tham chiếu dựa trên mức độ khớp token, kết hợp precision, recall và mức độ liên tục/thứ tự của các token khớp. |
| **ROUGE-L** | Độ đo phụ | Đánh giá mức độ tương đồng dựa trên Longest Common Subsequence (LCS), phản ánh mức độ nội dung và thứ tự thông tin được bảo toàn. |

