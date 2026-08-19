# Research v1 — Phương pháp cho Legal Information Retrieval tiếng Việt

> Ngày tạo: 2026-08-15  
> Phạm vi: Task 1 — LegalIR  
> Trạng thái: Draft nghiên cứu ban đầu, chưa phải kết quả thực nghiệm  
> Mục tiêu: Xây dựng hướng tiếp cận retrieval và chunking phù hợp với corpus pháp luật tiếng Việt của DSC 2026.

## 1. Kết luận chính

Task 1 không nên truy hồi trực tiếp trên 8.532 văn bản luật nguyên khối. Hướng phù hợp nhất là:

```text
Chia văn bản theo cấu trúc pháp luật
→ truy hồi ở mức Điều/Khoản
→ gom điểm về document_id
→ rerank
→ trả về tối đa 5 văn bản
```

Corpus hiện tại có dấu hiệu mạnh là một phiên bản xử lý lại hoặc cùng dòng dữ liệu với bộ Vietnamese Legal Text Retrieval trong bài COLING 2020. Bộ dữ liệu trong nghiên cứu đó gồm 8.586 văn bản thu thập từ `vbpl.vn` và `thuvienphapluat.vn`, sau đó được chia thành 117.545 điều luật; số văn bản liên quan trung bình cho mỗi câu hỏi là 1,19. Corpus DSC hiện tại có 8.532 văn bản và trung bình 1,091 gold documents/query. Tuy nhiên, chưa có metadata để khẳng định hai bộ có mapping 1–1.

Nguồn: [Answering Legal Questions by Learning Neural Attentive Text Representation — COLING 2020](https://aclanthology.org/2020.coling-main.86.pdf).

## 2. Bản chất dữ liệu Task 1 hiện tại

Phân tích `IR/train.json` và `selected-contexts/` cho kết quả:

| Thuộc tính | Kết quả |
| --- | ---: |
| Số văn bản | 8.532 |
| Train queries | 7.000 |
| Tổng gold labels | 7.637 |
| Gold documents/query | 1,091 |
| Câu hỏi có 1 gold document | 6.447 — 92,1% |
| Câu hỏi có từ 2 gold documents | 553 — 7,9% |
| Độ dài văn bản trung vị | khoảng 23.210 ký tự |
| Độ dài gold document trung vị | khoảng 66.232 ký tự |
| Văn bản có heading giống `Điều` | 7.319 |
| Tổng số Điều ước lượng bằng regex | khoảng 162.532 |
| Độ dài một Điều trung vị | khoảng 856 ký tự |

Hai hệ quả quan trọng:

1. Gold documents dài hơn đáng kể corpus thông thường. Whole-document embedding hoặc whole-document BM25 dễ bị nhiễu bởi phần lớn nội dung không liên quan đến query.
2. Hầu hết câu hỏi chỉ cần tìm đúng một văn bản, nhưng hệ thống vẫn cần giữ top 2–5 để tăng recall và bao phủ nhóm đa nhãn.

Các truy vấn chủ yếu được viết bằng ngôn ngữ tự nhiên, không phải truy vấn citation. Một phép dò regex sơ bộ chỉ tìm thấy rất ít câu hỏi nhắc trực tiếp `Điều/Khoản/Điểm/Chương/Mục`. Vì vậy citation matching chỉ là một feature bổ trợ; semantic retrieval vẫn cần thiết.

### 2.1. Phân bố độ phổ biến của gold documents

Train tham chiếu 7.637 lần tới 3.105 document riêng biệt. Phân bố nhãn khá lệch:

| Nhóm document | Tỷ lệ tổng gold labels tích lũy |
| --- | ---: |
| Top 1 | 1,43% |
| Top 10 | 9,31% |
| Top 50 | 20,09% |
| Top 100 | 28,24% |
| Top 500 | 54,17% |
| Top 1.000 | 69,27% |

Có 1.863 gold documents chỉ xuất hiện một lần, trong khi một số bộ luật lớn xuất hiện rất thường xuyên. Do đó có thể dùng document prior như một feature nhỏ trong reranking, nhưng không được để prior lấn át evidence giữa query và chunk vì long-tail vẫn rất lớn.

## 3. Cách tổ chức document của các dataset pháp luật liên quan

### 3.1. Vietnamese Legal Text Retrieval — COLING 2020

Đây là dataset gần nhất với Task 1 hiện tại:

- Dữ liệu thô là các văn bản luật tiếng Việt nguyên khối.
- Nguồn gồm `vbpl.vn` và `thuvienphapluat.vn`.
- Sau làm sạch, 8.586 văn bản được tách thành 117.545 điều luật.
- Retrieval unit là `article`, không phải toàn bộ văn bản luật.
- Hệ thống đầu tiên lấy top 1.000 bằng Elasticsearch, sau đó neural rerank.
- Article được giới hạn khoảng 600 tokens; mô hình phân cấp dùng tối đa 30 câu, mỗi câu tối đa 25 từ.
- Phần có ích trong một relevant article thường chỉ là một vài câu.

Kết quả được báo cáo cho thấy BM25 Recall@20 khoảng 0,357, trong khi mô hình hierarchical attention tốt nhất đạt khoảng 0,825. Điều này củng cố giả thuyết rằng retrieval ở cấp Điều/đoạn có lợi hơn retrieval trực tiếp trên toàn văn bản.

Nguồn: [COLING 2020](https://aclanthology.org/2020.coling-main.86.pdf).

### 3.2. ALQAC

ALQAC tổ chức corpus theo dạng:

```text
law_id
└── article_id
    └── article text
```

Gold label là cặp `{law_id, article_id}`. Đơn vị tìm kiếm và đánh giá là Điều luật, không phải toàn bộ đạo luật.

Nguồn: [ALQAC 2023](https://sites.google.com/view/ALQAC2023).

### 3.3. Zalo Legal QA và dữ liệu tổng hợp tiếng Việt

Các phiên bản Zalo Legal QA được dùng trong nghiên cứu gần đây có khoảng 61 nghìn passages/articles. Mỗi query gắn với đoạn hoặc Điều có thể trả lời nó.

Nghiên cứu tạo dữ liệu tổng hợp cho Vietnamese legal retrieval chia văn bản theo hierarchy:

```text
Văn bản
→ Chương
→ Mục
→ Điều
→ Khoản
```

Mỗi passage giữ metadata như lĩnh vực, tên văn bản, heading và nội dung. Nghiên cứu tạo hơn 620 nghìn query, lọc còn hơn 500 nghìn query hợp lệ, rồi huấn luyện bi-encoder và ColBERT với hard negatives. Kết quả cũng cho thấy document-level hit cao hơn nhiều exact passage hit, phản ánh việc tìm đúng văn bản cha dễ hơn tìm đúng đoạn chính xác.

Nguồn: [Improving Vietnamese Legal Document Retrieval using Synthetic Data](https://arxiv.org/abs/2412.00657).

### 3.4. VLSP DRiLL 2025

ViDRILL làm việc trên 59.636 điều luật và dùng hai cấp chunk:

- Chunk ngắn: tối đa 450 ký tự, theo Khoản, không overlap, dùng cho retrieval.
- Chunk dài: tối đa 2.000 ký tự, ngắt theo newline và có overlap, dùng cho reranking.
- BM25 lấy top 200; các dense retriever lấy top 30.
- Kết quả từ nhiều retriever được hợp nhất, rerank rồi quy về article ID.
- Không xem mọi chunk trong gold article là positive; hệ thống chọn chunk đại diện có điểm cao nhất.
- Sau vòng huấn luyện đầu, hệ thống mine lại hard negatives từ các kết quả sai nhưng có thứ hạng cao.

Đây là kiến trúc gần nhất với pipeline phù hợp cho Task 1 hiện tại.

Nguồn: [ViDRILL — VLSP 2025](https://aclanthology.org/2025.vlsp-1.17.pdf).

### 3.5. BSARD

BSARD lưu từng Điều luật kèm:

- code/law;
- article number;
- hierarchy description;
- full article content.

Với Điều quá dài, tác giả cắt thành các chunk 200 tokens, overlap 20, rồi tổng hợp embedding của các chunk. Fine-tuning theo miền cải thiện mạnh retrieval; encoder tổng quát dùng zero-shot hoạt động rất kém. Điều này cho thấy domain/task fine-tuning quan trọng hơn chỉ thay một embedding model mới.

Nguồn: [BSARD — ACL 2022](https://aclanthology.org/2022.acl-long.468.pdf).

### 3.6. COLIEE

COLIEE dùng từng Điều của Bộ luật Dân sự Nhật Bản làm retrieval unit. Gold chứa những Điều độc lập hoặc kết hợp có thể trả lời câu hỏi. Đây tiếp tục là bằng chứng rằng benchmark legal retrieval thường index Điều, giữ bộ luật/văn bản làm parent entity.

Nguồn: [COLIEE 2025 Task 3](https://coliee.org/COLIEE2025/tasks/task3).

### 3.7. Mẫu số chung

Các dataset và hệ thống pháp lý hiệu quả thường có chung thiết kế:

```text
Retrieval unit: Điều/Khoản/passage
Parent/output unit: văn bản luật hoặc article ID theo yêu cầu benchmark
```

Đây chính là điểm cần điều chỉnh cho DSC: truy hồi trên child chunks nhưng luôn giữ mapping về `document_id` gốc để tạo output.

## 4. Chiến lược chunking đề xuất

### 4.1. Cấu trúc phân cấp

Parser nên nhận dạng hierarchy:

```text
Document
├── Metadata/preamble
├── Chương
│   ├── Mục
│   │   ├── Điều
│   │   │   ├── Khoản
│   │   │   │   └── Điểm
└── Phụ lục
```

Mỗi chunk nên lưu tối thiểu:

```json
{
  "document_id": 245154,
  "chunk_id": "245154:article:173:clause:2",
  "chunk_type": "clause",
  "document_name": "...",
  "document_number": "...",
  "chapter": "...",
  "section": "...",
  "article": "Điều 173. Tội trộm cắp tài sản",
  "clause": "2",
  "text": "...",
  "start": 12345,
  "end": 13012
}
```

Offsets phải trỏ về text đã chuẩn hóa hoặc có thêm raw offsets rõ ràng để có thể kiểm tra chunk và phục vụ QA sau này.

### 4.2. Retrieval chunks

Thứ tự ưu tiên:

1. Giữ nguyên toàn bộ `Điều` nếu Điều không quá dài.
2. Nếu Điều dài hơn khoảng 1.500–2.500 ký tự hoặc 384–512 tokens, chia theo `Khoản`.
3. Nếu Khoản còn dài, chia tiếp theo `Điểm`.
4. Chỉ dùng fixed-size chunk khi không parse được cấu trúc.

Fallback ban đầu:

- 350–500 tokens;
- overlap 64–100 tokens hoặc khoảng 15%;
- ưu tiên ngắt theo đoạn/newline;
- không cắt giữa số hiệu văn bản, Khoản hoặc Điểm.

Nghiên cứu chunking trên luật thành văn cũng cho thấy chunk theo section/subsection thường hiệu quả hơn các phương pháp semantic hoặc LLM phức tạp khi văn bản đã có cấu trúc pháp lý rõ ràng.

Nguồn: [Statutory Legal Text Chunking](https://arxiv.org/abs/2605.19806).

### 4.3. Hierarchy prefix

Nội dung đưa vào index nên có dạng:

```text
[Tên văn bản]
[Số/ký hiệu]
[Chương ... > Mục ... > Điều ...]
[Nội dung Khoản/Điểm]
```

Ví dụ:

```text
Bộ luật Hình sự 2015
Chương XVI > Điều 173. Tội trộm cắp tài sản
2. Phạm tội thuộc một trong các trường hợp sau đây...
```

Prefix giúp chunk nhỏ giữ ngữ cảnh, tăng khả năng phân biệt các Điều có nội dung tương tự và hỗ trợ query nhắc tới tên hoặc loại văn bản.

### 4.4. Metadata chunk

Mỗi văn bản nên có một chunk riêng chứa:

- tên văn bản;
- số/ký hiệu;
- loại văn bản;
- cơ quan ban hành;
- ngày ban hành;
- lĩnh vực;
- phạm vi điều chỉnh;
- danh sách heading hoặc mục lục nếu có.

Metadata chunk hữu ích với câu hỏi dạng “văn bản nào quy định…”, “nghị định xử phạt…” hoặc query có số hiệu văn bản.

Không nên sinh metadata bằng LLM trong baseline đầu tiên. Title/summary do LLM tạo có thể được thử như data augmentation ở giai đoạn sau, nhưng cần đánh giá riêng vì có nguy cơ bỏ mất điều kiện hoặc thêm thông tin không tồn tại trong nguồn.

### 4.5. Gom điểm chunk về document

Không nên cộng điểm của tất cả chunk, vì văn bản càng dài càng có nhiều cơ hội nhận điểm.

Baseline an toàn:

```text
document_score = max(chunk_scores)
```

Có thể ablation thêm:

```text
document_score =
    0.70 × best_chunk
  + 0.20 × second_best_chunk
  + 0.10 × metadata_chunk
```

Tuy nhiên `MaxP` phải là baseline chính. Khi hợp nhất nhiều retriever, có thể rank chunks trước rồi dùng Reciprocal Rank Fusion ở mức document.

## 5. Pipeline retrieval đề xuất

```text
Query
 ├─ BM25 trên structural chunks
 ├─ Dense retrieval trên structural chunks
 └─ Metadata/citation matching
             ↓
     Gom theo document_id
             ↓
        RRF / score fusion
             ↓
     Top 30–50 documents
             ↓
 Cross-encoder rerank top chunks
             ↓
   Top 5 document IDs duy nhất
```

### 5.1. Stage 1 — BM25

Nên bắt đầu với hai lexical indexes:

- text tiếng Việt chưa word-segment;
- text đã word-segment bằng VnCoreNLP hoặc tokenizer tương đương.

Sau đó hợp nhất bằng Reciprocal Rank Fusion.

Không nên loại stopword mạnh. Những từ như `không`, `chưa`, `trừ`, `được`, `phải` có ý nghĩa pháp lý. Cần giữ nguyên:

- số Điều/Khoản;
- số hiệu văn bản;
- ngày tháng;
- tỷ lệ phần trăm;
- mức tiền;
- chữ viết tắt.

Benchmark tiếng Việt gần đây cho thấy BM25 vẫn rất mạnh. Trên ALQAC, BM25 đạt Recall@10 khoảng 97,9%; hybrid BM25 + BGE-M3 tăng thêm chất lượng top rank và recall.

Nguồn: [Which Works Best for Vietnamese? — Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.110/).

### 5.2. Stage 2 — Dense retrieval

Hai lựa chọn thực tế cho baseline:

- **BGE-M3**: multilingual, hỗ trợ dense, sparse và multi-vector, context dài tới 8.192 tokens. Nguồn: [BGE-M3](https://arxiv.org/abs/2402.03216).
- **mGTE**: multilingual encoder và reranker, hỗ trợ context dài. Nguồn: [mGTE](https://aclanthology.org/2024.emnlp-industry.103/).

Dù model hỗ trợ context dài, vẫn nên embed Điều/Khoản thay vì toàn bộ văn bản. Long-context capability không giải quyết hoàn toàn dilution khi một văn bản chứa hàng trăm quy định không liên quan.

### 5.3. Stage 3 — Reranking

Các lựa chọn ban đầu:

- `bge-reranker-v2-m3`;
- multilingual GTE reranker;
- PhoBERT hoặc một Vietnamese legal encoder được fine-tune dưới dạng cross-encoder.

Mỗi document chỉ cần đưa vào reranker:

- metadata chunk;
- 2–3 retrieval chunks tốt nhất;
- tùy chọn: toàn bộ Điều chứa các chunk đó.

Không ghép toàn bộ văn bản gốc vào cross-encoder.

## 6. Fine-tuning khi gold chỉ có ở mức document

Không được đánh dấu mọi chunk trong gold document là positive. Với một bộ luật dài hàng trăm Điều, phần lớn chunk không liên quan đến query và sẽ tạo label noise rất lớn.

Quy trình pseudo-positive/multiple-instance đề xuất:

1. Với mỗi `(query, gold_document)`, chạy BM25 và dense trên các chunk bên trong gold document.
2. Chọn 1–3 chunk có điểm cao nhất làm pseudo-positive.
3. Chọn hard-negative document từ top BM25/dense nhưng không nằm trong gold.
4. Dùng chunk tốt nhất của hard-negative document làm negative instance.
5. Fine-tune retriever lần đầu.
6. Dùng retriever mới mine lại hard negatives có thứ hạng cao.
7. Fine-tune thêm một vòng và kiểm tra ablation.

Cách này tương tự lựa chọn representative chunk trong ViDRILL. Nghiên cứu synthetic Vietnamese legal retrieval cũng cho thấy ColBERT chỉ thật sự cạnh tranh sau khi được huấn luyện theo miền với query tổng hợp và hard negatives.

Vì vậy:

- Không nên kết luận ColBERT kém chỉ từ kết quả zero-shot.
- ColBERT phù hợp ở giai đoạn sau BM25 + dense baseline.
- Query expansion bằng LLM nên để sau cùng vì có thể làm mất ngoại lệ hoặc thêm sai thuật ngữ.

Một phương pháp query expansion pháp luật đã thử tách legal facts/terms cho BM25 và rewrite query cho dense retriever, sau đó ensemble các kết quả. Đây là hướng đáng thử sau khi pipeline retrieval cơ bản ổn định.

Nguồn: [Legal Query Expansion](https://arxiv.org/abs/2410.12154).

## 7. Ý nghĩa của average relevant documents = 1,091

Trong train:

- 92,1% câu hỏi chỉ có một gold document.
- Với nhóm này, recall của query bằng 1 nếu gold xuất hiện ở bất kỳ vị trí nào trong top 5.
- Vì Recall là metric xếp hạng chính và output được phép có tối đa 5 document, mặc định nên trả đủ 5 document IDs duy nhất.
- Precision là tie-break, nên chỉ trả ít hơn 5 nếu validation chứng minh Recall@1 hoặc Recall@3 gần như bằng Recall@5.

Các metric nội bộ cần theo dõi:

```text
Recall@1, Recall@3, Recall@5
Precision@1, Precision@3, Precision@5
MRR@5
Recall@5 cho single-label queries
Recall@5 cho multi-label queries
```

Không nên chỉ nhìn một Recall tổng hợp. Hệ thống có thể tốt trên 92% single-label nhưng bỏ sót document thứ hai hoặc thứ ba của nhóm đa nhãn.

## 8. Kế hoạch thí nghiệm

| ID | Thí nghiệm | Mục tiêu |
| --- | --- | --- |
| E0 | Whole-document BM25 | Baseline gốc |
| E1 | Điều/Khoản BM25 + MaxP | Đo lợi ích của structural chunking |
| E2 | BM25 raw + BM25 word-segment, hợp nhất bằng RRF | Cải thiện lexical retrieval tiếng Việt |
| E3 | BGE-M3 dense + BM25 RRF | Tăng hybrid recall |
| E4 | Cross-encoder rerank top 30 documents | Cải thiện thứ tự top 5 |
| E5 | Fine-tune với pseudo-positive và hard negatives | Domain adaptation |
| E6 | ColBERT hoặc BGE-M3 multi-vector | Fine-grained matching |
| E7 | Synthetic queries/query expansion | Cải thiện long-tail query |

Thứ tự ưu tiên là:

```text
E0 → E1 → E2 → E3 → E4
```

Chỉ sau khi có evaluator đáng tin cậy và ablation rõ ràng mới tiến tới E5–E7.

### 8.1. Ablation chunking tối thiểu

Nên so sánh ít nhất:

1. Whole document.
2. Whole article.
3. Article, nhưng Điều dài được chia theo Khoản.
4. Fixed 450 characters như ViDRILL.
5. Fixed 384 tokens, overlap 64.
6. Hierarchical chunk có prefix so với không có prefix.
7. MaxP so với top-2 aggregation.

### 8.2. Chia validation

Warmup trùng 346 query IDs với train, nên không phải validation độc lập. Cần tạo split nội bộ sau khi:

- chuẩn hóa câu hỏi;
- gom các câu trùng hoặc gần trùng vào cùng một group;
- cân bằng single-label/multi-label;
- cân bằng head documents và long-tail documents nếu có thể.

Mọi bảng kết quả phải nêu rõ validation có loại duplicate hay không.

## 9. Những rủi ro cần kiểm tra

1. **Regex parse sai cấu trúc:** line break trong corpus có dạng `Điều\n1.` hoặc heading bị tách dòng.
2. **Phụ lục và biểu mẫu:** không phải lúc nào cũng có `Điều`; cần fallback chunker.
3. **Văn bản sửa đổi/bãi bỏ:** query có thể cần phân biệt phiên bản, hiệu lực và văn bản thay thế.
4. **Document-length bias:** cộng điểm chunk sẽ thiên vị văn bản dài.
5. **False positives cùng chủ đề:** nhiều Điều trong cùng một lĩnh vực chia sẻ từ vựng pháp lý gần giống nhau.
6. **Pseudo-positive sai:** top chunk trong gold document chưa chắc là căn cứ thực sự; nên dùng ensemble hoặc teacher reranker.
7. **Popular-document bias:** các bộ luật phổ biến xuất hiện nhiều trong train nhưng không được phép làm mất recall của long-tail.
8. **Validation leakage:** query trùng giữa train và warmup làm kết quả quá lạc quan.

## 10. Đề xuất triển khai v1

Phiên bản thực nghiệm đầu tiên nên đủ đơn giản để kiểm tra giả thuyết quan trọng nhất:

1. Viết structural parser cho `Điều → Khoản → Điểm`, có fallback fixed chunks.
2. Sinh file chunk corpus kèm mapping `chunk_id → document_id` và hierarchy metadata.
3. Xây BM25 index trên chunks.
4. Lấy top chunks, dùng `MaxP` gom về document.
5. Trả top 5 document IDs và chạy evaluator Recall/Precision theo đúng quy định cuộc thi.
6. So E0 whole-document BM25 với E1 structural-chunk BM25.
7. Chỉ thêm dense retrieval nếu E1 và evaluator đã được kiểm chứng.

Đây là bước có tỷ lệ lợi ích/chi phí tốt nhất, đồng thời tạo nền tảng dùng lại được cho Task 2 LegalQA sau này.

## 11. Tài liệu tham khảo chính

1. [Answering Legal Questions by Learning Neural Attentive Text Representation — COLING 2020](https://aclanthology.org/2020.coling-main.86.pdf)
2. [Which Works Best for Vietnamese? A Practical Study of Information Retrieval Methods across Domains — EACL 2026](https://aclanthology.org/2026.findings-eacl.110/)
3. [ViDRILL: A Multi-Stage Retrieval Framework for Vietnamese Legal Document Search — VLSP 2025](https://aclanthology.org/2025.vlsp-1.17.pdf)
4. [Improving Vietnamese Legal Document Retrieval using Synthetic Data](https://arxiv.org/abs/2412.00657)
5. [Vietnamese Legal Text Retrieval based on Sparse and Dense Retrieval approaches](https://www.sciencedirect.com/science/article/pii/S1877050924003521)
6. [BSARD: A Benchmark for Belgian Statutory Article Retrieval — ACL 2022](https://aclanthology.org/2022.acl-long.468.pdf)
7. [COLIEE 2025 Task 3](https://coliee.org/COLIEE2025/tasks/task3)
8. [BGE-M3](https://arxiv.org/abs/2402.03216)
9. [mGTE](https://aclanthology.org/2024.emnlp-industry.103/)
10. [Statutory Legal Text Chunking](https://arxiv.org/abs/2605.19806)
11. [Legal Query Expansion](https://arxiv.org/abs/2410.12154)

## 12. Phạm vi của bản v1

Bản này tổng hợp hướng nghiên cứu và thiết kế dự kiến. Nó chưa chứa:

- code chunker/indexer;
- benchmark latency hoặc memory;
- kết quả Recall/Precision thực nghiệm;
- hyperparameter tuning;
- lựa chọn model cuối cùng.

Các bản tiếp theo trong `research_method/` phải ghi rõ thay đổi so với v1 và liên kết tới artifact/kết quả thực nghiệm tương ứng.
