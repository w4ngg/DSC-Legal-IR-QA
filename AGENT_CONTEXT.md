# DSC Legal — Context cho các session sau

> Snapshot đã được đọc và kiểm tra toàn bộ vào 2026-08-15; khung retrieval Task 1 được cập nhật model stack vào 2026-08-20. Đọc tệp này đầu tiên khi bắt đầu làm việc trong workspace.

## Mục đích workspace

Đây là dữ liệu cho **UIT Data Science Challenge 2026**, gồm hai tác vụ độc lập nhưng dùng chung kho văn bản pháp luật tiếng Việt:

| Tác vụ | Đầu vào | Đầu ra cần tạo | Đánh giá chính |
| --- | --- | --- | --- |
| LegalIR (Task 1) | Câu hỏi pháp luật | Tối đa 5 ID văn bản liên quan | Recall trung bình; Precision là tie-break |
| LegalQA (Task 2) | Câu hỏi pháp luật | Câu trả lời tự nhiên dựa trên căn cứ pháp lý | METEOR; ROUGE-L là phụ |

Hai tập câu hỏi IR và QA không dùng chung query ID. LegalQA không có nhãn document ID; có thể dùng corpus chung và kiến thức retriever học từ IR, nhưng đây là một suy luận triển khai chứ không phải nhãn QA được cung cấp.

## Cấu trúc hiện có

```text
.
├── AGENT_CONTEXT.md                         # tệp này
├── DSC2026_Task1_LegalIR_Data_Overview.docx.md
├── DSC2026_Task2_LegalQA_Data_Overview.docx.md
├── IR/
│   ├── train.json
│   ├── warmup.json
│   └── public-official.json
├── QA/
│   ├── train.json
│   ├── warmup.json
│   └── public-official.json
├── research_method/                         # các phiên bản nghiên cứu phương pháp IR/QA
│   └── research_v1.md                       # draft phương pháp LegalIR đầu tiên
├── retrieval/                               # code pipeline khung Task 1
│   ├── configs/default.yaml                 # model/top-k/RRF/reranker config
│   ├── pyproject.toml                       # package và dependency
│   ├── README.md                            # chunk/split/evaluate/build/search/ablation
│   ├── src/legal_ir/                        # chunker, split/evaluate, retrieval và CLI
│   └── tests/                               # unit test dùng mock, không tải model
├── selected-contexts/                       # 8.532 context_*.json, khoảng 483 MB
└── test.ipynb                               # notebook khảo sát đơn giản, không phải baseline
```

Corpus/dữ liệu gốc chiếm khoảng 501 MB. Workspace hiện có README, source và dependency cho **retrieval inference/indexing skeleton**, nhưng chưa có code training/fine-tuning, `AGENTS.md` hoặc thư mục `.git`; đừng dựa vào lệnh Git để lấy trạng thái thay đổi. Có một `.DS_Store` không liên quan.

Tài liệu overview có nêu `private-official.json` và `selected-contexts.zip` theo timeline cuộc thi, nhưng **các tệp private không có trong workspace** và corpus đã được giải nén ở `selected-contexts/`.

## Corpus văn bản chung

Mỗi tệp `selected-contexts/context_<document_id>.json` là một JSON object. Tất cả 8.532 tệp đều parse JSON hợp lệ; tên tệp luôn khớp chính xác với trường `id` bên trong và mọi ID là duy nhất.

Schema thực tế:

```json
{
  "id": 740,
  "link": "https://thuvienphapluat.vn/...",
  "name": "Quyet-dinh-...",
  "passage": "toàn văn/nội dung văn bản..."
}
```

- `id`, `link`, `passage` có trong mọi document. `id` là **số**.
- `name` là tùy chọn: 7.407 document có đủ bốn trường, còn 1.125 document chỉ có `id`, `link`, `passage`. Code index/đọc dữ liệu không được giả định `name` luôn tồn tại; có thể dùng slug của `link` làm fallback.
- Mọi `link` trỏ tới `thuvienphapluat.vn`.
- Có 20 `passage` rỗng; nên loại khỏi chỉ mục hoặc xử lý an toàn. Với các passage không rỗng: trung vị 23.210 ký tự UTF-16, p95 135.624, lớn nhất 5.983.358. Tránh nạp/tách toàn bộ corpus vào prompt hoặc bộ nhớ không kiểm soát.
- Tiền tố `name` phổ biến: `Thong-tu` 2.748, `Quyet-dinh` 2.650, `Nghi-dinh` 1.101, `Nghi-quyet` 286. Corpus là văn bản hành chính/pháp lý đầy đủ, không phải các đoạn ngắn đã chunk sẵn.

## Định dạng dữ liệu câu hỏi

Mọi tệp dưới `IR/` và `QA/` là một JSON object, map từ **query ID dạng chuỗi** sang record. Mỗi record luôn có đúng hai trường `question` và `answer`.

### LegalIR

Với train/warmup, `answer` là mảng các **document ID dạng chuỗi**:

```json
{
  "86666": {
    "question": "Thời hạn cấp đăng ký xe máy... là bao lâu?",
    "answer": ["280282"]
  }
}
```

| Tệp | Số câu | Nhãn | Phân phối số doc/câu |
| --- | ---: | --- | --- |
| `IR/train.json` | 7.000 | Có | 1: 6.447; 2: 485; 3: 53; 4: 14; 5: 1 |
| `IR/warmup.json` | 500 | Có | 1: 463; 2: 33; 3: 3; 4: 1 |
| `IR/public-official.json` | 1.000 | Không — mọi `answer` là `null` | Không áp dụng |

- Không có câu hỏi/nhãn rỗng trong train và warmup. Câu hỏi có trung vị 86 và 85 ký tự tương ứng.
- Train tham chiếu 7.637 lần tới 3.105 document riêng biệt (36,4% corpus); mọi ID nhãn đều tồn tại trong corpus. Warmup tham chiếu 542 lần tới 426 document riêng biệt.
- Không có nhãn train/warmup nào vượt quá 5 document. Vẫn phải tự kiểm tra output: quá 5 ID ở một câu khiến **cả Recall và Precision của câu đó bằng 0**.

### LegalQA

Với train/warmup, `answer` là chuỗi văn xuôi tiếng Việt, thường chứa trích dẫn điều/khoản và liệt kê chi tiết:

```json
{
  "82051": {
    "question": "Vận chuyển động vật ... thì bị xử phạt thế nào?",
    "answer": "Căn cứ khoản 3, khoản 5 Điều 17 Nghị định ..."
  }
}
```

| Tệp | Số câu | Nhãn | Độ dài answer (trung vị / p95 / lớn nhất) |
| --- | ---: | --- | --- |
| `QA/train.json` | 7.000 | Có, chuỗi | 1.410 / 3.145 / 10.755 ký tự |
| `QA/warmup.json` | 500 | Có, chuỗi | 1.370 / 2.967 / 8.089 ký tự |
| `QA/public-official.json` | 1.000 | Không — mọi `answer` là `null` | Không áp dụng |

- Câu hỏi QA có trung vị 85, 83 và 85 ký tự theo thứ tự train/warmup/public; không có câu hỏi rỗng.
- Các đáp án tham chiếu tương đối dài và giàu chi tiết. Tối ưu QA nên tính đến mức giống tham chiếu theo token/thứ tự, không chỉ đúng ý ở mức rất ngắn.
- Tài liệu Task 2 hiện chỉ mô tả metric, **không có phần định dạng submission**. Không tự khẳng định schema nộp QA nếu chưa có hướng dẫn bổ sung từ ban tổ chức.

## Split overlap cần biết

Các split không hoàn toàn tách rời. Đây là dữ kiện đã kiểm tra bằng ID và nội dung câu hỏi chính xác:

| Tác vụ | Cặp split | Query ID trùng | Nhận xét |
| --- | --- | ---: | --- |
| IR | train ↔ warmup | 346 | Câu hỏi và nhãn giống hệt nhau |
| IR | warmup ↔ public | 52 | Câu hỏi giống hệt; public có `answer: null` |
| IR | train ↔ public | 0 | Có 5 câu trùng nguyên văn nhưng khác ID |
| QA | train ↔ warmup | 387 | Câu hỏi và đáp án giống hệt nhau |
| QA | warmup ↔ public | 40 | Câu hỏi giống hệt; public có `answer: null` |
| QA | train ↔ public | 0 | Có 1 câu trùng nguyên văn nhưng khác ID |

Các con số trên mô tả dữ liệu hiện diện, không thay thế quy định của cuộc thi. Khi đánh giá mô hình nội bộ, cần tách/ghi nhận các câu lặp để tránh báo cáo validation bị lạc quan quá mức và tuân thủ quy tắc cuộc thi khi dùng chúng.

## Quy tắc đánh giá và output IR

Theo `DSC2026_Task1_LegalIR_Data_Overview.docx.md`:

- Recall được tính theo từng câu là tỷ lệ document đúng được trả về, sau đó lấy trung bình các câu; đây là điểm xếp hạng chính.
- Precision là tỷ lệ document trả về đúng, cũng trung bình theo câu, là tiêu chí phụ khi bằng Recall. Nếu không trả về document thì precision câu đó là 0.
- Bài IR phải là `submission.zip` chứa duy nhất `submission.json`. Schema được nêu:

```json
{
  "147194": {"answer": ["177504", "740"]}
}
```

- Tối đa 5 document ID mỗi query. Giữ document ID dưới dạng chuỗi khi serialize, dù trường `id` trong context là số.

Theo `DSC2026_Task2_LegalQA_Data_Overview.docx.md`, LegalQA xếp hạng bằng METEOR (chính) và ROUGE-L (phụ). Cả hai overview đều xác định `passage` là căn cứ để retrieval/QA.

## Notebook và tình trạng kỹ thuật

`test.ipynb` chỉ có các cell thử nghiệm:

- In passage của `selected-contexts/context_30639.json` (tệp này tồn tại).
- Đọc `IR/train.json`, liệt kê các câu có nhiều hơn một document đúng, đếm phân phối số document đúng, và vẽ biểu đồ bằng `matplotlib`.

Notebook không chứa retriever, model QA, evaluation script, submission generator, hoặc môi trường tái lập. Pipeline Task 1 mới nằm riêng trong `retrieval/`; không nên coi notebook là code nền tảng.

## Khung retrieval Task 1 hiện có

`retrieval/` được tạo ngày 2026-08-19 để nhận dữ liệu sau khi chunking. Ngày 2026-08-20 đã viết fixed-token chunker dành cho Kaggle, nhưng chưa chạy và chưa có `artifacts/chunks`, index hay model checkpoint local. Structural chunking vẫn chưa được triển khai. Không đưa trực tiếp whole document rất dài vào pipeline.

Kiến trúc đã code:

```text
BM25(original query) ──────────────────┐
Vietnamese_Embedding_v2(query) ────────┼─ MaxP chunk→document
Vi-Qwen2-3B-RAG → HyDE → dense(HyDE) ──┘          ↓
                                       weighted document-level RRF
                                                    ↓ top candidates
                         Vietnamese_Reranker(query gốc, chunk thật)
                                                    ↓
                                          tối đa 5 document IDs
```

Các quyết định quan trọng:

- Default model từ ngày 2026-08-20: `AITeamVN/Vietnamese_Embedding_v2` (567.754.752 tham số), `AITeamVN/Vietnamese_Reranker` (567.755.777) và `AITeamVN/Vi-Qwen2-3B-RAG` (3.085.938.688). Tổng unique parameters chính xác là 4.221.449.217, vượt 4B khoảng 5,54%; người dùng đã chấp nhận sai số này cho pipeline Task 1. Config pin full revision SHA của cả ba checkpoint.
- `Vietnamese_Embedding_v2` dùng CLS pooling, L2 normalization, vector 1024 chiều; dense index vẫn là FAISS inner product. `Vietnamese_Reranker` là `XLMRobertaForSequenceClassification` một logit, được gọi qua `CrossEncoder` với Identity activation chứ không dùng như bi-encoder.
- Với `dtype: auto`, code dùng BF16 trên CUDA có hỗ trợ, FP16 trên CUDA còn lại/MPS và FP32 trên CPU. Vi-Qwen được ép `use_cache=True`; adapter Qwen2 không truyền `enable_thinking` của Qwen3.
- Không so sánh/cộng raw BM25, cosine và HyDE score. Từng lane gom chunk về document bằng max score, sau đó fusion rank bằng weighted RRF (`k=60`; weight mặc định 1,0/1,0/0,5).
- HyDE là dense-only lane. Không chạy BM25 trên đoạn do SLM sinh; reranker chỉ nhận query gốc và chunk thật.
- Text do HyDE sinh được canonicalize bằng policy versioned `hyde_nfc_ws_v1` tại generator, cache và trước dense retrieval: Unicode NFC; line break/tab thật và literal `\\n`/`\\r`/`\\t`; HTML non-breaking-space allowlist; control/zero-width characters; code fence và whitespace thừa được dọn. Không lowercase, bỏ dấu, sửa punctuation, số hoặc viện dẫn pháp luật. Version nằm trong cache namespace nên cache cũ tự miss thay vì đưa raw hypothesis vào dense lane.
- Default lấy BM25 top 300 chunks, dense query top 200, dense HyDE top 200, fuse top 50 documents, rerank tối đa 2 evidence chunks/document, output 5 document IDs. Đây là giá trị khởi đầu, chưa được tune trên DSC.
- Input contract là JSONL gồm `chunk_id`, `document_id`, `passage`, `retrieval_text` tùy chọn và `metadata`. `retrieval_text` nên ghép metadata title/Điều/Khoản với passage; nếu thiếu thì dùng `passage`.
- BM25, FAISS và chunk store dùng chung stable row order; manifest kiểm tra hash mapping/nội dung. Output luôn dùng `document_id` dạng chuỗi.
- CLI hỗ trợ build index, search một query, search cả split, HyDE JSONL cache, diagnostics và các flag `--disable-hyde`, `--disable-reranker` cho ablation.
- Unit test dùng mock backend, kiểm tra document-level RRF, BM25 zero-score padding, HyDE dense-only/cache, evidence grounding, reranker dùng query gốc/chunk thật, mapping chunk→document, config và schema tối đa 5 ID. Lệnh kiểm tra hiện tại:

```bash
PYTHONPATH=retrieval/src python -m unittest discover -s retrieval/tests -v
```

Sau lần đổi model stack, 23 test đều pass, `compileall` và CLI `--help` smoke-test thành công. Chưa chạy end-to-end model thật vì chưa có dữ liệu chunk; các model/dependency nặng cũng chưa được tải trong workspace. Đọc `retrieval/README.md` trước khi build index.

Model card của `Vi-Qwen2-3B-RAG` đang có nội dung/lineage sao chép từ bản 7B dù config và weights là kiến trúc 3B; không dùng benchmark hoặc chuỗi license trong card như bằng chứng chắc chắn nếu chưa xác minh với tác giả. Checkpoint này được fine-tune để trả lời từ context, nên hiệu quả khi dùng làm HyDE generator vẫn là giả thuyết cần ablation, không phải kết luận đã được kiểm chứng.

### Fixed-size preprocessing cho Kaggle

Ngày 2026-08-20 đã thêm `retrieval/src/legal_ir/chunk_fixed_size.py` và console entrypoint `legal-ir-chunk-fixed-size`. Đây là baseline preprocessing riêng, không phải structural chunking:

- đọc streaming `selected-contexts/context_*.json`, sort theo numeric document ID và validate filename ↔ `id`;
- normalize bằng `legal_nfc_ws_v1`: chỉ decode HTML non-breaking-space entity, Unicode NFC, line ending, rồi collapse whitespace/control/invisible characters; không lowercase, bỏ dấu hoặc sửa số/citation;
- dùng fast tokenizer của `AITeamVN/Vietnamese_Embedding_v2` tại revision đã pin;
- default 384 content tokens, overlap 64; passage được slice bằng offset trên normalized text;
- output dự kiến `/kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl` cùng `chunk_fixed_size.manifest.json`;
- output đúng contract `chunk_id`, `document_id`, `passage`, `retrieval_text`, `metadata`; chunk ID chứa normalization version/size/overlap;
- skip passage rỗng có ghi manifest, còn JSON/schema/duplicate/mismatch lỗi thì fail-fast;
- hỗ trợ tokenizer cache và `--local-files-only` cho Kaggle không Internet.

Theo yêu cầu người dùng, chunker này **chỉ được viết để đưa lên Kaggle và chưa được chạy/compile/test tại local**. Con số 23 test phía trên là kết quả trước khi thêm chunker; không được diễn giải là runtime verification cho `chunk_fixed_size.py`. Chưa có `chunk_fixed_size.jsonl` hay manifest thật trong workspace.

### Internal split và evaluation Task 1

Ngày 2026-08-20 đã thêm hai utility chỉ dùng Python standard library:

- `retrieval/src/legal_ir/create_val_test.py` cùng entrypoint `legal-ir-create-val-test`: lấy mẫu không hoàn lại từ `IR/train.json`, mặc định seed 2026 và split 5.600/700/700; ghi `train.json`, `val.json`, `test.json`, `split_manifest.json` vào output directory mới, không sửa dữ liệu gốc.
- Split được random ở cấp nhóm câu hỏi sau NFC + `casefold` + collapse whitespace. 16 nhóm câu hỏi trùng trong train vì vậy không bị tách qua các split; manifest lưu SHA-256, phân phối số gold documents và thống kê 5 nhóm duplicate có nhãn xung đột. Baseline này chưa stratify theo số gold documents; không nhập warmup vào train mới nếu chưa kiểm tra overlap.
- `retrieval/src/legal_ir/evaluate_recall_precision.py` cùng entrypoint `legal-ir-evaluate`: tính official macro Recall và macro Precision trên toàn bộ gold queries. Query thiếu được tính như prediction rỗng, query dư được báo/không vào mẫu số, raw answer dài hơn 5 nhận 0/0, duplicate ID được báo là contract invalid.
- Gold phải có `answer` là list ID chuỗi không rỗng; `IR/public-official.json` có nhãn null nên evaluator chủ động từ chối.
- 11 unit tests cho hai utility đã pass; sau khi thêm HyDE normalization/cache invariant, toàn bộ suite hiện là 39/39 test, không tải model/GPU. Default split đã được kiểm tra read-only trên dữ liệu thật và cho đúng 5.600/700/700; chưa tạo split artifact thật trong workspace.

## Thư mục nghiên cứu phương pháp

`research_method/` dùng để lưu các bản nghiên cứu, đề xuất kiến trúc, ablation plan và kết luận thực nghiệm cho LegalIR/LegalQA. Đây là nơi agent cần kiểm tra trước khi đề xuất hoặc thay đổi method, nhằm tránh lặp lại nghiên cứu đã hoàn thành.

Quy ước:

- Đặt tên tuần tự `research_v1.md`, `research_v2.md`, ...; không ghi đè bản cũ khi có thay đổi lớn về giả thuyết, pipeline hoặc kết luận.
- Mỗi bản phải ghi ngày tạo, phạm vi Task 1/Task 2, trạng thái draft hay đã được thực nghiệm, nguồn tham khảo và khác biệt so với bản trước.
- Kết quả thực nghiệm phải ghi rõ split, cách xử lý duplicate/leakage, metric và artifact/code đã dùng.
- Nghiên cứu hiện có: `research_method/research_v1.md` — draft ban đầu cho Task 1, tập trung vào dataset legal retrieval, structural chunking, hybrid BM25+dense, document aggregation, reranking và hard-negative training.
- `research_v1.md` là đề xuất nghiên cứu, chưa phải benchmark đã được xác nhận trên dữ liệu DSC.

## Hướng dẫn làm việc cho agent tiếp theo

1. Đọc tệp này, sau đó đọc hai overview nếu cần chi tiết quy tắc chính thức.
2. Với công việc về method, đọc các bản liên quan trong `research_method/` trước; tạo version mới thay vì ghi đè lịch sử nghiên cứu.
3. Giữ nguyên dữ liệu gốc trong `IR/`, `QA/`, `selected-contexts/`; đặt index, cache, output và thử nghiệm vào thư mục mới rõ ràng.
4. Luôn mở JSON với UTF-8 và xử lý `answer: null` ở public như nhãn không công khai, không phải record lỗi.
5. Khi nối nhãn IR với corpus, chuẩn hóa bằng `str(document["id"])`; không giả định có `name` hay passage không rỗng.
6. Với corpus dài, ưu tiên indexing/chunking có kiểm soát và lưu mapping chunk → document ID để output IR luôn là ID của **văn bản gốc**.
7. Trước khi tạo submission IR, kiểm tra đủ query ID đích, mỗi `answer` là mảng ID chuỗi, không trùng lặp, và độ dài từ 1 đến 5 (hoặc chấp nhận rỗng một cách có chủ đích theo chiến lược).


## Lưu ý
- Mục tiêu chung vẫn là tổng model dưới 4B params và không dùng API trả phí. Ngoại lệ hiện tại chỉ dành cho pipeline Task 1 AITeamVN nói trên: 4,221B params, đã được người dùng chấp nhận rõ ràng ngày 2026-08-20.
