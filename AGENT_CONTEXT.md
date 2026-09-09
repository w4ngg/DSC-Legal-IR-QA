# DSC Legal — Context cho các session sau

> Snapshot được audit lại ngày **2026-09-03** từ source, tài liệu, dữ liệu gốc, index và run artifact hiện có; context cho dual long-reranking được cập nhật ngày **2026-09-05**. HEAD lúc audit là `65ca8fb`; một số thay đổi mới vẫn chưa commit và được ghi rõ bên dưới. Đọc tệp này đầu tiên khi bắt đầu làm việc trong workspace.

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
├── analysis/                                # script offline recall/candidate-grid + tests
├── dual_chunks_v1.zip                      # short/long/mapping/manifest dual; bị gitignore
├── indexes/
│   ├── vietlegal_harrier_06b_v1/           # index Harrier thật; bị gitignore
│   ├── vn_embedding_v2_dual_v1/            # index 2.050.281 short chunks
│   └── vn_embedding_v2_dual_v1.zip         # bundle index dual khoảng 5 GiB
├── indexes.zip                              # bản nén của index; bị gitignore
├── research_method/                         # các phiên bản nghiên cứu phương pháp IR/QA
│   ├── research_v1.md                       # draft phương pháp LegalIR đầu tiên
│   └── method.md                            # roadmap cải tiến và ablation theo diagnostics
├── retrieval/                               # code pipeline khung Task 1
│   ├── configs/default.yaml                 # model/top-k/RRF/reranker config
│   ├── configs/vietlegal_harrier.yaml       # preset Harrier 0.6B, cần index riêng
│   ├── configs/vietnamese_embedding_dual.yaml # preset build/search index short hiện có
│   ├── configs/vietnamese_embedding_dual_long_rerank.yaml # B50/D100 + long rerank T4x2
│   ├── pyproject.toml                       # package và dependency
│   ├── README.md                            # chunk/split/evaluate/build/search/ablation
│   ├── src/legal_ir/                        # chunker, retrieval, CLI và isolated GPU worker
│   └── tests/                               # unit test dùng mock, không tải model
├── runs/                                    # validation/public diagnostics và submissions; bị gitignore
├── selected-contexts/                       # 8.532 context_*.json, khoảng 483 MB
└── test.ipynb                               # notebook khảo sát đơn giản, không phải baseline
```

Corpus/dữ liệu gốc chiếm khoảng 501 MB. Ngày 2026-09-05, `indexes/` chiếm khoảng 19 GiB vì chứa index Harrier 2,8 GiB, index short Vietnamese Embedding dual khoảng 12 GiB và ZIP dual-index khoảng 5 GiB; ngoài ra có `indexes.zip` 987 MB, `dual_chunks_v1.zip` 321 MB và `runs/` khoảng 445 MB. Workspace đã có implementation retrieval/indexing cùng artifact từ các lần chạy thật, nhưng system Python hiện tại chưa cài đủ dependency runtime. Vẫn chưa có code training/fine-tuning, pipeline LegalQA hoặc `AGENTS.md`; parser hierarchy pháp lý đầy đủ cũng chưa được triển khai. Repo có `.git`; giữ nguyên mọi thay đổi người dùng/unrelated trong worktree và không xóa các artifact bị ignore nếu chưa được yêu cầu. Có nhiều `.DS_Store`/`__MACOSX` không liên quan.

Tài liệu overview có nêu `private-official.json` và `selected-contexts.zip` theo timeline cuộc thi, nhưng **các tệp private không có trong workspace** và corpus đã được giải nén ở `selected-contexts/`.

Trạng thái worktree ngày 2026-09-05 còn **chưa commit**: `.gitignore`, context/README/default config và các module CLI/config/indexing/io/pipeline/reranker/schema có thay đổi; config dual-long, `dual_rerank.py`, worker reranker multi-GPU cùng test tương ứng là file mới. `analysis/` cũng có các script/test recall và candidate-grid mới được whitelist bởi `.gitignore`. `indexes/`, các ZIP, `runs/` và Python cache vẫn bị ignore nên không đi theo clone/commit. Phải commit/push source mới trước khi notebook Kaggle `git clone` có thể dùng mode này.

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
- Có 20 `passage` rỗng. Index hiện tại bỏ đúng 20 document này; 6 trong số đó (`10533`, `55497`, `131890`, `149317`, `263763`, `288457`) xuất hiện trong gold của 11 query IR train. Vì vậy cần metadata-only fallback hoặc xử lý riêng: nếu mọi document khác đều hoàn hảo, việc thiếu chúng vẫn giới hạn macro Recall train ở khoảng 0,998571. Internal val hiện có 2 query single-label thuộc nhóm này, nên trần tương ứng là khoảng 0,997143.
- Với 8.512 passage không rỗng: trung vị chuẩn là 23.215 Unicode code point; trung vị theo UTF-16 code unit tình cờ cũng bằng 23.215. P95 khoảng 135,6 nghìn code point và lớn nhất 5.983.358. Tránh nạp/tách toàn bộ corpus vào prompt hoặc bộ nhớ không kiểm soát.
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
| `QA/warmup.json` | 500 | Có, chuỗi | 1.373 / 2.967 / 8.089 ký tự |
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

`test.ipynb` chỉ có 7 cell khảo sát (cell cuối rỗng):

- In passage của `selected-contexts/context_56852.json`.
- Đọc `IR/train.json`, liệt kê các câu có nhiều hơn một document đúng, đếm phân phối số document đúng, và vẽ biểu đồ bằng `matplotlib`.

Notebook không chứa retriever, model QA, evaluation script, submission generator, hoặc môi trường tái lập. Pipeline Task 1 mới nằm riêng trong `retrieval/`; không nên coi notebook là code nền tảng.

## Khung retrieval Task 1 hiện có

`retrieval/` được tạo ngày 2026-08-19; fixed-token chunker được thêm ngày 2026-08-20. Workspace hiện có index Harrier và index short Vietnamese Embedding dual hoàn chỉnh cùng các run thật trong thư mục bị gitignore, nhưng không có model checkpoint/cache Hugging Face trong repo. Dual clause-character chunker đã có; parser hierarchy pháp lý đầy đủ vẫn chưa triển khai. Không đưa trực tiếp whole document rất dài vào pipeline.

Kiến trúc legacy vẫn là mặc định:

```text
BM25(original query) ──────────────────┐
Dense SentenceTransformer(query) ──────┼─ MaxP chunk→document
Vi-Qwen2-1.5B-RAG → HyDE → dense(HyDE) ─┘          ↓
                                       weighted document-level RRF
                                                    ↓ top candidates
                         Vietnamese_Reranker(query gốc, chunk thật)
                                                    ↓ MaxP evidence mặc định
                                          tối đa 5 document IDs
```

Mode dual long-context mới là opt-in và giữ nguyên index short:

```text
BM25@50 short ───────┐
Dense@100 short ─────┼─ union/dedup short IDs → map/dedup long IDs
HyDE short (optional)┘                         ↓ full hoặc cutoff top-C
                           Vietnamese_Reranker(query gốc, mọi long candidate)
                                                  ↓ reranker top 20 long
                                           MaxP long → document
                                                  ↓
                                         tối đa 5 document IDs
```

Các quyết định quan trọng:

- Default HyDE được đổi ngày 2026-08-24 sang `AITeamVN/Vi-Qwen2-1.5B-RAG`, revision `c8272ce4ad08da4cc27b4bda59faabc66caedf07`, chuẩn `Qwen2ForCausalLM`, 1.543.714.304 BF16 params. Cùng `AITeamVN/Vietnamese_Embedding_v2` (567.754.752) và `AITeamVN/Vietnamese_Reranker` (567.755.777), baseline có tổng chính xác 2.679.224.833 tham số và nằm dưới 4B.
- `Vietnamese_Embedding_v2` dùng CLS pooling, L2 normalization, vector 1024 chiều; dense index vẫn là FAISS inner product. `Vietnamese_Reranker` là `XLMRobertaForSequenceClassification` một logit, được gọi qua `CrossEncoder` với Identity activation chứ không dùng như bi-encoder.
- Dense ablation mới nằm ở `retrieval/configs/vietlegal_harrier.yaml`: `mainguyen9/vietlegal-harrier-0.6b`, revision `91a0e1ebe4b63b4475bbae40658b8ca9231bea74`, 596.049.920 tham số, Qwen3 backbone, last-token pooling, vector normalized 1024 chiều và native max length 512. Preset dùng explicit FP16 cho encoder forward trên T4; vector vẫn được cast float32 khi thêm vào FAISS. Cả stack Harrier + reranker + HyDE 1.5B là 2.707.520.001 tham số, cao hơn baseline khoảng 28,3M nhưng vẫn dưới 4B.
- Dense adapter là generic Sentence Transformers v5: query thật gọi `encode_query()` để Harrier tự dùng saved prompt `query`; corpus chunk và HyDE hypothetical passage gọi `encode_document()` nên không nhận query instruction. AITeamVN không lưu prompt nên hai role tương đương encode cũ. Không tự prepend prompt vào dữ liệu chunk.
- Ngày 2026-08-25, đường build dense T4×2 đã bỏ shared-parent multiprocessing của SentenceTransformers vì Harrier có thể treo khi một worker chết nhưng parent vẫn block trên output queue. `multi_gpu.py` resolve một immutable HF snapshot, chia contiguous shard, mở `python -m legal_ir.dense_worker` độc lập cho từng visible GPU và chỉ trao đổi JSONL/NPY/status/log trên disk. Mỗi worker bị cô lập bằng `CUDA_VISIBLE_DEVICES`, tự load model trên logical `cuda:0`, encode document blocks và heartbeat; parent poll exit code/stall timeout, terminate peer khi lỗi, validate shape/order/finiteness và ghép shard theo rank. Không model hoặc CUDA tensor nào đi qua shared memory/queue.
- Kaggle T4×2 có hai T4 16 GB riêng nhưng chỉ 4 CPU core/29 GB RAM host. Worker tự cap khoảng `floor(cpu_count/GPU_count)` host threads; `batch_size` vẫn là mỗi GPU. `multi_process_chunk_size` nay là macro-block/heartbeat, mặc định nội bộ 256 nếu null; preset Harrier đặt 256. `multi_gpu_stall_timeout_seconds` mặc định 1800 và chỉ là runtime config. Query/HyDE dense inference vẫn single-GPU.
- Dual long-context dùng preset `retrieval/configs/vietnamese_embedding_dual_long_rerank.yaml`: BM25 lấy top 50 short chunks, dense lấy top 100, HyDE mặc định tắt nhưng có thể bật thành lane dense(HyDE), reranker pretrained bật, `reranker.multi_gpu: true`, post-reranker top 20 long chunks và final tối đa 5 documents. Existing `default.yaml`, `vietlegal_harrier.yaml` và `vietnamese_embedding_dual.yaml` vẫn chạy flow legacy; không tự chuyển mode cũ sang long reranking.
- Index dual vẫn chỉ chứa short chunks: `build-index` nhận `short_chunks.jsonl`, còn `long_chunks.jsonl`/`short_to_long.jsonl` là artifact runtime riêng, không được encode vào BM25/FAISS. Search long-context nhận thư mục dual bằng `--dual-chunks-dir`; runtime nạp long chunks và tái dùng mapping đã nhúng trong `INDEX_DIR/chunks.jsonl` để tránh một dictionary hai triệu dòng. Startup stream-compare toàn bộ standalone mapping với metadata index và kiểm tra SHA-256 mapping/long chunks theo manifest. Loader mode dual giữ compact short/long records (ID, `index_text`, mapping/granularity cần thiết) thay vì toàn bộ metadata và hai bản text, nhằm giảm host RAM mà không đổi scoring. Không hard-code đường Kaggle trong YAML để cùng preset chạy được local và notebook.
- Candidate pool long được tạo theo đúng thứ tự: lấy per-lane short top-k → union/dedup `short_chunk_id` → map qua toàn bộ `long_chunk_ids` → dedup `long_chunk_id` → xếp retrieval support bằng weighted short-rank RRF. `long_context.candidate_mode: full` đi tiếp với toàn bộ unique long (`candidate_top_k: null`); mode `cutoff` lấy top `candidate_top_k` sau mapping/dedup và trước reranker.
- Trong mode long, `fusion.candidate_documents` và `fusion.evidence_chunks_per_document` không được dùng; chỉ `fusion.rrf_k`/`channel_weights` được tái dùng để pre-rank long candidates. Deep diagnostics là raw lane trace trước short→long mapping, không phải trace của legacy document-fusion cutoff.
- Reranker score mọi pair `(original_query, selected_long_chunk.retrieval_text)`, không dùng HyDE text làm query. Sau khi có toàn bộ score, pipeline giữ reranker top 20 long chunks rồi MaxP về document và trả tối đa 5 documents. Vì vậy `rerank_top_k_chunks: 20` là post-score cutoff, không phải chỉ score 20 pair.
- `diagnostics.json` trong mode mới giữ toàn bộ long candidates đã score, không chỉ top 20/5: count short union, mapping occurrences, unique long trước/sau cutoff, retrieval support/rank, reranker score/rank và provenance short/lane; đồng thời giữ top long và document ranking sau MaxP. Cấu hình ablation đặt `diagnostics_store_all_candidates: true` để có thể replay/phân tích mà không gọi model lại.
- Multi-GPU reranker khác multi-GPU build index: với mỗi query, passage list được chia thành contiguous shard cho hai persistent worker, mỗi T4 giữ một model replica, rồi parent ghép score về đúng thứ tự input. Đây không phải hai GPU hợp thành 32 GB và cũng không phải chia query độc lập giữa GPU. Timeout dùng `reranker.multi_gpu_stall_timeout_seconds`; scratch tùy chọn qua `LEGAL_IR_RERANKER_MULTI_GPU_TMPDIR`.
- `sentence-transformers==5.1.2` và `transformers==4.57.6` được pin trong `pyproject.toml`: source cần role-aware ST v5 API, còn Harrier công bố metadata Transformers 4.57.6; không để scheduled Kaggle run tự nâng sang Transformers 5 loader. Parent thêm source root vào worker `PYTHONPATH`, nên direct notebook API vẫn dùng được ngay cả khi project chỉ được thêm vào `sys.path`.
- Đổi `model_name`, `revision`, `max_length`, dtype/normalization hoặc FAISS type phải build một index directory mới. Đổi batch/multi-GPU settings không làm index cũ mất hiệu lực. Chỉ đổi dense model không invalidate HyDE JSONL cache; cache phụ thuộc HyDE model/prompt/generation/normalization, không phụ thuộc dense checkpoint.
- Ngược lại, đổi HyDE 3B → 1.5B không cần build lại BM25/dense index nhưng làm cache namespace đổi hoàn toàn. Dùng file mới như `hyde_vi_qwen2_1_5b.jsonl`; cache 3B là artifact lịch sử, không được tái sử dụng cho run mới.
- Với `dtype: auto`, code dùng BF16 trên CUDA có hỗ trợ, FP16 trên CUDA còn lại/MPS và FP32 trên CPU. Vi-Qwen được ép `use_cache=True`; adapter Qwen2 không truyền `enable_thinking` của Qwen3.
- Không so sánh/cộng raw BM25, cosine và HyDE score. Từng lane gom chunk về document bằng max score, sau đó fusion rank bằng weighted RRF (`k=60`; weight mặc định 1,0/1,0/0,5).
- Có **hai phép MaxP khác nhau**: (1) MaxP chunk → document bên trong từng retrieval lane trước RRF; (2) MaxP các `evidence_rerank_scores` → document sau cross-encoder. Utility top-2 mean mới chỉ thay phép (2), không thay candidate pool, evidence selector hoặc MaxP trước fusion.
- HyDE là dense-only lane. Không chạy BM25 trên đoạn do SLM sinh; reranker chỉ nhận query gốc và chunk thật.
- Text do HyDE sinh được canonicalize bằng policy versioned `hyde_nfc_ws_v1` tại generator, cache và trước dense retrieval: Unicode NFC; line break/tab thật và literal `\\n`/`\\r`/`\\t`; HTML non-breaking-space allowlist; control/zero-width characters; code fence và whitespace thừa được dọn. Không lowercase, bỏ dấu, sửa punctuation, số hoặc viện dẫn pháp luật. Version nằm trong cache namespace nên cache cũ tự miss thay vì đưa raw hypothesis vào dense lane.
- Checked-in `default.yaml` hiện lấy BM25 top 300 chunks, dense query top 200, dense HyDE top 200, fuse top 30 documents, rerank tối đa 3 evidence chunks/document, output 5 document IDs. Đây là giá trị đang thử nghiệm, chưa được tune đầy đủ trên DSC.
- CLI **không tự đọc** `default.yaml`. Nếu bỏ `--config`, `PipelineConfig()` dùng dense batch 8, fusion candidates 50 và 2 evidence/document, khác YAML (32/30/3). Luôn truyền config và lưu resolved config của run; test hiện chỉ khóa model/revision chứ không bảo đảm mọi Python default trùng YAML.
- Input contract là JSONL gồm `chunk_id`, `document_id`, `passage`, `retrieval_text` tùy chọn và `metadata`. `retrieval_text` nên ghép metadata title/Điều/Khoản với passage; nếu thiếu thì dùng `passage`.
- BM25, FAISS và chunk store dùng chung stable row order; manifest kiểm tra hash mapping/nội dung. Output luôn dùng `document_id` dạng chuỗi.
- CLI hỗ trợ build index, search một query, search cả split, HyDE JSONL cache, diagnostics và các flag `--disable-hyde`, `--disable-reranker` cho ablation legacy; long-context bắt buộc reranker nên không dùng `--disable-reranker` với preset mới. Batch search có thêm `--deep-diagnostics PATH`: ghi streaming/atomic một JSON riêng chứa toàn bộ BM25, dense và HyDE trước fusion cutoff, gồm canonical chunk ranks/raw scores và document ranks sau MaxP; tên file do người gọi chọn, artifact hiện tại dùng `deep_diagnostics.json`. Long-context search thêm `--dual-chunks-dir PATH`; dùng cùng index short và preset `vietnamese_embedding_dual_long_rerank.yaml`. `diagnostics.json` legacy giữ `fused_candidates`/evidence score; mode long giữ toàn bộ long-candidate diagnostics. Deep diagnostics không chứa passage/metadata; join short hit bằng `chunk_id` với đúng `INDEX_DIR/chunks.jsonl`.
- Utility mới `legal_ir.diagnostics_top2_mean_submission` / entrypoint `legal-ir-diagnostics-top2-mean` chỉ replay schema diagnostics legacy: xếp candidate bằng trung bình tối đa 2 reranker score tốt nhất, tie-break bằng fusion score rồi document ID, và ghi tối đa 5 ID/query. Nó yêu cầu diagnostics được tạo với reranker bật và chỉ replay pool/evidence đã log; chưa dùng utility này để thay aggregation của mode long. Vì source, test và entrypoint đang chưa commit, console script cần `pip install -e` lại; gọi `python -m ...` với đúng `PYTHONPATH` thì dùng trực tiếp được.
- Utility độc lập `analysis/evaluate_retrieval_recall.py` chấm retrieval recall từ `deep_diagnostics.json` với cutoff do người dùng truyền riêng cho BM25/dense/HyDE optional. Nó luôn truncate `chunk_hits` trước rồi mới deduplicate/MaxP về document, báo macro/micro Recall, Hit, FullHit cho từng lane và `oracle_union`; nếu có thêm `diagnostics.json` thì chấm `fused_candidates` và `results` ở các document cutoff được chọn. Bỏ `--hyde-top-k` nghĩa là bỏ HyDE khỏi cả lane report lẫn union. Script chỉ dùng standard library, có test độc lập trong `analysis/test_evaluate_retrieval_recall.py` và hướng dẫn tại `analysis/README.md`.
- Unit test dùng mock backend, kiểm tra document-level RRF, BM25 zero-score padding, HyDE dense-only/cache, evidence grounding, reranker dùng query gốc/chunk thật, mapping chunk→document, config, schema tối đa 5 ID và top-2 mean replay. Lệnh kiểm tra hiện tại:

```bash
PYTHONPATH=retrieval/src python -m unittest discover -s retrieval/tests -v
```

Ngày 2026-09-03, **62/62** unit tests không tải model/GPU đều pass. Sau khi tích hợp dual long-reranking ngày 2026-09-05, suite tăng lên **93/93** tests pass; `compileall` cho `retrieval/src`, `retrieval/tests` và `analysis` cũng pass. Test multi-GPU kiểm tra orchestration bằng mock, không chạy CUDA/subprocess/model thật. Hiện không có dedicated test cho `chunk_fixed_size.py`, không có end-to-end test chunk → BM25/FAISS → search → evaluate, và chưa chạy model reranker thật trong lần audit local này. Artifact index/run chứng minh pipeline đã từng được chạy ở môi trường khác hoặc được chép vào workspace, không thay thế smoke test tái lập trên T4×2.

Model card/model tree của `Vi-Qwen2-1.5B-RAG` vẫn ghi lineage từ Qwen2-7B-Instruct, nhưng revision pin thực tế có config `Qwen2ForCausalLM`, hidden size 1536, 28 layers, max positions 32768 và đúng 1.543.714.304 params. Không dùng lineage/benchmark trên card làm bằng chứng kiến trúc. Checkpoint được fine-tune cho RAG, nhưng hiệu quả làm HyDE generator vẫn phải ablation trên DSC.

### Fixed-size preprocessing cho Kaggle

Ngày 2026-08-20 đã thêm `retrieval/src/legal_ir/chunk_fixed_size.py` và console entrypoint `legal-ir-chunk-fixed-size`. Đây là baseline preprocessing riêng, không phải structural chunking:

- đọc streaming `selected-contexts/context_*.json`, sort theo numeric document ID và validate filename ↔ `id`;
- normalize bằng `legal_nfc_ws_v1`: chỉ decode HTML non-breaking-space entity, Unicode NFC, line ending, rồi collapse whitespace/control/invisible characters; không lowercase, bỏ dấu hoặc sửa số/citation;
- dùng fast tokenizer của `AITeamVN/Vietnamese_Embedding_v2` tại revision đã pin;
- default 384 content tokens, overlap 64; passage được slice bằng offset trên normalized text;
- preset Harrier có max length 512 theo tokenizer riêng; không bắt buộc rechunk cho ablation công bằng, nhưng `retrieval_text` gồm title có thể bị truncate và cần đo tỷ lệ truncation;
- output khuyến nghị trên Kaggle là `/kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl` cùng `chunk_fixed_size.manifest.json`;
- output đúng contract `chunk_id`, `document_id`, `passage`, `retrieval_text`, `metadata`; chunk ID chứa normalization version/size/overlap;
- skip passage rỗng có ghi manifest, còn JSON/schema/duplicate/mismatch lỗi thì fail-fast;
- hỗ trợ tokenizer cache và `--local-files-only` cho Kaggle không Internet.

Index hiện có chứa **323.875** chunk mang ID `fixed_legal_nfc_ws_v1_384_64`, phủ đúng 8.512 document không rỗng và loại đúng 20 document rỗng. Bản `chunks.jsonl` dùng để build được lưu lại bên trong index, nhưng workspace không có standalone `chunk_fixed_size.manifest.json`; vì vậy thiếu tokenizer fingerprint, input hash và danh sách skip ở cấp chunk artifact. Source chunker compile được nhưng vẫn chưa có dedicated unit/integration test. Không suy ra rằng current checkout có thể tái tạo byte-identical artifact nếu chưa chạy lại với tokenizer revision đã pin và so hash.

### Internal split và evaluation Task 1

Ngày 2026-08-20 đã thêm hai utility chỉ dùng Python standard library:

- `retrieval/src/legal_ir/create_val_test.py` cùng entrypoint `legal-ir-create-val-test`: lấy mẫu không hoàn lại từ `IR/train.json`, mặc định seed 2026 và split 5.600/700/700; ghi `train.json`, `val.json`, `test.json`, `split_manifest.json` vào output directory mới, không sửa dữ liệu gốc.
- Split được random ở cấp nhóm câu hỏi sau NFC + `casefold` + collapse whitespace. 16 nhóm câu hỏi trùng trong train vì vậy không bị tách qua các split; manifest lưu SHA-256, phân phối số gold documents và thống kê 5 nhóm duplicate có nhãn xung đột. Baseline này chưa stratify theo số gold documents; không nhập warmup vào train mới nếu chưa kiểm tra overlap.
- `retrieval/src/legal_ir/evaluate_recall_precision.py` cùng entrypoint `legal-ir-evaluate`: tính official macro Recall và macro Precision trên toàn bộ gold queries. Query thiếu được tính như prediction rỗng, query dư được báo/không vào mẫu số, raw answer dài hơn 5 nhận 0/0, duplicate ID được báo là contract invalid.
- Gold phải có `answer` là list ID chuỗi không rỗng; `IR/public-official.json` có nhãn null nên evaluator chủ động từ chối.
- Suite hiện bao phủ split/evaluator, HyDE normalization/cache, fusion, deep diagnostics, dense adapters và top-2 mean mà không tải model/GPU. Default split được kiểm tra lại trên dữ liệu thật và cho đúng 5.600/700/700.
- `runs/version1/val.json` là bản sao **chính xác** của validation 700 query do current `split_records(..., seed=2026, val=700, test=700)` tạo ra; mọi record đều khớp `IR/train.json`. Chỉ file val được giữ trong workspace, không có `train.json`, `test.json` hay `split_manifest.json` đi kèm, nên phải tái tạo đủ bộ nếu cần provenance hoàn chỉnh.

## Index, runs và kết quả đã có

### Index Harrier

`indexes/vietlegal_harrier_06b_v1/` là artifact thật, không phải placeholder:

- `chunks.jsonl`: 323.875 dòng/chunk, 8.512 document ID duy nhất, fixed-token 384/64 với `legal_nfc_ws_v1` và title trong `retrieval_text` khi có;
- `dense.faiss`: Flat inner-product index của `mainguyen9/vietlegal-harrier-0.6b`, revision `91a0e1ebe4b63b4475bbae40658b8ca9231bea74`, max length 512 và normalized embeddings. Encoder được cấu hình FP16, nhưng code cast vector sang **float32** trước khi thêm vào FAISS; kích thước file cũng khớp 323.875 × 1.024 vector float32;
- `bm25/`: bm25s Lucene (`k1=1.2`, `b=0.75`), 323.875 rows;
- `manifest.json`: format v1, `chunk_records_sha256=1eafbe3b0d70302fa38f534e47430cc8d512886e5055e52f083665deb12013bf`;
- `indexes.zip` là bản nén có cùng manifest nhưng kèm metadata `__MACOSX`; đây chỉ là artifact vận chuyển, không phải submission.

Manifest index không lưu `hnsw_ef_search` vì đó là search-time config, nhưng có khóa model/revision/max length/dtype/normalization/index type, HNSW build params, BM25 params và hash row/content. Không load FAISS từ nguồn không tin cậy.

### Index Vietnamese Embedding dual

`indexes/vn_embedding_v2_dual_v1/` là index short-chunk hoàn chỉnh mới:

- `chunks.jsonl`: 2.050.281 short chunks, khoảng 3,1 GB; metadata giữ mapping tới long chunks;
- `dense.faiss`: 8.397.951.021 bytes, exact Flat-IP với 2.050.281 × 1.024 vector float32; encoder là `AITeamVN/Vietnamese_Embedding_v2`, revision `18b44161e041bf1d3a333ab5144b5b7b93f914d2`, max length 2.048, encode FP16 và normalized;
- `bm25/`: bm25s Lucene cùng 2.050.281 rows;
- `manifest.json`: `chunk_records_sha256=62d318a997f30bdc5a966993151b91114411dc4570620182641dbdde1f97d02c`;
- `indexes/vn_embedding_v2_dual_v1.zip` khoảng 5 GiB là bundle vận chuyển và hiện có `.DS_Store`/`__MACOSX`; không coi đó là submission ZIP.

`dual_chunks_v1.zip` ở repo root chứa thư mục `dual_chunks_v1/` với đủ `short_chunks.jsonl` (3.207.358.735 bytes), `short_to_long.jsonl` (598.433.273 bytes), `long_chunks.jsonl` (1.376.834.383 bytes) và `manifest.json`. Workspace hiện chưa giải nén thư mục này; local long-context search phải extract trước và trỏ `--dual-chunks-dir` vào thư mục `dual_chunks_v1/`. Trên Kaggle nên dùng Dataset đã giải nén thay vì tự bung hơn 5 GB vào `/kaggle/working`.

Người dùng xác nhận ngày 2026-09-09 rằng cả short index và long index đã được
build trên Kaggle. Long index chưa có artifact/path local trong workspace nên
chưa được agent kiểm tra manifest hay số row trực tiếp. Stage 1 reranker mới
không build lại hai index và không cần diagnostics: module
`legal_ir.mine_reranker_stage1` dùng mapping nhúng trong short
`chunks.jsonl`, reconstruct vector từ long `dense.faiss` để chọn positive giới
hạn trong gold document, rồi dùng pretrained reranker mine grouped data. Module
`legal_ir.train_reranker_stage1` train listwise `1 positive + N negatives` bằng
DDP và lưu checkpoint Hugging Face. Dataset bắt buộc verify `train.json` qua
`split_manifest.json` trừ khi người chạy chủ động dùng flag unsafe; mọi gold
document của query đều bị loại khỏi negatives. Hướng dẫn Kaggle và cách nạp
checkpoint nằm ở mục 9 của `retrieval/README.md`. Hai notebook thực thi là
`notebooks/kaggle_mine_reranker_stage1.ipynb` và
`notebooks/kaggle_train_reranker_stage1.ipynb`.

System Python lúc audit không cài `numpy`, `faiss`, `bm25s`, `torch`, `transformers`, `sentence-transformers` hay `PyYAML`. Vì vậy unit tests mock/standard-library và script `analysis_v1.py` chạy được, nhưng current shell chưa thể load index hoặc chạy retrieval/model thật nếu chưa tạo environment và cài `./retrieval[dev]`.

### Validation Harrier v1

`runs/val_v1_harrier/` chứa một run 700 query trên `runs/version1/val.json`. `deep_diagnostics.json.pipeline_config` xác nhận đúng preset Harrier: BM25 300, dense 200, HyDE 200, RRF `k=60` trọng số `1/1/0,5`, fusion 30, 3 evidence/document và final top 5. Run có 780 gold labels, trung bình 1,1143/query.

Kết quả upstream từ `analysis/analysis_v1.py` và `runs/val_v1_harrier/analysis_v1.json`:

| Ranking document | Macro Recall@5 | Macro Recall@30 |
| --- | ---: | ---: |
| BM25 lane, MaxP | 0,754881 | 0,907143 |
| Harrier dense lane, MaxP | 0,872381 | 0,964881 |
| HyDE dense lane, MaxP | 0,794405 | 0,917500 |
| Weighted-RRF fusion trước reranker | 0,891310 | 0,974643 |

Kết quả final top 5:

| Aggregation sau reranker | Macro Recall | Macro Precision | Trạng thái |
| --- | ---: | ---: | --- |
| MaxP mặc định | 0,882976 | 0,191143 | lưu tại `metrics.json`; submission contract hợp lệ |
| Top-2 mean evidence score | 0,896667 | 0,194000 | replay kiểm tra ngày 2026-09-03; output chỉ ghi tạm ngoài workspace |

Các kết luận đúng trong phạm vi split/run này:

- candidate pool còn headroom lớn: fusion Recall@30 là 0,974643;
- MaxP reranker cuối làm Recall thấp hơn fusion@5 khoảng 0,00833;
- top-2 mean dùng đúng pool/evidence đã log tăng Recall khoảng 0,01369 so với MaxP và vượt fusion@5 khoảng 0,00536;
- đây là **một internal validation đã dùng để phân tích**, không phải kết quả public/private và chưa thay thế xác nhận trên test 700 query cần được tái tạo từ default split rồi giữ ngoài vòng tune.

`analysis/analysis_v1.py` recompute MaxP từ `deep_diagnostics.json`, kiểm tra đủ ba lane và báo Recall/Hit/FullHit theo cutoff; lần audit 2026-09-03 tái tạo byte-identical `analysis_v1.json`. Script và artifact này bị gitignore, chưa có unit test và chưa triển khai đầy đủ oracle union/complementarity/reranker promotion–drop như roadmap `method.md`.

### Các run khác và submission

- `runs/version1/diagnostics.json` là run validation cũ với fusion pool khoảng 50; chỉ có regular diagnostics nên không suy ra đầy đủ config.
- `runs/public_test/` và `runs/vn_embedding/` chứa diagnostics/submission cho 1.000 public query. Các submission JSON hiện có đủ query ID, đúng 5 ID chuỗi duy nhất/query và mọi ID thuộc corpus, nhưng public không có label nên **không có metric chất lượng local**.
- `runs/public_test/diagnostics2508.json` và `diagnostics_25_08.json` là hai file byte-identical; hai bản public submission gốc và `25-08/` cũng byte-identical.
- Public submission hiện tại khớp `fused_candidates[:5]` trước reranker ở cả regular diagnostics cũ và bản 25-08; nó khác `results` sau reranker ở lần lượt 999/1.000 và 998/1.000 query. Vì vậy phải gọi rõ đây là **pre-reranker fusion submission**, không phải output final của pipeline hiện tại.
- `runs/vn_embedding/submission.json` không tái lập rõ từ diagnostics cùng thư mục: khác `results` ở 817/1.000 query và khác `fused_candidates[:5]` ở 992/1.000 query. Không dùng artifact này làm benchmark/comparison cho tới khi tìm được config và quy tắc tạo submission tương ứng.
- Cả ba ZIP submission hiện có đều chứa thêm `__MACOSX/._submission.json`, trái yêu cầu ZIP chỉ chứa duy nhất `submission.json`. Không nộp lại các ZIP này; phải tạo archive sạch và kiểm tra `unzip -l`.
- `runs/test.py` là script ad-hoc lấy 5 phần tử đầu của `fused_candidates` và bỏ qua reranker score; nó không validate schema/query IDs và không phải entrypoint chuẩn. Trên validation, cách này khác final reranked submission ở 700/700 query. Dùng utility trong `retrieval/` cùng evaluator thay cho script này.
- Chỉ run `val_v1_harrier` có full resolved config trong deep diagnostics. Không suy config của run khác từ tên thư mục; tin artifact/config/manifest của chính run đó.

## Thư mục nghiên cứu phương pháp

`research_method/` dùng để lưu các bản nghiên cứu, đề xuất kiến trúc, ablation plan và kết luận thực nghiệm cho LegalIR/LegalQA. Đây là nơi agent cần kiểm tra trước khi đề xuất hoặc thay đổi method, nhằm tránh lặp lại nghiên cứu đã hoàn thành.

Quy ước:

- Đặt tên tuần tự `research_v1.md`, `research_v2.md`, ...; không ghi đè bản cũ khi có thay đổi lớn về giả thuyết, pipeline hoặc kết luận.
- Mỗi bản phải ghi ngày tạo, phạm vi Task 1/Task 2, trạng thái draft hay đã được thực nghiệm, nguồn tham khảo và khác biệt so với bản trước.
- Kết quả thực nghiệm phải ghi rõ split, cách xử lý duplicate/leakage, metric và artifact/code đã dùng.
- Nghiên cứu hiện có: `research_method/research_v1.md` — draft ban đầu cho Task 1, tập trung vào dataset legal retrieval, structural chunking, hybrid BM25+dense, document aggregation, reranking và hard-negative training.
- `research_method/method.md` — research draft ngày 2026-08-26 cho pipeline Harrier hiện tại, tập trung vào stage-ceiling analysis từ regular/deep diagnostics, ablation có thể replay offline, thí nghiệm cần rerun/rebuild và thứ tự training LambdaMART → reranker → Harrier. Phần metadata mới trong worktree mô tả đúng schema hiện có nhưng vẫn là tài liệu thiết kế.
- `research_v1.md` là đề xuất nghiên cứu, chưa phải benchmark đã được xác nhận trên dữ liệu DSC.
- `method.md` cũng là đề xuất nghiên cứu và bản thân file chưa ghi kết quả thực nghiệm DSC; metric thật hiện nằm trong `runs/val_v1_harrier/`. Người dùng yêu cầu tên file này nên đây là ngoại lệ có chủ đích so với quy ước tên tuần tự.

Fixed-size metadata hiện chỉ đủ trace/reproducibility: chunk strategy/version/index, token và character span, document length/count, source file/link/name/title. `retrieval_text` có title mới trực tiếp đi vào BM25/dense/reranker; `metadata` dict chưa được dùng làm scoring feature và chưa parse `Chương/Mục/Điều/Khoản/Điểm/Phụ lục`. Roadmap ưu tiên theo thứ tự P0 empty-doc/truncation/reproducibility → P1 diagnostics/offline replay → P2 reranker/evidence/HyDE → P3 structural/lexical → P4 LambdaMART/reranker/Harrier training → P5 synthetic/late interaction.

## Hướng dẫn làm việc cho agent tiếp theo

1. Đọc tệp này, sau đó đọc hai overview nếu cần chi tiết quy tắc chính thức.
2. Với công việc về method, đọc các bản liên quan trong `research_method/` trước; tạo version mới thay vì ghi đè lịch sử nghiên cứu.
3. Giữ nguyên dữ liệu gốc trong `IR/`, `QA/`, `selected-contexts/`; không xóa/ghi đè index và run hiện có. Đặt artifact mới vào thư mục versioned rõ ràng.
4. Luôn mở JSON với UTF-8 và xử lý `answer: null` ở public như nhãn không công khai, không phải record lỗi.
5. Khi nối nhãn IR với corpus, chuẩn hóa bằng `str(document["id"])`; không giả định có `name` hay passage không rỗng.
6. Với corpus dài, ưu tiên indexing/chunking có kiểm soát và lưu mapping chunk → document ID để output IR luôn là ID của **văn bản gốc**.
7. Trước khi tạo submission IR, kiểm tra đủ query ID đích, mỗi `answer` là mảng ID chuỗi, không trùng lặp, và có tối đa 5 phần tử. Tạo ZIP sạch chứa đúng một file tên `submission.json`; không dùng lại các ZIP hiện có có `__MACOSX`.
8. Luôn truyền `--config`, lưu resolved config và đối chiếu `manifest.json`; Python defaults không hoàn toàn trùng `default.yaml`.
9. Khi dùng diagnostics, giữ regular và deep diagnostics từ cùng một command/run. Dùng regular diagnostics để replay reranker aggregation; dùng deep diagnostics để replay lane/fusion và join text qua đúng index `chunks.jsonl`.
10. Chọn cấu hình theo luật lexicographic của cuộc thi: macro Recall trước, chỉ khi bằng nhau mới xét macro Precision; sau đó mới cân nhắc latency/VRAM. Không dùng public leaderboard như validation loop.
11. Ưu tiên sửa 20 empty documents và đo truncation/reproducibility trước khi thay model lớn hoặc training. Nếu training, group duplicate questions và không gán mọi chunk trong gold document là positive.


## Lưu ý

- Mục tiêu chung là tổng model dưới 4B params và không dùng API trả phí. Stack AITeamVN dense + reranker + Vi-Qwen2-1.5B-RAG là 2,679B; stack Harrier + cùng reranker/HyDE là 2,708B; cả hai đáp ứng giới hạn.
- Các model card, benchmark và số liệu dataset ngoài DSC trong `research_method/` chỉ là prior chọn thí nghiệm. Chỉ coi số trong `runs/.../metrics.json` hoặc báo cáo tái tạo từ đúng gold/config/artifact là kết quả của workspace này.
