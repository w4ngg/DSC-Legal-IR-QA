# Khung retrieval cho DSC 2026 LegalIR

Thư mục này chứa fixed-size preprocessing và retrieval pipeline. Fixed-token chunker là baseline để chạy thử; structural chunking vẫn để dành cho giai đoạn sau. Code không sửa dữ liệu gốc trong `selected-contexts/`.

Pipeline mặc định:

```text
BM25(query) ──────────────────────────┐
Vietnamese_Embedding_v2(query) ───────┼─ MaxP chunk→document
Vi-Qwen2-3B-RAG → HyDE → dense(HyDE) ─┘          ↓
                                      weighted document-level RRF
                                                   ↓ top 50 documents
                               Vietnamese_Reranker(query, real chunks)
                                                   ↓
                                           tối đa 5 document IDs
```

Các nguyên tắc đã được khóa trong implementation và unit test:

- Không cộng hoặc so sánh trực tiếp raw score của BM25, cosine và HyDE.
- Mỗi lane gom chunk về document bằng MaxP rồi mới dùng weighted RRF.
- HyDE chỉ chạy qua dense retrieval; BM25 luôn dùng câu hỏi thật.
- Reranker luôn dùng câu hỏi gốc và `retrieval_text` của chunk thật, không dùng hypothetical document.
- Trong các evidence đưa vào reranker, code giữ ít nhất một chunk do query gốc tìm được nếu document có chunk như vậy; HyDE không được chiếm toàn bộ evidence slots.
- Output được deduplicate ở cấp document và bị giới hạn tối đa 5 ID.
- Có thể tắt riêng HyDE/reranker để chạy ablation.

Model mặc định là `AITeamVN/Vietnamese_Embedding_v2` (567.754.752), `AITeamVN/Vietnamese_Reranker` (567.755.777) và `AITeamVN/Vi-Qwen2-3B-RAG` (3.085.938.688 tham số). Tổng checkpoint duy nhất là **4.221.449.217 tham số**; đây là sai số 5,54% so với giới hạn 4B đã được chấp nhận cho thí nghiệm này. Các revision đầy đủ được pin trong cả dataclass và YAML để index/query không vô tình dùng hai phiên bản weights khác nhau. Quantization chỉ giảm VRAM, không thay đổi số tham số.

`Vietnamese_Embedding_v2` tạo vector 1024 chiều bằng CLS pooling rồi normalize; FAISS vì vậy dùng inner product tương đương cosine. `Vietnamese_Reranker` là cross-encoder `XLMRobertaForSequenceClassification` với một raw logit cho mỗi cặp query–passage. Code dùng `CrossEncoder` với Identity activation; không dùng checkpoint reranker như một bi-encoder dù snippet tự sinh ở đầu trang Hugging Face có thể gây hiểu nhầm.

Giới hạn mặc định bám theo model card: dense tối đa 2048 token và reranker tối đa 2304 token cho cả cặp query–passage. `batch_size` 8/4 chỉ là điểm khởi đầu; giảm xuống 4/1–2 nếu chunk dài hoặc GPU ít VRAM.

## 1. Chuẩn dữ liệu chunk đầu vào

Pipeline nhận một file JSONL, mỗi dòng là một chunk:

```json
{
  "chunk_id": "21:article_3:clause_1:000",
  "document_id": "21",
  "passage": "Nội dung nguyên văn của chunk...",
  "retrieval_text": "Tên luật ... | Điều 3 ... | Khoản 1 ... | Nội dung nguyên văn...",
  "metadata": {
    "document_name": "...",
    "document_type": "Nghị định",
    "article": "Điều 3",
    "clause": "Khoản 1"
  }
}
```

Ý nghĩa:

- `chunk_id`: duy nhất trong toàn corpus.
- `document_id`: ID văn bản gốc, luôn serialize thành chuỗi; đây là ID được trả về cho cuộc thi.
- `passage`: nội dung thật được slice từ document sau bước normalize; không phải text do model sinh.
- `retrieval_text`: tùy chọn; nên là metadata quan trọng ghép với passage. Nếu thiếu, code dùng `passage`.
- `metadata`: được giữ lại để phân tích/debug, chưa được dùng như filter cứng trong baseline.

BM25, FAISS và `chunks.jsonl` trong index dùng chung một thứ tự row. Manifest chứa hash của ID, document mapping, nội dung index và thứ tự để ngăn việc load nhầm artifact.

## 2. Cài đặt

Từ root workspace:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "./retrieval[dev]"
```

Model sẽ được tải từ Hugging Face ở lần chạy thật đầu tiên. Cấu hình yêu cầu Python 3.10+, `transformers>=4.51`, PyTorch, `sentence-transformers`, `bm25s` và `faiss-cpu`.

Với `dtype: auto`, code dùng BF16 trên CUDA có hỗ trợ, FP16 trên CUDA còn lại/MPS và FP32 trên CPU. Có thể đặt riêng `device`/`dtype` cho dense, SLM và reranker nếu VRAM hạn chế. Ba model được lazy-load nhưng sẽ cùng tồn tại sau query đầu; riêng weights half precision đã khoảng 8,44 GB, còn FP32 CPU khoảng 16,89 GB, đều chưa tính activation/KV cache. Vì vậy phải đo peak memory trên máy chạy thật. Adapter HyDE ép `use_cache=True` vì config gốc của Vi-Qwen đặt giá trị này thành `false`.

## 3. Normalize và chunk fixed-size trên Kaggle

Baseline fixed-size nằm riêng trong `src/legal_ir/chunk_fixed_size.py` để không lẫn với structural chunking sau này. Nó thực hiện:

- Unicode NFC và chỉ decode các HTML entity biểu diễn non-breaking space;
- chuẩn hóa line ending, rồi collapse whitespace/control/invisible characters thành khoảng trắng;
- giữ nguyên chữ hoa/thường, dấu tiếng Việt, số, dấu câu, citation và thứ tự đoạn;
- dùng fast tokenizer đã pin của `Vietnamese_Embedding_v2`;
- chia cửa sổ 384 content token, overlap 64 token;
- lấy `passage` bằng character offset trên normalized text, không decode token IDs;
- prepend tên văn bản vào `retrieval_text`, nhưng không trộn metadata vào `passage`;
- bỏ qua document rỗng và ghi ID của chúng trong manifest;
- ghi từng artifact qua temporary file riêng rồi atomic replace; manifest được commit cuối và chứa SHA-256 để phát hiện output không đồng bộ sau interruption.

Bước này chỉ tải tokenizer, không tải weights embedding và không cần bật GPU Kaggle.

Trong Kaggle notebook, sau khi cài package, chạy:

```bash
python -m legal_ir.chunk_fixed_size \
  --input-dir /kaggle/input/<dataset-slug>/selected-contexts \
  --output /kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl \
  --manifest /kaggle/working/artifacts/chunks/chunk_fixed_size.manifest.json \
  --tokenizer-cache-dir /kaggle/tmp/huggingface \
  --chunk-size 384 \
  --overlap 64
```

Hoặc dùng console script tương đương:

```bash
legal-ir-chunk-fixed-size --help
```

Nếu Kaggle notebook không có Internet, mount tokenizer như một Kaggle Dataset rồi dùng:

```bash
python -m legal_ir.chunk_fixed_size \
  --input-dir /kaggle/input/<dataset-slug>/selected-contexts \
  --output /kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl \
  --tokenizer /kaggle/input/<model-slug>/Vietnamese_Embedding_v2 \
  --local-files-only
```

Lệnh tạo đúng hai artifact; thống kê nằm trong manifest, không có stats file thứ ba:

```text
chunk_fixed_size.jsonl
chunk_fixed_size.manifest.json
```

Mỗi `chunk_id` chứa normalization version, kích thước và overlap, ví dụ:

```text
21:fixed_legal_nfc_ws_v1_384_64:000000
```

`token_start/end` và `normalized_character_start/end` trong metadata đều là half-open offsets trên normalized document. Manifest mặc định là `chunk_fixed_size.manifest.json` nếu không truyền `--manifest`.

## 4. Tạo train/validation/test nội bộ

`create_val_test.py` lấy ngẫu nhiên các cặp `query_id -> record` từ `IR/train.json` mà không sửa file gốc. Mặc định tạo split 80/10/10 với seed cố định: train 5.600, validation 700 và test 700.

Các câu hỏi trùng nhau sau Unicode NFC, `casefold` và collapse whitespace được giữ trong cùng một split để tránh leakage. Record đầu ra không bị normalize hoặc thay đổi nội dung.

```bash
python -m legal_ir.create_val_test \
  --input /kaggle/input/<dataset-slug>/IR/train.json \
  --output-dir /kaggle/working/artifacts/splits/seed_2026 \
  --val-size 700 \
  --test-size 700 \
  --seed 2026
```

Console script tương đương:

```bash
legal-ir-create-val-test --help
```

Kết quả gồm:

```text
seed_2026/
├── train.json
├── val.json
├── test.json
└── split_manifest.json
```

Manifest lưu seed, SHA-256 nguồn và từng output, phân phối số gold documents, thống kê câu hỏi trùng và các invariant disjoint/union. Nếu fine-tune sau này, phải dùng `seed_2026/train.json`, không dùng lại `IR/train.json` gốc vì file gốc vẫn chứa validation và test. Dùng `--overwrite` có chủ đích nếu muốn thay toàn bộ artifact đã tồn tại.

Đây là random grouped split, chưa stratify theo số gold documents; phải kiểm tra histogram trong manifest trước khi dùng làm benchmark chính. Không tự động nhập `IR/warmup.json` vào training split vì warmup có nhiều câu trùng với train và có thể làm leak validation/test.

Hai utility split/evaluate chỉ dùng standard library. Nếu chưa muốn cài package retrieval cùng Torch/FAISS, có thể chạy trực tiếp file từ Kaggle Dataset chứa source:

```bash
python /kaggle/input/<source-slug>/retrieval/src/legal_ir/create_val_test.py \
  --input /kaggle/input/<dataset-slug>/IR/train.json \
  --output-dir /kaggle/working/artifacts/splits/seed_2026
```

## 5. Build index sau khi có chunk

Đổi từ BGE-M3 sang Vietnamese_Embedding_v2 làm thay đổi toàn bộ vector space. Phải build index mới; manifest sẽ chủ động từ chối index được tạo bởi checkpoint/config cũ. HyDE cache cũ cũng không được tái sử dụng vì namespace bao gồm model, revision và prompt.

```bash
legal-ir build-index \
  --chunks /kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl \
  --index-dir artifacts/indexes/vietnamese_embedding_v2_v1 \
  --config retrieval/configs/default.yaml
```

Kết quả:

```text
artifacts/indexes/vietnamese_embedding_v2_v1/
├── bm25/
├── chunks.jsonl
├── dense.faiss
└── manifest.json
```

Mặc định dense dùng exact `IndexFlatIP` trên embedding đã normalize, nên inner product là cosine. Có thể đổi `dense.index_type` thành `hnsw` khi số chunk khiến exact search quá chậm; cần rebuild index sau khi đổi.

Không load file FAISS từ nguồn không tin cậy. FAISS không đảm bảo kiểm tra đầy đủ artifact hỏng/độc hại khi đọc index.

## 6. Retrieval

Một câu hỏi:

```bash
legal-ir search-one \
  --index-dir artifacts/indexes/vietnamese_embedding_v2_v1 \
  --config retrieval/configs/default.yaml \
  --hyde-cache artifacts/cache/hyde_vi_qwen2_3b.jsonl \
  --query "Thời hạn cấp đăng ký xe máy là bao lâu?"
```

Toàn bộ split theo schema chính thức:

```bash
legal-ir search \
  --queries IR/warmup.json \
  --index-dir artifacts/indexes/vietnamese_embedding_v2_v1 \
  --config retrieval/configs/default.yaml \
  --hyde-cache artifacts/cache/hyde_vi_qwen2_3b.jsonl \
  --output artifacts/runs/v1/submission.json \
  --diagnostics artifacts/runs/v1/diagnostics.json
```

`submission.json` có đúng dạng:

```json
{
  "86666": {"answer": ["280282", "..."]}
}
```

File diagnostics giữ đầy đủ top candidate theo thứ tự trước rerank, fusion score, reranker score, raw score/rank trong từng channel, evidence chunk ID và hypothetical document để phân tích lỗi. Không nộp file diagnostics. Cache HyDE được fingerprint theo model revision, prompt và generation config, nên thay đổi thí nghiệm không tái dùng nhầm hypothesis cũ.

## 7. Đánh giá Recall và Precision

`evaluate_recall_precision.py` đọc trực tiếp gold có nhãn và `submission.json`, sau đó tính macro Recall/Precision theo đúng Task 1:

```bash
python -m legal_ir.evaluate_recall_precision \
  --gold /kaggle/working/artifacts/splits/seed_2026/val.json \
  --predictions /kaggle/working/artifacts/runs/v1/val_submission.json \
  --output /kaggle/working/artifacts/runs/v1/val_metrics.json \
  --per-query-output /kaggle/working/artifacts/runs/v1/val_per_query.json \
  --strict-submission
```

Hoặc:

```bash
legal-ir-evaluate --help
```

Evaluator lấy toàn bộ query trong gold làm mẫu số macro. Query thiếu prediction được tính như output rỗng; query dư không tham gia điểm nhưng được báo cáo. Một answer list dài hơn 5 nhận Recall và Precision bằng 0, không bị truncate. Document ID trùng được tính theo công thức tập hợp nhưng làm `submission_contract_valid=false`. `IR/public-official.json` có `answer: null`, nên không thể dùng làm gold.

Headline trong report là `official_macro_recall` và `official_macro_precision`; file per-query là tùy chọn để phân tích lỗi. `--strict-submission` vẫn ghi report và in summary nhưng trả exit code 1 nếu contract không hợp lệ. Bỏ flag này để chấm một output chưa hoàn chỉnh; `--strict-query-ids` là lựa chọn hẹp hơn và dừng ngay khi tập query ID thiếu hoặc dư.

Không cần cài package nếu chạy file trực tiếp:

```bash
python /kaggle/input/<source-slug>/retrieval/src/legal_ir/evaluate_recall_precision.py \
  --gold /kaggle/working/artifacts/splits/seed_2026/val.json \
  --predictions /kaggle/working/artifacts/runs/v1/val_submission.json
```

## 8. Ablation tối thiểu

Hai flag runtime không yêu cầu rebuild index:

```bash
# BM25 + dense, không HyDE và không reranker
legal-ir search ... --disable-hyde --disable-reranker

# BM25 + dense + Vietnamese_Reranker
legal-ir search ... --disable-hyde

# BM25 + dense + HyDE, không reranker
legal-ir search ... --disable-reranker
```

Nên đo ít nhất candidate Recall@50 trước reranker, Recall@5/Precision@5 cuối, latency và peak VRAM. Các giá trị `top_k_chunks`, trọng số HyDE, số candidate documents và số evidence chunks/document trong YAML là điểm khởi đầu, chưa phải hyperparameter đã được xác nhận trên DSC.

## 9. Test logic không cần tải model

```bash
PYTHONPATH=retrieval/src python -m unittest discover -s retrieval/tests -v
```

Test dùng backend giả để kiểm tra fusion, document aggregation, BM25 zero-score padding, đường đi/cache HyDE, input của reranker, evidence grounding, config, schema submission, random grouped split và metric Recall/Precision.

## Tài liệu API/model

- [bm25s](https://github.com/xhluca/bm25s)
- [Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2)
- [Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
- [Vi-Qwen2-3B-RAG](https://huggingface.co/AITeamVN/Vi-Qwen2-3B-RAG)
- [Sentence Transformers CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
- [FAISS index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

Lưu ý reproducibility: model card Vi-Qwen2-3B-RAG hiện còn nội dung sao chép từ bản 7B và mô tả lineage không khớp config 3B. Pipeline pin trực tiếp revision của checkpoint 3B; các benchmark/giấy phép downstream ngoài phạm vi nghiên cứu hoặc cuộc thi vẫn cần được xác minh với tác giả model.
