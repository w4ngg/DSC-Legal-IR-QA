# Khung retrieval cho DSC 2026 LegalIR

Thư mục này chứa fixed-size preprocessing và retrieval pipeline. Fixed-token chunker là baseline để chạy thử; structural chunking vẫn để dành cho giai đoạn sau. Code không sửa dữ liệu gốc trong `selected-contexts/`.

Pipeline mặc định:

```text
BM25(query) ──────────────────────────┐
Dense SentenceTransformer(query) ─────┼─ MaxP chunk→document
Vi-Qwen2-1.5B-RAG → HyDE → dense(HyDE) ─┘          ↓
                                      weighted document-level RRF
                                                   ↓ top-K documents
                               Vietnamese_Reranker(query, real chunks)
                                                   ↓
                                           tối đa 5 document IDs
```

Các nguyên tắc đã được khóa trong implementation và unit test:

- Không cộng hoặc so sánh trực tiếp raw score của BM25, cosine và HyDE.
- Mỗi lane gom chunk về document bằng MaxP rồi mới dùng weighted RRF.
- HyDE chỉ chạy qua dense retrieval; BM25 luôn dùng câu hỏi thật.
- Dense adapter tách đúng vai trò: câu hỏi thật dùng `encode_query()`, corpus và hypothetical passage dùng `encode_document()`. Vì vậy model có query instruction như VietLegal-Harrier được áp dụng prompt đúng chỗ mà không làm nhiễm corpus/HyDE.
- Output HyDE được chuẩn hóa trước khi vào dense search và diagnostics: Unicode NFC, xuống dòng/control/invisible characters, literal `\\n`/`\\r`/`\\t`, code fence và khoảng trắng thừa được collapse; chữ hoa/thường, dấu tiếng Việt, số, dấu câu và viện dẫn pháp luật được giữ nguyên.
- Reranker luôn dùng câu hỏi gốc và `retrieval_text` của chunk thật, không dùng hypothetical document.
- Trong các evidence đưa vào reranker, code giữ ít nhất một chunk do query gốc tìm được nếu document có chunk như vậy; HyDE không được chiếm toàn bộ evidence slots.
- Output được deduplicate ở cấp document và bị giới hạn tối đa 5 ID.
- Có thể tắt riêng HyDE/reranker để chạy ablation.

Model mặc định là `AITeamVN/Vietnamese_Embedding_v2` (567.754.752), `AITeamVN/Vietnamese_Reranker` (567.755.777) và `AITeamVN/Vi-Qwen2-1.5B-RAG` (1.543.714.304 tham số). Tổng checkpoint duy nhất là **2.679.224.833 tham số**, nằm dưới giới hạn 4B. Preset thay dense bằng `mainguyen9/vietlegal-harrier-0.6b` (596.049.920 tham số) có tổng **2.707.520.001**, cũng dưới 4B. Các revision đầy đủ được pin trong YAML để runtime/cache không vô tình dùng phiên bản weights khác. Quantization chỉ giảm VRAM, không thay đổi số tham số.

`Vietnamese_Embedding_v2` tạo vector 1024 chiều bằng CLS pooling rồi normalize. `vietlegal-harrier-0.6b` cũng tạo vector 1024 chiều đã normalize nhưng dùng Qwen3 backbone, last-token pooling và query instruction dành cho luật Việt Nam. FAISS dùng inner product, tương đương cosine khi vector đã normalize. `Vietnamese_Reranker` là cross-encoder `XLMRobertaForSequenceClassification` với một raw logit cho mỗi cặp query–passage. Code dùng `CrossEncoder` với Identity activation; không dùng checkpoint reranker như một bi-encoder dù snippet tự sinh ở đầu trang Hugging Face có thể gây hiểu nhầm.

Giới hạn mặc định bám theo model card: `Vietnamese_Embedding_v2` tối đa 2048 token, Harrier tối đa 512 token và reranker tối đa 2304 token cho cả cặp query–passage. Baseline dùng dense batch 32, preset Harrier bắt đầu ở 16; đây là kích thước **trên mỗi GPU**, không phải tổng của hai GPU. Có thể tăng dần sau khi đo VRAM hoặc giảm còn 8 nếu Harrier OOM. Reranker mặc định dùng batch 4.

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

Model sẽ được tải từ Hugging Face ở lần chạy thật đầu tiên. Cấu hình yêu cầu Python 3.10+, `transformers==4.57.6`, `sentence-transformers==5.1.2`, PyTorch, `bm25s` và `faiss-cpu`. Hai thư viện model được pin vì Harrier công bố metadata bằng Transformers 4.57.6, còn source cần API `encode_query()`/`encode_document()` của SentenceTransformers 5.1.2; không để Kaggle tự nâng lên major version mới trong một scheduled run.

Với `dtype: auto`, code dùng BF16 trên CUDA có hỗ trợ, FP16 trên CUDA còn lại/MPS và FP32 trên CPU. Preset Harrier đặt rõ `float16` để index build và query cùng precision trên T4; T4 không có native BF16. Có thể đặt riêng `device`/`dtype` cho dense, SLM và reranker nếu VRAM hạn chế. Ba model được lazy-load nhưng sẽ cùng tồn tại sau query đầu; riêng weights của stack mặc định khoảng 5,36 GB ở FP16/BF16 hoặc 10,72 GB ở FP32, đều chưa tính activation/KV cache. Vì vậy phải đo peak memory trên máy chạy thật. Adapter HyDE ép `use_cache=True` vì config gốc của Vi-Qwen đặt giá trị này thành `false`.

Khi `dense.multi_gpu: true`, `device: auto` hoặc `cuda`, và thấy từ hai CUDA device trở lên, riêng bước encode toàn corpus trong `build-index` chạy các Python worker độc lập. Parent resolve checkpoint một lần nhưng không load model; mỗi worker chỉ nhìn thấy một GPU qua `CUDA_VISIBLE_DEVICES`, tự load một model replica, encode một contiguous shard và ghi `.npy` tạm. Parent poll exit code/status, ghép shard đúng thứ tự corpus và dừng toàn bộ peer nếu một worker lỗi hoặc không báo tiến độ trong `multi_gpu_stall_timeout_seconds`. Thiết kế này không dùng shared parent model hoặc queue tensor của SentenceTransformers.

Trên Kaggle T4×2, hai T4 có 16 GB VRAM riêng, không hợp thành một GPU 32 GB; notebook chỉ có 4 CPU core và 29 GB host RAM. Worker tự giới hạn khoảng hai CPU thread, dùng một T4/model replica và ghi log riêng để tránh progress bar chồng nhau. Dense query và HyDE vẫn single-GPU vì mỗi lần chỉ encode một text. `batch_size` là số passage mỗi forward trên **mỗi GPU**; `multi_process_chunk_size` là macro-block được ghi heartbeat sau khi hoàn tất, mặc định nội bộ 256 nếu để `null`. Có thể đặt `LEGAL_IR_MULTI_GPU_TMPDIR` nếu muốn chọn scratch directory khác cho input/output shard tạm.

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

Sau khi `json.load` đọc document, escape JSON `\n` và `\r` đã trở thành ký tự xuống dòng/carriage-return thật. `normalize_legal_text()` collapse chúng cùng tab và mọi whitespace liên tiếp thành đúng một dấu cách, nên `passage` sau normalize không còn line break. Chuỗi literal `/n` hoặc `/r` (dấu gạch chéo xuôi rồi tới chữ) không phải ký tự xuống dòng và được giữ nguyên.

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

Đổi dense checkpoint làm thay đổi vector space. Phải dùng một thư mục index mới; manifest sẽ chủ động từ chối index được tạo bởi checkpoint/config khác. Đổi `batch_size`, `multi_gpu`, `multi_process_chunk_size` hoặc `multi_gpu_stall_timeout_seconds` chỉ thay runtime encode và không làm index cũ mất hiệu lực. HyDE cache **vẫn tái sử dụng được** khi chỉ đổi dense model, miễn là model/prompt/generation config và normalization policy của HyDE không đổi.

Baseline `Vietnamese_Embedding_v2`:

```bash
legal-ir build-index \
  --chunks /kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl \
  --index-dir artifacts/indexes/vietnamese_embedding_v2_v1 \
  --config retrieval/configs/default.yaml
```

VietLegal-Harrier:

```bash
legal-ir build-index \
  --chunks /kaggle/working/artifacts/chunks/chunk_fixed_size.jsonl \
  --index-dir artifacts/indexes/vietlegal_harrier_06b_v1 \
  --config retrieval/configs/vietlegal_harrier.yaml
```

Có thể sửa trực tiếp phần `dense` của `default.yaml`; các trường cần khớp Harrier là `model_name`, `revision`, `max_length: 512` và `normalize_embeddings: true`. File preset giúp tránh quên một trường và giữ baseline để đối chiếu. Không cần chunk lại để chạy ablation trên cùng corpus, nhưng Harrier sẽ truncate `retrieval_text` sau 512 token theo tokenizer của chính nó. Vì fixed chunk hiện được cắt bằng tokenizer AITeamVN rồi prepend title, nên cần theo dõi tỷ lệ truncation khi kết luận model nào tốt hơn.

Trước khi build trên Kaggle T4×2, xác nhận notebook thực sự được cấp hai GPU:

```python
import subprocess
import sys

subprocess.run(["nvidia-smi", "-L"], check=True)
subprocess.run(
    [
        sys.executable,
        "-c",
        (
            "import torch; "
            "print('CUDA:', torch.cuda.is_available()); "
            "print('GPU count:', torch.cuda.device_count()); "
            "print([torch.cuda.get_device_name(i) "
            "for i in range(torch.cuda.device_count())])"
        ),
    ],
    check=True,
)
```

Kết quả cần có hai T4 và `GPU count: 2`. Có thể gọi CLI hoặc Python API trực tiếp; parent sẽ tự mở hai module worker bằng interpreter sạch. Log đúng có dạng `Encoding ... with 2 isolated GPU workers ['cuda:0', 'cuda:1']`, sau đó là `Dense multi-GPU progress ... (gpu0=..., gpu1=...)`. Nếu worker lỗi/OOM/đứng quá timeout, lệnh trả exception kèm status, traceback và cuối worker log thay vì chờ vô hạn. `dense.faiss` được ghi qua file tạm rồi replace; worker shard cũng nằm trong temporary directory và được dọn khi hoàn tất hoặc interrupt.

Kết quả:

```text
artifacts/indexes/vietnamese_embedding_v2_v1/
├── bm25/
├── chunks.jsonl
├── dense.faiss
└── manifest.json
```

Mặc định dense dùng exact `IndexFlatIP` trên embedding đã normalize, nên inner product là cosine. Có thể đổi `dense.index_type` thành `hnsw` khi số chunk khiến exact search quá chậm; cần rebuild index sau khi đổi. Hai model đều cho vector 1024 chiều, nhưng tuyệt đối không dùng file `dense.faiss` của model này với model kia.

Không load file FAISS từ nguồn không tin cậy. FAISS không đảm bảo kiểm tra đầy đủ artifact hỏng/độc hại khi đọc index.

## 6. Retrieval

Một câu hỏi:

```bash
legal-ir search-one \
  --index-dir artifacts/indexes/vietnamese_embedding_v2_v1 \
  --config retrieval/configs/default.yaml \
  --hyde-cache artifacts/cache/hyde_vi_qwen2_1_5b.jsonl \
  --query "Thời hạn cấp đăng ký xe máy là bao lâu?"
```

Toàn bộ split theo schema chính thức:

```bash
legal-ir search \
  --queries IR/warmup.json \
  --index-dir artifacts/indexes/vietnamese_embedding_v2_v1 \
  --config retrieval/configs/default.yaml \
  --hyde-cache artifacts/cache/hyde_vi_qwen2_1_5b.jsonl \
  --output artifacts/runs/v1/submission.json \
  --diagnostics artifacts/runs/v1/diagnostics.json \
  --deep-diagnostics artifacts/runs/v1/deep_diag.json
```

`submission.json` có đúng dạng:

```json
{
  "86666": {"answer": ["280282", "..."]}
}
```

`diagnostics.json` giữ các document đã qua fusion top-K: thứ tự trước rerank, fusion score, reranker score, score/rank của channel nếu document còn trong pool fusion, evidence chunk ID và hypothetical document.

`deep_diag.json` là trace riêng trước fusion cutoff. Với mỗi query, file lưu toàn bộ kết quả canonical của `bm25`, `dense` và `hyde` ở hai cấp:

- `chunk_hits`: rank một-based, `chunk_id`, `document_id` và raw score trong chính channel đó;
- `document_hits`: rank sau MaxP, raw score lớn nhất, `best_chunk_id` và rank của chunk tạo ra score đó;
- `search_text` và `search_text_source`: câu hỏi thật cho BM25/dense, hypothetical document đã normalize cho HyDE;
- số top chunk yêu cầu và số chunk/document thực trả về.

Ví dụ rút gọn:

```json
{
  "format_version": 1,
  "pipeline_config": {"bm25": {}, "dense": {}, "hyde": {}},
  "queries": {
    "86666": {
      "query": "...",
      "hypothetical_document": "...",
      "channels": {
        "bm25": {
          "search_text_source": "query",
          "requested_top_k_chunks": 300,
          "returned_chunk_count": 300,
          "returned_document_count": 184,
          "chunk_hits": [
            {"rank": 1, "chunk_id": "...", "document_id": "280282", "score": 19.2}
          ],
          "document_hits": [
            {
              "rank": 1,
              "document_id": "280282",
              "score": 19.2,
              "best_chunk_id": "...",
              "best_chunk_rank": 1
            }
          ]
        }
      }
    }
  }
}
```

Chunk bị trùng được giữ score tốt nhất, score `NaN`/`Infinity` bị loại và tie được sắp deterministic giống hệt fusion. Không so sánh raw score giữa BM25 và dense/HyDE. File không lặp passage/metadata; dùng `chunk_id` để join với `INDEX_DIR/chunks.jsonl`. Nếu tắt HyDE thì channel `hyde` không xuất hiện.

Deep diagnostics chỉ được thu thập khi có flag, được stream theo từng query qua file tạm rồi atomic replace, nhưng file cuối có thể lớn khoảng hàng trăm MB. Trên Kaggle phải ghi nó vào `/kaggle/working`, không phải `/kaggle/input`. Không nộp `diagnostics.json` hoặc `deep_diag.json` làm submission.

HyDE dùng policy `hyde_nfc_ws_v1`: output từ Qwen, output đọc từ cache và text ngay trước dense retrieval đều được đưa về cùng dạng canonical. Ngoài line break thật, policy còn xử lý literal `\\n`, `\\r`, `\\t` mà model có thể sinh ra dưới dạng hai ký tự escape, cùng control/zero-width characters, non-breaking space và code fence. Policy không lowercase, không bỏ dấu, không sửa con số hay citation. Phiên bản policy nằm trong fingerprint cache cùng model revision, prompt và generation config; vì vậy các dòng cache cũ vẫn có thể nằm trong cùng file JSONL nhưng sẽ không được tái sử dụng sau thay đổi này.

Đổi từ Vi-Qwen2-3B-RAG sang Vi-Qwen2-1.5B-RAG làm namespace HyDE thay đổi. Cache 3B không được dùng cho kết quả mới; nên dùng tên file `hyde_vi_qwen2_1_5b.jsonl` để artifact rõ ràng. Thay HyDE model không làm thay đổi corpus vectors, do đó không cần build lại BM25 hoặc `dense.faiss`.

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

Nên đo candidate recall tại đúng cutoff `fusion.candidate_documents` trước reranker, Recall@5/Precision@5 cuối, latency và peak VRAM. Các giá trị `top_k_chunks`, trọng số HyDE, số candidate documents và số evidence chunks/document trong YAML là điểm khởi đầu, chưa phải hyperparameter đã được xác nhận trên DSC.

## 9. Test logic không cần tải model

```bash
PYTHONPATH=retrieval/src python -m unittest discover -s retrieval/tests -v
```

Test dùng backend giả để kiểm tra fusion, document aggregation, BM25 zero-score padding, chuẩn hóa/đường đi/cache HyDE, input của reranker, evidence grounding, config, schema submission, random grouped split và metric Recall/Precision.

## Tài liệu API/model

- [bm25s](https://github.com/xhluca/bm25s)
- [Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2)
- [VietLegal-Harrier 0.6B](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b)
- [Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker)
- [Vi-Qwen2-1.5B-RAG](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG)
- [Sentence Transformers multi-process/multi-GPU encoding](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html#multi-process-multi-gpu-encoding)
- [Kaggle Notebook T4×2 specifications](https://www.kaggle.com/docs/notebooks)
- [NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [Sentence Transformers CrossEncoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
- [FAISS index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

Lưu ý reproducibility: model card/model tree Vi-Qwen2-1.5B-RAG ghi lineage từ Qwen2-7B-Instruct, trong khi checkpoint pin thực tế là `Qwen2ForCausalLM`, hidden size 1536, 28 layers và 1.543.714.304 tham số. Pipeline dựa vào config/weights của revision đã pin; không dùng mô tả lineage hoặc benchmark trên card làm bằng chứng kiến trúc.
