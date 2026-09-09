# Khung retrieval cho DSC 2026 LegalIR

Thư mục này chứa fixed-size baseline, dual-granularity preprocessing và retrieval pipeline. Dual chunker dùng biên pháp lý/danh sách để tạo short retrieval chunks cùng long reranker chunks; parser hierarchy đầy đủ vẫn để dành cho giai đoạn sau. Code không sửa dữ liệu gốc trong `selected-contexts/`.

Pipeline mặc định (legacy, vẫn là mode mặc định):

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
- Trong flow legacy có thể tắt riêng HyDE/reranker để chạy ablation; mode
  long-context luôn cần reranker.

Ngoài flow legacy, pipeline có mode dual long-context opt-in: BM25/dense/HyDE
vẫn tìm trên **short chunks** trong index, sau đó union short candidate, map qua
long chunks thật và dùng pretrained reranker trên long context. Mode này không
thay schema/index của flow cũ; chỉ được bật bởi một config `long_context` riêng
và một thư mục dual artifacts được truyền ở runtime.

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

Reranker có đường multi-GPU riêng. Khi `reranker.multi_gpu: true`, `device: auto`
hoặc `cuda` và có ít nhất hai CUDA device, mỗi GPU giữ một model replica;
với **mỗi query**, danh sách long passages được chia thành các contiguous shard
cho các worker rồi score được ghép lại đúng thứ tự input. Đây là data parallel
theo passage, không phải gộp VRAM và không chia các query độc lập cho từng GPU.
Nếu chỉ có một GPU, code fallback về scorer single-device. Timeout worker dùng
`reranker.multi_gpu_stall_timeout_seconds`; scratch có thể đặt bằng
`LEGAL_IR_RERANKER_MULTI_GPU_TMPDIR`.

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

### Dual-granularity: short retrieval + long reranker

Để thử method short/long mà vẫn giữ baseline fixed-token, dùng entry point riêng. Nó đọc mỗi source document đúng một lần, giữ các line boundary không rỗng, tạo short chunk tối đa 450 ký tự không overlap và đồng thời pack chúng thành long chunk tối đa 2.000 ký tự, overlap bằng một short chunk hoàn chỉnh:

```bash
python -m legal_ir.chunk_dual_granularity \
  --input-dir /kaggle/input/<dataset-slug>/selected-contexts \
  --output-dir /kaggle/working/artifacts/chunks/dual_v1 \
  --short-max-characters 450 \
  --long-max-characters 2000 \
  --long-overlap-short-chunks 1
```

Lệnh này chỉ dùng CPU và standard library; không tải tokenizer/model và không cần GPU. Nó stream theo từng document thay vì giữ toàn corpus trong RAM. Output gồm:

```text
dual_v1/
├── short_chunks.jsonl     # input duy nhất cho BM25 + dense build-index
├── long_chunks.jsonl      # context cho reranker, không encode/index
├── short_to_long.jsonl    # mapping explicit, một record/short chunk
└── manifest.json          # hash, config, thống kê và skipped IDs
```

Short chunk ưu tiên biên `Điều`/`Khoản`/mục đánh số ở đầu source line; clause dài tiếp tục được tách ở dấu kết câu, line rồi word boundary. Short chunks không overlap. Long chunks chỉ ghép các short chunk hoàn chỉnh, vì vậy mọi short chunk luôn nằm trọn trong ít nhất một long chunk. Mapping chứa `primary_long_chunk_id` và toàn bộ `long_chunk_ids`; metadata hai file dùng chung half-open offset trên normalized document. Title có thể nằm trong `retrieval_text` nhưng giới hạn 450/2.000 chỉ áp dụng cho `passage`.

Không ghi đè baseline: dùng thư mục version mới như `dual_v1`. `manifest.json` được publish cuối như commit marker; khi copy artifact giữa Kaggle Dataset/notebook phải copy cả bốn file.

Index được build từ `short_chunks.jsonl` và bản copy row-stable của file này trở
thành `INDEX_DIR/chunks.jsonl`. `long_chunks.jsonl` không được encode hay thêm
vào BM25/FAISS. Khi search long-context, giữ index short riêng và mount cả thư
mục `dual_v1` để CLI đọc mapping/long text qua `--dual-chunks-dir`.

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

`Vietnamese_Embedding_v2` với dual-granularity trên Kaggle T4×2:

```bash
python -m legal_ir.cli build-index \
  --chunks /kaggle/working/artifacts/chunks/dual_v1/short_chunks.jsonl \
  --index-dir /kaggle/working/artifacts/indexes/vietnamese_embedding_v2_dual_v1 \
  --config retrieval/configs/vietnamese_embedding_dual.yaml
```

`vietnamese_embedding_dual.yaml` pin `AITeamVN/Vietnamese_Embedding_v2` tại revision `18b44161e041bf1d3a333ab5144b5b7b93f914d2`, `max_length: 2048`, vector normalized 1.024 chiều, explicit FP16, batch 16 mỗi GPU và `dense.multi_gpu: true`. Khi notebook thấy hai CUDA device, build dense tự chia short corpus thành hai contiguous shard, mỗi subprocess chỉ nhìn thấy một T4, rồi ghép vector về đúng thứ tự row trước khi ghi FAISS. BM25 vẫn build trên CPU. Chỉ `short_chunks.jsonl` được load và encode; không truyền `long_chunks.jsonl` vào `build-index`.

Trước khi chạy full build, nên smoke-test 100–1.000 short chunks trong một output/index directory tạm và kiểm tra log có dòng `Encoding ... with 2 isolated GPU workers ['cuda:0', 'cuda:1']`. Nếu chỉ thấy một GPU, code tự fallback single-device thay vì giả lập multi-GPU. Không thay `CUDA_VISIBLE_DEVICES` giữa lúc parent đang chạy.

Cảnh báo capacity: audit chỉ đọc trên 8.532 source document hiện tại với rule mặc định dự kiến tạo 2.050.281 short chunks và 205.407 long chunks. Riêng ma trận dense 1.024 chiều float32 đã khoảng 7,82 GiB, chưa gồm FAISS, BM25, `ChunkStore`, worker shard và model. Multi-GPU làm nhanh bước encode nhưng không giảm host RAM hoặc dung lượng scratch. `build-index` hiện vẫn ghép toàn bộ worker shard trước khi `index.add`, nên chưa nên coi full dual build là chắc chắn vừa Kaggle 29 GiB; bước engineering kế tiếp là add vector shard vào FAISS theo block và giải phóng BM25 trước giai đoạn dense, hoặc giảm candidate corpus bằng một ablation chunking có chủ đích.

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

### Dual short retrieval → long-context pretrained reranker

Preset
`retrieval/configs/vietnamese_embedding_dual_long_rerank.yaml` khóa ablation
hiện tại ở BM25 top 50 short chunks, dense top 100 short chunks, HyDE tắt,
pretrained `AITeamVN/Vietnamese_Reranker`, multi-GPU bật và tối đa 5 document.
HyDE vẫn là lane tùy chọn: bật `hyde.enabled` nếu muốn thêm lane dense(HyDE) vào
short candidate union; reranker luôn nhận câu hỏi gốc.

Trong mode này, `fusion.candidate_documents` và
`fusion.evidence_chunks_per_document` không tham gia scoring; chúng chỉ được giữ
để schema config tương thích flow legacy. Long pre-ranking chỉ dùng
`fusion.rrf_k` và `fusion.channel_weights`.

Luồng chính xác:

```text
BM25@50 short ───────┐
Dense@100 short ─────┼─ union + dedup short_chunk_id
HyDE short (optional)┘              ↓ map tất cả long_chunk_ids
                           dedup long_chunk_id
                                  ↓ pre-rank bằng weighted short-rank support
                        full hoặc cutoff top-C long candidates
                                  ↓ score mọi (query, long_chunk)
                         reranker top 20 long chunks
                                  ↓ MaxP long chunk → document
                            tối đa 5 document IDs
```

`long_context.candidate_mode` điều khiển cutoff **sau short→long mapping và long
ID dedup, nhưng trước reranker**:

```yaml
# Không cắt: số pair/query bằng toàn bộ unique mapped long chunks.
long_context:
  enabled: true
  candidate_mode: full
  candidate_top_k: null
  rerank_top_k_chunks: 20
  document_aggregation: maxp
  diagnostics_store_all_candidates: true
```

Để ablate một ngân sách cố định, đổi đồng thời:

```yaml
long_context:
  enabled: true
  candidate_mode: cutoff
  candidate_top_k: 75
  rerank_top_k_chunks: 20
  document_aggregation: maxp
  diagnostics_store_all_candidates: true
```

`rerank_top_k_chunks: 20` không có nghĩa model chỉ score 20 pair. Model score
toàn bộ long candidates đã chọn; pipeline mới lấy top 20 theo reranker score để
MaxP về document rồi trả tối đa 5 document. Với mode `cutoff`,
`candidate_top_k` phải là số nguyên dương và không nhỏ hơn
`rerank_top_k_chunks`.

Chạy local từ repo root sau `pip install -e ./retrieval`:

```bash
legal-ir search \
  --queries runs/version1/val.json \
  --index-dir indexes/vn_embedding_v2_dual_v1 \
  --dual-chunks-dir /path/to/extracted/dual_chunks_v1 \
  --config retrieval/configs/vietnamese_embedding_dual_long_rerank.yaml \
  --output runs/dual_long_pretrained/submission.json \
  --diagnostics runs/dual_long_pretrained/diagnostics.json \
  --deep-diagnostics runs/dual_long_pretrained/deep_diagnostics.json
```

Ví dụ Kaggle sau khi clone repo và attach các Dataset chứa index, dual chunks
và validation split:

```bash
cd /kaggle/working/DSC-Legal-IR-QA
python -m pip install -q ./retrieval
python -m legal_ir.cli --verbose search \
  --queries /kaggle/input/<split-dataset>/seed_2026/val.json \
  --index-dir /kaggle/input/<index-dataset>/index \
  --dual-chunks-dir /kaggle/input/<dual-chunk-dataset>/dual_v1 \
  --config retrieval/configs/vietnamese_embedding_dual_long_rerank.yaml \
  --output /kaggle/working/runs/dual_long_pretrained/submission.json \
  --diagnostics /kaggle/working/runs/dual_long_pretrained/diagnostics.json \
  --deep-diagnostics /kaggle/working/runs/dual_long_pretrained/deep_diagnostics.json
```

`--dual-chunks-dir` phải trỏ tới đúng bộ dual artifacts đã sinh index short.
Runtime yêu cầu `manifest.json`, `long_chunks.jsonl` và
`short_to_long.jsonl`; nên giữ thêm `short_chunks.jsonl` trong Dataset để bộ
artifact đầy đủ và audit được provenance. Pipeline dùng mapping đã nhúng trong
`INDEX_DIR/chunks.jsonl` để không nạp thêm một dictionary khoảng hai triệu dòng;
file mapping độc lập vẫn là phần bắt buộc của portable artifact và dùng cho các
phân tích offline. Khi khởi động, CLI stream-compare toàn bộ mapping độc lập với
metadata trong index và kiểm tra SHA-256 của mapping/long chunks theo manifest.
Riêng mode này, loader cũng bỏ các metadata/text không dùng sau khi đã parse:
short store chỉ giữ ID, `index_text` và mapping; long store chỉ giữ ID,
`index_text` cùng granularity. Cách nạp compact không đổi hash/index/ranking và
giảm đáng kể host RAM so với giữ nguyên toàn bộ JSON object.

`submission.json` có đúng dạng:

```json
{
  "86666": {"answer": ["280282", "..."]}
}
```

Trong flow legacy, `diagnostics.json` giữ các document đã qua fusion top-K: thứ tự trước rerank, fusion score, reranker score, score/rank của channel nếu document còn trong pool fusion, evidence chunk ID và hypothetical document.

Trong long-context mode, mỗi query còn có `long_context`: các count từ short
union, số lần mapping trước dedup, số unique long trước/sau cutoff, toàn bộ long
candidates đã score cùng retrieval support/rank, reranker score/rank và
provenance short/lane. File cũng giữ riêng reranker top-20 và document ranking
sau MaxP. Preset giữ `diagnostics_store_all_candidates: true`; không nên tắt nó
trong ablation này: khi tắt, file vẫn giữ top-20/document results nhưng bỏ danh
sách đầy đủ trước top-20, nên không replay được toàn bộ promotion/drop.
Config này điều khiển mức chi tiết; vẫn phải truyền `--diagnostics PATH` thì CLI
mới ghi `diagnostics.json`.

`deep_diag.json` là trace retrieval-lane riêng: nó nằm trước document fusion của
legacy và trước short→long mapping/cutoff của mode dual. Với mỗi query, file lưu
toàn bộ kết quả canonical của `bm25`, `dense` và `hyde` ở hai cấp:

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

### Replay top-2 mean từ diagnostics

Utility này áp dụng cho diagnostics của **flow legacy**. Nếu muốn thử ablation
đổi MaxP sau reranker sang top-2 mean mà không chạy lại model, dùng
`diagnostics.json` legacy đã có:

```bash
python -m legal_ir.diagnostics_top2_mean_submission \
  --diagnostics artifacts/runs/v1/diagnostics.json \
  --output artifacts/runs/v1/submission_top2_mean.json
```

Script đọc `fused_candidates`, tính lại document score bằng trung bình hai điểm cao nhất trong `evidence_rerank_scores`, rồi ghi output theo schema chính thức `{"query_id": {"answer": [...]}}` với tối đa 5 document/query. Nếu một document chỉ có một evidence score thì dùng chính score đó.

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

Hai flag runtime dưới đây dành cho flow legacy và không yêu cầu rebuild index.
Không truyền `--disable-reranker` cùng preset long-context, vì mode đó bắt buộc
có reranker:

```bash
# BM25 + dense, không HyDE và không reranker
legal-ir search ... --disable-hyde --disable-reranker

# BM25 + dense + Vietnamese_Reranker
legal-ir search ... --disable-hyde

# BM25 + dense + HyDE, không reranker
legal-ir search ... --disable-reranker
```

Với legacy, đo candidate recall tại `fusion.candidate_documents`. Với dual,
đo riêng recall của short union và long pool tại `long_context.candidate_top_k`
(hoặc full), rồi đo Recall@5/Precision@5 cuối, latency và peak VRAM. Các giá trị
top-k/trọng số trong YAML vẫn cần được xác nhận trên validation sạch.

## 9. Stage 1 fine-tune reranker từ short index và long index

Stage 1 gồm đúng hai job tách biệt: mine grouped dataset bằng pretrained
reranker `R0`, sau đó train `R1` bằng grouped listwise cross-entropy. Không dùng
validation/test để mine. Input `--gold` phải là `seed_2026/train.json`; mặc định
script bắt buộc có `split_manifest.json` để kiểm tra filename, số query và
SHA-256 của train split.

Hai notebook Kaggle chạy trọn quy trình nằm tại
`notebooks/kaggle_mine_reranker_stage1.ipynb` và
`notebooks/kaggle_train_reranker_stage1.ipynb`. Mỗi notebook có cell khai báo
đường dẫn Dataset riêng, validation artifact, smoke run và full run. Notebook
mining còn có cell tùy chọn tạo lại split 5.600/700/700 trực tiếp từ
`IR/train.json`; mặc định tái sử dụng split đầy đủ đã tồn tại và không ghi đè.

Miner dùng index có sẵn, không build lại index:

- short BM25 top 50 + short dense top 100, union và map sang toàn bộ unique long
  chunks theo config production;
- với mỗi gold document, long dense chọn top 8 chunk trong chính document đó,
  rồi pretrained reranker chọn một representative positive;
- candidate negative là union của mapped production pool và global long-dense
  top 60; loại mọi gold document, MaxP còn một chunk/document;
- mỗi group có một positive và bảy negative document khác nhau, ưu tiên
  violating, near-margin, long-dense và mapped hard negative.

Kaggle notebook cell để mine:

```python
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path("/kaggle/working/DSC-Legal-IR-QA")
TRAIN_SPLIT = Path("/kaggle/working/artifacts/splits/seed_2026/train.json")
SPLIT_MANIFEST = TRAIN_SPLIT.parent / "split_manifest.json"
SHORT_INDEX_DIR = Path("/kaggle/input/<short-index-dataset>/<short-index-folder>")
LONG_INDEX_DIR = Path("/kaggle/input/<long-index-dataset>/<long-index-folder>")
CONFIG = REPO_ROOT / "retrieval/configs/vietnamese_embedding_dual_long_rerank.yaml"
STAGE1_DATA_DIR = Path("/kaggle/working/artifacts/reranker/stage1_data")

subprocess.run(
    [
        sys.executable,
        "-m",
        "legal_ir.mine_reranker_stage1",
        "--gold", str(TRAIN_SPLIT),
        "--split-manifest", str(SPLIT_MANIFEST),
        "--short-index-dir", str(SHORT_INDEX_DIR),
        "--long-index-dir", str(LONG_INDEX_DIR),
        "--config", str(CONFIG),
        "--output-dir", str(STAGE1_DATA_DIR),
        "--positive-dense-top-k", "8",
        "--direct-long-top-k", "60",
        "--negatives-per-positive", "7",
    ],
    check=True,
)
```

Nên smoke test trước bằng `--max-queries 10` và một output directory riêng.
Miner reconstruct long vectors ở RAM để tìm top chunk giới hạn trong từng gold
document; với long index hiện tại khoảng 205 nghìn vector × 1.024 chiều, bản sao
float32 cần khoảng 0,78 GiB host RAM. Dataset được stream qua file tạm rồi mới
atomic replace thành `stage1_train.jsonl`; `stage1_manifest.json` khóa hash của
dataset, split, config và hai index manifest.

Train trên Kaggle T4×2 bằng DDP (`torchrun` semantics):

```python
CHECKPOINT_ROOT = Path("/kaggle/working/artifacts/reranker/stage1_model")

subprocess.run(
    [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "legal_ir.train_reranker_stage1",
        "--train-data", str(STAGE1_DATA_DIR / "stage1_train.jsonl"),
        "--data-manifest", str(STAGE1_DATA_DIR / "stage1_manifest.json"),
        "--config", str(CONFIG),
        "--output-dir", str(CHECKPOINT_ROOT),
        "--epochs", "1",
        "--per-device-group-batch-size", "1",
        "--gradient-accumulation-steps", "8",
        "--learning-rate", "2e-5",
        "--mixed-precision", "fp16",
        "--gradient-checkpointing",
    ],
    check=True,
)
```

Đây là data parallel: mỗi T4 giữ một model replica và nhận các groups khác nhau.
`per-device-group-batch-size=1` tương ứng 8 query–passage pairs/GPU vì group có
1 positive + 7 negatives. Nếu OOM, giảm số negatives khi mine (tạo ablation mới)
hoặc giảm `max_length`; không âm thầm cắt một group đã mine khi train.

Sau mỗi epoch, checkpoint Hugging Face nằm tại
`stage1_model/checkpoint-step-<N>/`; mặc định chỉ giữ checkpoint mới nhất và
không lưu optimizer state để tiết kiệm disk. `LAST_CHECKPOINT.txt` chứa tên thư
mục cần dùng. Kiểm tra artifact:

```python
checkpoint_name = (CHECKPOINT_ROOT / "LAST_CHECKPOINT.txt").read_text().strip()
checkpoint = CHECKPOINT_ROOT / checkpoint_name
assert (checkpoint / "config.json").is_file()
assert (checkpoint / "stage1_training_manifest.json").is_file()
assert any(checkpoint.glob("*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file()
print("Stage 1 checkpoint:", checkpoint)
```

Để inference/ablation với `R1`, copy config sang file mới, đặt
`reranker.model_name` bằng absolute checkpoint path và
`reranker.revision: null`. Giữ preset pretrained cũ nguyên vẹn để so sánh công
bằng R0 với R1 trên cùng validation. Stage 2 là một job tiếp theo: dùng R1 này
re-mine negative rồi train R2; không tái sử dụng nguyên negative Stage 1.

## 10. Test logic không cần tải model

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
