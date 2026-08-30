# Method — Roadmap cải tiến pipeline LegalIR và kế hoạch ablation theo diagnostics

> Ngày tạo: 2026-08-26  
> Phạm vi: DSC 2026 Task 1 — Vietnamese Legal Information Retrieval  
> Trạng thái: research draft; chưa phải kết luận thực nghiệm  
> Pipeline đích: preset `vietlegal_harrier.yaml`, tổng model khoảng 2,708B tham số, dưới giới hạn 4B  
> Quan hệ với bản trước: [`research_v1.md`](research_v1.md) nghiên cứu kiến trúc/chunking tổng quát; file này chuyển các giả thuyết đó thành thứ tự thử nghiệm và ablation có thể thực hiện sau lần chạy `diagnostics` tiếp theo.

## 1. Kết luận điều hành

Không nên bắt đầu bằng cách thay model lớn hơn hoặc fine-tune ngay. Với pipeline hiện tại, thứ tự có tỷ lệ lợi ích/chi phí tốt nhất là:

1. Sửa các lỗi làm giảm trần Recall dù ranking có tốt đến đâu: document rỗng bị bỏ index, truncation không được đo, và artifact không đủ thông tin tái lập.
2. Dùng `diagnostics.json` và `deep_diag.json` để xác định gold mất ở retrieval, fusion hay reranker.
3. Replay offline các lane, top-K, RRF, document aggregation và final cutoff; không tốn thêm GPU.
4. Chạy lại reranker cho 3–5 cấu hình fusion/evidence tốt nhất; không cần build lại dense index.
5. Chỉ rechunk/build-index khi diagnostics chứng minh upstream retrieval là bottleneck. Ưu tiên structural chunking, title/hierarchy và lexical lane tiếng Việt.
6. Khi cần training, ưu tiên LambdaMART ở document level, sau đó fine-tune reranker, cuối cùng mới LoRA/fine-tune Harrier.
7. ColBERT/late interaction, learned sparse và synthetic-query training là nhánh sau cùng vì tăng độ phức tạp index và rủi ro dữ liệu.

Điểm tối quan trọng của bài thi: **macro Recall là metric chính, macro Precision chỉ tie-break**. Với cùng một ranking, trả top 5 không thể có Recall thấp hơn trả top 1–4. Do đó top 5 vẫn là mặc định; chỉ dùng threshold/adaptive cutoff khi Recall bằng hệt trên validation và được xác nhận trên split chưa dùng để tune. Không tối ưu F1/F2 thay cho luật chấm chính thức.

## 2. Baseline phải được coi là đối chứng

Theo [`vietlegal_harrier.yaml`](../retrieval/configs/vietlegal_harrier.yaml), pipeline hiện tại là:

```text
BM25(query) top 300 chunks ────────────────┐
Harrier encode_query(query) top 200 ───────┼─ MaxP chunk → document
Vi-Qwen2-1.5B-RAG → HyDE ──────────────────┘
              Harrier encode_document(HyDE) top 200
                              ↓
weighted document RRF(k=60; BM25=1, dense=1, HyDE=0.5)
                              ↓ top 30 documents
chunk-level RRF chọn 3 evidence chunks/document
                              ↓
Vietnamese_Reranker(query gốc, từng evidence chunk thật)
                              ↓ MaxP evidence score/document
                         top 5 document IDs
```

Các chi tiết cần giữ cố định khi so sánh:

- Dense model: `mainguyen9/vietlegal-harrier-0.6b`, vector 1024 chiều, normalize, FAISS Flat inner product.
- Harrier nhận query qua `encode_query()` và corpus/HyDE qua `encode_document()`; không thêm query instruction vào passage.
- HyDE là một dense-only lane. BM25 không chạy trên text do LLM sinh.
- Raw BM25 score, cosine dense và cosine HyDE không cùng thang đo; baseline hợp nhất bằng rank, không cộng trực tiếp.
- Reranker trả raw one-logit score với identity activation, không phải xác suất đã calibration.
- FAISS Flat là exact nearest-neighbor search. Đổi sang HNSW là thí nghiệm latency/bộ nhớ, không phải đòn bẩy tăng chất lượng.

Lần chạy cũ có thể dùng config khác preset hiện tại. Khi phân tích, phải tin `deep_diag.json.pipeline_config` và index manifest của chính run đó, không suy ra config từ tên folder.

## 3. P0 — sửa trần chất lượng và khả năng tái lập

### 3.1. Document rỗng

Repo có 20 document với `passage` rỗng. Sáu document rỗng khác nhau xuất hiện trong gold của 11 query train. Fixed chunker hiện skip passage rỗng, vì vậy các gold này không thể được retrieve từ index hiện tại.

Đề xuất:

- Tạo một metadata-only fallback chunk từ `name`, URL slug và document type nếu passage rỗng.
- Gắn `metadata.empty_passage = true` để phân tích riêng, không giả vờ đây là căn cứ đầy đủ.
- Đo Recall trên nhóm 11 query trước/sau; không trộn thay đổi này với structural chunking.

### 3.2. Đo truncation bằng đúng tokenizer

Fixed chunk hiện cắt 384 content tokens bằng tokenizer của model AITeamVN, sau đó còn prepend title. Harrier có giới hạn 512 tokens và tokenizer khác. Việc quan sát input dài khoảng 480 tokens không chứng minh toàn bộ văn bản gốc chỉ dài 480 tokens; cần đo trên `retrieval_text` thực tế bằng tokenizer Harrier.

Mỗi manifest/run nên lưu:

- p50/p95/max token count của `passage` và `retrieval_text`;
- tỷ lệ chunk vượt `dense.max_length`;
- tỷ lệ pair `(query, evidence)` vượt `reranker.max_length`;
- số token title/hierarchy chiếm trong mỗi input;
- cờ `dense_truncated` và `reranker_truncated` trong deep diagnostics hoặc summary.

### 3.3. Reproducibility contract

Mỗi run phải đi kèm:

- source commit;
- config YAML đã resolve đầy đủ;
- model name + immutable revision;
- normalization/chunking version;
- index manifest/hash và chunk manifest/hash;
- HyDE cache namespace/prompt/generation config;
- seed, package versions, device/backend;
- thời gian từng stage và peak VRAM.

`do_sample: false` làm `temperature` và `top_p` của HyDE không có tác dụng. Chỉ coi chúng là hyperparameter khi bật sampling.

## 4. Diagnostics phải trả lời câu hỏi gì

### 4.1. Hai artifact hiện tại

`diagnostics.json` lưu:

- `results`: tối đa 5 document sau reranker;
- `fused_candidates`: pool theo thứ tự fusion trước reranker;
- `fusion_score`, `rerank_score`, channel ranks/raw scores;
- evidence chunk IDs và reranker score từng evidence;
- hypothetical document.

Trong code hiện tại, `fused_candidates` là shallow copy trước rerank nhưng cùng giữ các candidate object được gắn reranker score. Vì vậy có thể replay cách sort/aggregate reranker trên toàn bộ fusion pool hiện có, không chỉ top 5. Đây là hành vi cần có unit test để tránh vô tình thay đổi.

`deep_diag.json` lưu toàn bộ BM25/dense/HyDE trước fusion cutoff:

- query và HyDE text;
- mỗi channel có canonical `chunk_hits` đến configured top-K;
- `document_hits` sau MaxP;
- full pipeline config ở header.

Deep diagnostics không chứa passage/metadata. Muốn kiểm tra evidence text, document length và hierarchy phải join `chunk_id` với `INDEX_DIR/chunks.jsonl`.

### 4.2. Metrics theo từng stage

Với gold set \(G_q\) và top-K prediction \(P_q^K\):

\[
Recall@K = \frac{1}{N}\sum_q \frac{|G_q \cap P_q^K|}{|G_q|}
\]

Ngoài official macro Recall/Precision, cần tính:

- document Recall@`1,3,5,10,20,30,50,100` cho từng BM25, dense, HyDE;
- `Hit@K`: tỷ lệ query có ít nhất một gold;
- `FullHit@K`: tỷ lệ query lấy đủ mọi gold;
- MRR/nDCG chỉ để quan sát thứ tự, không thay official metric;
- oracle-union Recall của các channel;
- fusion Recall@`5,10,15,20,30`;
- final Recall/Precision@`1..5`;
- first-gold-rank distribution và missing-rate;
- latency p50/p95 từng stage.

Phân rã headroom:

```text
retrieval loss = 1 - Recall(oracle union ở độ sâu đã log)
fusion loss    = Recall(oracle union) - Recall(fusion@C)
reranker loss  = Recall(fusion@C) - Recall(final@5)
```

Các số này trả lời stage nào đáng đầu tư. Reranker không thể cứu gold chưa đi vào candidate pool.

### 4.3. Complementarity của các lane

Cho từng gold document, ghi pattern xuất hiện:

```text
BM25 only | dense only | HyDE only
BM25+dense | BM25+HyDE | dense+HyDE | all three | none
```

Đặc biệt đo:

- `unique_recall(channel)`: gold chỉ channel đó tìm được;
- số query HyDE thêm gold mới;
- số query gold vốn trong fusion nhưng bị HyDE đẩy qua cutoff;
- Jaccard overlap top-K giữa các channel;
- Δ first-gold-rank khi thêm từng lane.

Một lane có raw Recall khá nhưng unique contribution gần 0 có thể không đáng latency. Ngược lại, một lane độc lập có Recall thấp vẫn hữu ích nếu cứu đúng các query hai lane kia bỏ lỡ.

### 4.4. Reranker lift/harm

Tính trên từng query:

- promotion: gold có fusion rank `>5` nhưng rerank rank `<=5`;
- drop: gold có fusion rank `<=5` nhưng rerank rank `>5`;
- rank delta của từng gold;
- oracle Recall@5 nếu candidate pool được reorder hoàn hảo;
- score/margin distribution của gold và non-gold;
- evidence nào tạo MaxP và evidence còn lại có hỗ trợ hay mâu thuẫn.

Nếu drops xấp xỉ hoặc lớn hơn promotions, bottleneck nằm ở evidence selection, input context, aggregation hoặc checkpoint reranker.

### 4.5. Error slices

Không chỉ xem macro average. Tách ít nhất theo:

- một gold so với nhiều gold;
- query length quartile;
- query có `Điều/Khoản/Điểm`, số hiệu văn bản, ngày/tháng, tiền/phần trăm;
- câu hỏi trực tiếp viện dẫn luật so với câu tình huống/paraphrase;
- document chunk-count quartile;
- gold có title/hierarchy parse được hay không;
- empty document/truncated chunk;
- HyDE có bịa số hiệu/Điều không xuất hiện trong query hay không.

HyDE hallucinated citation có thể phát hiện sơ bộ bằng regex số hiệu/`Điều`/`Khoản`, sau đó so với query. Đây là diagnostic signal, không phải kết luận pháp lý.

### 4.6. Failure taxonomy cho từng sample

Mỗi missed gold phải thuộc một trong bốn lớp:

1. Không xuất hiện trong bất kỳ channel nào: data/chunk/index/encoder/query-expansion failure.
2. Có trong ít nhất một channel nhưng rơi khỏi fusion candidates: aggregation/fusion/cutoff failure.
3. Có trong fusion candidates nhưng rơi khỏi final top 5: evidence/reranker failure.
4. Có một gold trong final nhưng thiếu gold thứ hai trở đi: multi-label coverage failure.

## 5. Những ablation có thể replay offline

### 5.1. Chỉ cần `diagnostics.json`

Các phép thử sau không gọi model lại và exact trong phạm vi pool/evidence đã log:

| ID | Ablation | Grid ban đầu | Giới hạn |
| --- | --- | --- | --- |
| A01 | Bỏ reranker | lấy top 5 `fused_candidates` | chỉ đánh giá fusion hiện tại |
| A02 | Fusion candidate prefix | `C={5,10,15,20,25,30}` | không thử lớn hơn pool đã log |
| A03 | Reranker document aggregation | MaxP; mean; top-2 mean; `max+α·second`, `α={0.1,0.25,0.5}`; logsumexp | chỉ evidence hiện có |
| A04 | Reranker + fusion rank blend | `k={10,30,60}`, `λ={0,0.25,0.5,1}` | dùng rank, không cộng raw score |
| A05 | Final output K | `K={1,2,3,4,5}` | chỉ nhận K<5 nếu Recall bằng K=5 |
| A06 | Adaptive cutoff | top-score margin, largest gap, calibrated threshold + fallback | raw logit không phải probability |

Rank blend gợi ý:

\[
S(d)=\frac{1}{k+r_{rerank}(d)}+\lambda\frac{1}{k+r_{fusion}(d)}
\]

Với cùng evidence selector, giảm `evidence_chunks_per_document` từ 3 xuống 1/2 có thể replay bằng prefix evidence hiện tại. Thay selector hoặc tăng số evidence bắt buộc rerun reranker.

### 5.2. Cần thêm `deep_diag.json`

Deep diagnostics cho phép recompute candidate retrieval exact mà không encode lại:

| ID | Ablation | Grid ban đầu |
| --- | --- | --- |
| A10 | Channel subset | B; D; H; B+D; B+H; D+H; B+D+H |
| A11 | Channel depth | BM25 `{50,100,200,300}`; dense/HyDE `{25,50,100,200}` |
| A12 | RRF k | `{10,20,40,60,100}` |
| A13 | RRF weight | fix BM25=1; dense `{0.5,0.75,1,1.25,1.5}`; HyDE `{0,0.25,0.5,0.75,1}` |
| A14 | Fusion candidate C | `{5,10,15,20,30}` |
| A15 | Chunk→document aggregation | MaxP; top-2 mean; top-3 mean; `max+β·second`; logsumexp; chunk-rank RRF |
| A16 | Normalized score fusion | per-query min-max, z-score/robust percentile rồi convex combination |

Hai quy tắc kỹ thuật:

1. Khi thử channel top-K nhỏ hơn, phải truncate `chunk_hits` rồi chạy lại chunk→document aggregation; không truncate trực tiếp `document_hits`.
2. Không cộng raw BM25 score với cosine. Score fusion chỉ hợp lệ sau normalization trong từng channel/query và vẫn phải được tune trên validation.

Deep diagnostics không có reranker score cho candidate/evidence mới được một fusion grid đưa vào. Vì vậy A10–A16 cho **candidate Recall exact**, nhưng final reranked Recall chỉ là proxy. Chọn shortlist rồi chạy search thật.

### 5.3. Không chạy full Cartesian product

Thứ tự sweep tiết kiệm:

1. Đo từng channel và mọi channel subset ở default depth.
2. Chọn subset có oracle Recall tốt và complementarity hợp lý.
3. Tune depth theo từng lane.
4. Tune RRF `k`, sau đó coordinate-descent weights.
5. So RRF tốt nhất với normalized score fusion.
6. Tune candidate C và aggregation.
7. Chỉ rerun 3–5 cấu hình Pareto tốt nhất theo Recall, Precision, latency.

Nhân toàn bộ RRF weights với cùng một hằng số không đổi ranking; chỉ sweep tỷ lệ tương đối.

## 6. Ablation cần chạy search lại nhưng không build dense index

### 6.1. Fusion/evidence/reranker

| ID | Thay đổi | Grid ưu tiên | Lý do |
| --- | --- | --- | --- |
| B01 | Candidate documents | `{15,20,30,50}` | kiểm tra candidate ceiling và chi phí reranker |
| B02 | Evidence/document | `{1,2,3,5}` | MaxP hiện có thể bỏ evidence tốt hoặc tốn pair không cần thiết |
| B03 | Evidence selector | grounded-only; current grounded-reserved+fused; diverse/adjacent | phân biệt lỗi selector với lỗi model |
| B04 | Rerank context | chunk; parent Điều; chunk ± adjacent siblings; longer rerank chunk | luật có điều kiện phân tán qua nhiều Khoản |
| B05 | Document aggregation | MaxP; top-2 mean; logsumexp; learned small aggregator | MaxP chỉ dùng một passage |
| B06 | Reranker/fusion blend | best grid A04 hoặc LambdaMART | giữ lexical/dense evidence khi CE quá tự tin |
| B07 | Reranker max length | chỉ thử 512/1024/1536/2304 sau khi đo token histogram | tránh trả chi phí cho padding/context không dùng |

Không đưa toàn bộ document gốc vào cross-encoder nếu document dài. Chỉ thử full-document scoring khi histogram bằng **tokenizer reranker** xác nhận input nằm trong max length. Với corpus repo hiện tại có nhiều văn bản rất dài, parent Điều hoặc long evidence chunk là lựa chọn an toàn hơn.

### 6.2. HyDE

HyDE gốc tạo hypothetical document có thể hallucinate, rồi dựa vào dense encoder để tìm lân cận văn bản thật. Vì current Harrier đã được fine-tune trong miền luật, HyDE không mặc nhiên có lợi; cần đo unique contribution.

| ID | Ablation | Cấu hình |
| --- | --- | --- |
| H00 | HyDE off | BM25+dense |
| H01 | Current | 1 greedy hypothesis, 192 new tokens |
| H02 | Length | `{64,96,128,192}` greedy |
| H03 | Legal prompt | answer-like; provision-like; concise subject–act–condition–consequence |
| H04 | Multi-HyDE | 3 hypotheses sampled, mỗi hypothesis là một lane riêng rồi RRF |
| H05 | Embedding aggregation | average 3 hypothesis document embeddings |
| H06 | Conditional HyDE | chỉ gọi khi lexical/dense confidence hoặc agreement thấp |

Multi-HyDE cần `do_sample: true`; dùng cache namespace riêng cho từng prompt/generation config. Với Harrier bất đối xứng theo role, separate-lane RRF an toàn hơn việc trộn trực tiếp `encode_query(query)` và `encode_document(HyDE)`; cả hai cách vẫn phải ablate.

Một nhánh sau H00–H04 là generate 2–4 legal query paraphrases/subquestions, encode chúng bằng query role rồi fuse. So trực tiếp multi-query với HyDE; không tự động cộng cả hai vì latency và lane correlation tăng.

### 6.3. Query instruction của Harrier

Model card Harrier yêu cầu instruction-style queries; passage không có instruction. Pipeline role-aware hiện tại là đúng. Có thể thử current saved prompt so với một prompt legal-DSC ngắn hơn, nhưng:

- chỉ thay query-side instruction nên không cần re-encode corpus;
- bỏ instruction hoàn toàn có nguy cơ lệch distribution training;
- mọi prompt phải được log/hash và đánh giá theo unique recall, không chỉ cosine score.

## 7. Ablation bắt buộc rechunk hoặc build-index lại

### 7.1. Chunking và context pháp lý

Các hệ thống Vietnamese DRiLL 2025 cho bằng chứng trực tiếp rằng short retrieval chunks, longer reranking context, title và hierarchy đều đáng thử. Phải tách boundary ablation khỏi metadata ablation để biết improvement đến từ đâu.

| ID | Retrieval unit | Prefix | Rerank unit |
| --- | --- | --- | --- |
| C00 | fixed 384/64 hiện tại | current title | current evidence chunk |
| C01 | fixed 256/32 | current title | same chunk |
| C02 | fixed 448/64 hoặc 480/64 | current title | same chunk |
| C03 | Điều; fallback fixed | none | parent Điều |
| C04 | Khoản; fallback Điều/fixed | Điều heading | parent Điều hoặc long chunk |
| C05 | Khoản/Điểm leaves | document→Chương→Mục→Điều→Khoản path | parent Điều/adjacent siblings |
| C06 | dual granularity | short child index | long parent evidence for reranker |

Với mỗi boundary tốt nhất, ablate metadata riêng:

```text
body only
document title + body
document title + Điều heading + body
full bounded hierarchy path + body
```

Prefix phải có token budget riêng; title/hierarchy quá dài có thể đẩy phần căn cứ ra khỏi 512 tokens. Không copy chunk size theo paper mà không đo bằng tokenizer Harrier.

### 7.2. Lexical retrieval tiếng Việt

BM25 thường mạnh với số hiệu, thuật ngữ và trích dẫn pháp luật. Dense giúp paraphrase nhưng không thay thế exact lexical evidence. Các ablation:

| ID | BM25 lane | Ghi chú |
| --- | --- | --- |
| L00 | raw normalized Vietnamese hiện tại | baseline |
| L01 | word-segmented Vietnamese | giữ nguyên số hiệu, Điều/Khoản, ngày và dấu nối |
| L02 | raw BM25 + segmented BM25 | hai lane rồi RRF/LTR |
| L03 | `k1,b` grid | `k1={0.6,1.2,1.8,2.4}`, `b={0.2,0.5,0.75,1.0}` |
| L04 | title/hierarchy lexical lane | đo riêng trước khi merge body |

Không đưa text có underscore word segmentation vào Harrier trừ khi đó là một ablation dense độc lập.

### 7.3. Dense/index alternatives

- So Harrier với `AITeamVN/Vietnamese_Embedding_v2` trên cùng chunks và fusion grid; model scale không tự dự đoán retrieval quality.
- Có thể thêm late-interaction/ColBERT hoặc BGE-M3 sparse như một lane decorrelated sau khi baseline được tune. ColBERTv2 tăng expressivity nhờ token-level late interaction nhưng index lớn và triển khai phức tạp hơn single-vector.
- Không truncate vector Harrier tùy ý nếu checkpoint không được train bằng Matryoshka objective.
- Flat → HNSW chỉ làm khi cần giảm latency/RAM; so Recall với Flat exact làm reference.

## 8. Training theo thứ tự rủi ro tăng dần

### 8.1. LambdaMART trước

Đây là bước training đầu tiên hợp lý nhất vì DSC có label ở document level, đúng với unit mà LambdaMART rank. Không cần giả định chunk nào trong gold document là positive.

Mỗi row là `(query_id, candidate_document_id)`, group theo query, label `1` nếu document nằm trong tập gold IDs của query, ngược lại `0`. Candidate pool phải được tạo từ union BM25+dense+HyDE đủ sâu trên train.

Feature đề xuất:

| Nhóm | Features |
| --- | --- |
| Channel | presence mask; BM25/dense/HyDE rank; reciprocal rank; raw score; per-query normalized score |
| Chunk aggregation | best/second/mean/top-3 score; best-second gap; number of matched chunks |
| Agreement/fusion | số channel tìm thấy; RRF score/rank; pairwise rank differences |
| Reranker | max; top-2 mean; mean; std; top-second gap; best evidence rank |
| Query/text | query token length; number/date/citation flags; exact citation overlap |
| Metadata | query-title overlap; law/document type; hierarchy overlap; cross-reference count |
| Document | chunk count; token length; title length; optional smoothed train document prior |
| HyDE | query–HyDE overlap; added-citation flag; dense-vs-HyDE rank delta |

Quy tắc training:

- Split theo query group; các câu hỏi normalized trùng nhau phải ở cùng fold.
- Bắt đầu bằng logistic/linear fusion để có đối chứng nhỏ, sau đó LambdaMART/XGBoost `rank:ndcg`.
- Chọn checkpoint/hyperparameter bằng official macro Recall trước, Precision sau; nDCG chỉ là training surrogate.
- Dùng early stopping và group-aware cross-validation; không split candidate rows ngẫu nhiên.
- Feature document popularity chỉ là signal nhỏ. Không để prior lấn át long-tail và không lấy thống kê từ validation/test labels.

### 8.2. Fine-tune reranker

Training data phải phản ánh candidate distribution của hybrid pipeline, không dùng random negatives làm chính:

```text
positive document = mọi gold document của query
positive passage  = 1–3 representative chunks/parents trong gold document
negative document = high-ranked non-gold từ BM25/dense/HyDE/fusion/reranker
negative passage  = evidence mạnh nhất trong negative document
```

Không đánh dấu mọi chunk trong gold document là positive. Với văn bản dài, phần lớn chunk không trả lời query và sẽ tạo label noise. Chọn representative positives bằng ensemble/reranker hoặc dùng multiple-instance objective ở document level.

Loại toàn bộ gold documents khỏi negative pool. Audit near-duplicate và false negatives có khả năng thiếu nhãn trước khi dùng top-ranked non-gold làm hard negative. HYRR và các hệ thống DRiLL cho thấy hybrid-mined negatives phù hợp hơn negatives của một retriever duy nhất.

### 8.3. LoRA/fine-tune Harrier

Chỉ làm sau khi pseudo-positive passage đáng tin:

- giữ query instruction đúng format model;
- passage không có query instruction;
- dùng MNRL/CachedMNRL hoặc InfoNCE với in-batch + explicit hard negatives;
- mine negatives từ BM25, dense, HyDE và fusion; có reranker denoise;
- tránh để hai positives/near-duplicates của cùng query trở thành in-batch negatives;
- sau mỗi retriever checkpoint mới phải encode lại toàn corpus và rebuild dense index;
- thử một vòng mine → train → re-mine trước khi thêm nhiều vòng.

### 8.4. Synthetic queries và late interaction

Vietnamese legal retrieval research đã dùng LLM sinh query theo passage, lọc theo retrievability, rồi train bi-encoder/ColBERT với hard negatives. Đây là lựa chọn khi 7.000 query chưa đủ bao phủ các Điều/Khoản, nhưng có rủi ro query template, hallucination và distribution shift.

Ưu tiên:

1. Sinh query từ structural passage và metadata thật.
2. Loại query không thể retrieve source passage trong một depth hợp lý.
3. Dedupe theo normalized text/template.
4. Trộn synthetic với human queries theo tỷ lệ ablation.
5. Xác nhận trên human-only validation.

Late interaction/ColBERT là nhánh kiến trúc sau cùng. Nó có thể bắt exact token interactions tốt hơn vector đơn nhưng tăng footprint index; nên thêm như một lane ablation thay vì thay toàn bộ pipeline ngay.

## 9. Decision tree sau lần chạy diagnostics tiếp theo

| Quan sát | Kết luận gần nhất | Hành động ưu tiên |
| --- | --- | --- |
| Oracle union Recall thấp | gold chưa vào upstream pool | kiểm tra empty/truncation → chunking/title/BM25 segmentation → dense training |
| Từng lane khá nhưng union gần lane tốt nhất | các lane tương quan cao | bỏ/downweight lane tốn latency; thử lane decorrelated |
| Union cao, fusion@30 thấp | fusion/aggregation/cutoff sai | A10–A16; tăng C nếu cần |
| Fusion@30 cao, final@5 giảm mạnh | evidence/reranker làm mất gold | B02–B07; reranker blend/FT |
| HyDE unique recall thấp, latency cao | HyDE không đáng chi phí | H00/downweight/conditional HyDE |
| HyDE cứu paraphrase nhưng hại query có citation | HyDE nên có routing | conditional by citation/agreement |
| Miss tập trung ở document nhiều chunks | MaxP/length bias hoặc evidence selection | aggregation grid, chunk-count feature, structure-aware chunks |
| Miss tập trung ở exact number/law ID | lexical processing yếu | raw+segmented BM25, title/citation features |
| Dense mạnh ở scenario questions nhưng yếu citations | lane complementarity đúng kỳ vọng | giữ hybrid, tune weights theo rank/LTR |
| Reranker promotions > drops | reranker hữu ích | tune C/evidence, sau đó FT |
| Drops >= promotions | reranker/input context chưa phù hợp | blend fusion, parent context, hard-negative FT |

Không dùng ngưỡng “cao/thấp” tùy ý. So paired per-query deltas và bootstrap confidence interval. Nếu hai cấu hình không tách biệt rõ, chọn cấu hình đơn giản/nhanh hơn.

## 10. Protocol chọn experiment để tránh overfit

1. Dùng train folds cho sweep lớn/training; group normalized duplicate questions trong cùng fold.
2. Dùng val 700 query để xác nhận shortlist, không chọn hàng trăm grid points trực tiếp trên val.
3. Giữ internal test 700 query chưa đụng tới cho một lần xác nhận cuối.
4. Với mỗi comparison, báo macro Recall, macro Precision, paired delta và bootstrap CI trên query.
5. Chọn lexicographically: Recall cao hơn thắng; chỉ khi Recall bằng nhau mới xét Precision; sau đó xét latency/VRAM.
6. Mọi run thay đúng một nhóm giả thuyết. Ví dụ title on/off phải giữ chunk boundaries; structural vs fixed phải dùng cùng prefix khi có thể.
7. Không dùng public leaderboard như validation loop.

Các bảng nên sinh tự động:

```text
stage_funnel.csv       # stage/cutoff/Recall/Hit/FullHit/MRR/delta
channel_overlap.csv    # B/D/H presence patterns và unique contribution
fusion_grid.csv        # depths/weights/RRF-k/C/candidate Recall
rerank_grid.csv        # C/aggregation/blend/final-K/Recall/Precision/promote/drop
strata.csv             # metric theo query/gold/document slices
error_cases.json       # query/gold/ranks/failure stage/HyDE/evidence text+scores
```

## 11. Artifact cần giữ sau run tiếp theo

Bắt buộc:

- validation gold;
- `submission.json`;
- `diagnostics.json`;
- `deep_diag.json` chạy cùng một command;
- resolved config YAML;
- source commit;
- index/chunk manifests;
- run log có thời gian và lỗi/warning;
- metrics report.

Để phân tích evidence:

- `chunks.jsonl`, hoặc ít nhất một extract chứa mọi chunk được diagnostics tham chiếu;
- raw context của các gold/missed documents;
- HyDE cache tương ứng nếu cần tái lập generator output.

Không cần gửi `dense.faiss` để phân tích ranking đã log. Cần index chỉ khi muốn chạy search/reranker lại.

Checklist trước phân tích:

- query IDs của gold/submission/diagnostics/deep diagnostics khớp nhau;
- `deep_diag.pipeline_config` đúng preset dự định;
- channel trả đủ configured depth hoặc có lý do corpus nhỏ hơn;
- fusion scores giảm dần trong `fused_candidates`;
- toàn bộ candidate có finite reranker score khi reranker bật;
- `results` đúng top 5 sau reranker/tie-break;
- deep trace có đủ BM25, dense và HyDE khi HyDE bật;
- chunk IDs join được với đúng chunk store/index manifest.

## 12. Cải tiến schema diagnostics nên code tiếp

Để lần sau không phải suy đoán, nên bổ sung:

- `timings_ms`: BM25, dense query, HyDE generation, HyDE dense, fusion, reranker;
- peak GPU/CPU RAM và device/backend;
- token lengths + truncation flags cho query, HyDE, chunks, reranker pairs;
- source commit, config hash, index manifest hash, chunk manifest hash;
- fusion rank và final rank cho từng candidate;
- reranker input text hash, selected evidence reason và evidence selector score;
- first/best/second chunk score sau aggregation;
- channel return count và zero-score count;
- HyDE prompt hash/cache namespace/generation seed;
- optional compact error tag khi gold được cung cấp trong evaluation-only analysis.

Không nhúng toàn bộ passage vào `deep_diag.json` mặc định vì file sẽ phình rất lớn; join bằng stable `chunk_id` là đủ.

## 13. Thứ tự triển khai đề xuất

```text
P0  empty-doc fallback + truncation/reproducibility diagnostics
P1  offline stage funnel, channel overlap, RRF/score-fusion/aggregation grid
P2  exact reranker rerun cho 3–5 shortlist; evidence/context/HyDE ablations
P3  title/hierarchy + structural/dual-granularity chunks + BM25 segmentation
P4  LambdaMART → reranker hard-negative FT → Harrier LoRA
P5  synthetic data → learned sparse/late interaction
```

Nếu chỉ đủ thời gian cho ba việc, chọn:

1. Phân tích union/fusion/final headroom bằng deep diagnostics.
2. Rerun fusion shortlist + reranker evidence aggregation/blend.
3. Rebuild một structural/title-aware index đối chứng với fixed 384/64.

## 14. Nguồn chính

Nguồn về task/repo:

1. [DSC 2026 Task 1 overview](../DSC2026_Task1_LegalIR_Data_Overview.docx.md)
2. [Current Harrier config](../retrieval/configs/vietlegal_harrier.yaml)
3. [Current retrieval pipeline](../retrieval/src/legal_ir/pipeline.py) và [fusion implementation](../retrieval/src/legal_ir/fusion.py)

Nguồn Vietnamese/legal IR:

4. [DRILL Shared Task 2025 overview](https://aclanthology.org/2025.vlsp-1.16/)
5. [ViDRILL: multi-stage Vietnamese legal retrieval](https://aclanthology.org/2025.vlsp-1.17/)
6. [Data Augmentation and Hierarchical Chunking for DRiLL](https://aclanthology.org/2025.vlsp-1.18/)
7. [EDM Team: multi-stage retrieval and learning-to-rank](https://aclanthology.org/2025.vlsp-1.19/)
8. [Simple two-stage Vietnamese legal retrieval](https://aclanthology.org/2025.vlsp-1.20/)
9. [Vietnamese IR methods across domains, including legal](https://aclanthology.org/2026.findings-eacl.110/)
10. [Improving Vietnamese Legal Document Retrieval using Synthetic Data](https://arxiv.org/abs/2412.00657)
11. [ViLegalLM — hard-negative construction for legal retrieval](https://aclanthology.org/2026.findings-acl.1801/)
12. [VietLegal-Harrier-0.6B model card](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b) — kiến trúc/training claims do tác giả model tự báo, không phải benchmark độc lập trên DSC.

Nguồn phương pháp:

13. [HyDE: Precise Zero-Shot Dense Retrieval](https://aclanthology.org/2023.acl-long.99/)
14. [Reciprocal Rank Fusion](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
15. [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)
16. [PARADE: passage aggregation for document reranking](https://arxiv.org/abs/2008.09093)
17. [From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
18. [HYRR: Hybrid Infused Reranking](https://aclanthology.org/2024.lrec-main.748/)
19. [Sentence Transformers hard-negative mining](https://www.sbert.net/docs/package_reference/util/hard_negatives.html) và [losses](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html)
20. [ColBERTv2: late interaction retrieval](https://aclanthology.org/2022.naacl-main.272/)

Các kết quả từ dataset/paper khác chỉ là prior để chọn ablation. Chúng không được coi là improvement trên DSC cho tới khi chạy đúng split, config và official metric của repo này.
