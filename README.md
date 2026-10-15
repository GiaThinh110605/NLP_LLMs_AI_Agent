# Lộ trình & Tài liệu học NLP, LLMs & AI Agent

> **Lưu ý:** Trong quá trình học, ưu tiên tra cứu tài liệu chính thức (documentation), Google và tự triển khai mã nguồn; hạn chế lạm dụng AI để giải hộ bài tập nhằm nắm vững bản chất kỹ thuật.

### Slide bài giảng & Khởi động
- **Machine Learning & Deep Learning Slide:** https://canva.link/6xdb3n1gpijx1kw
- **FastAPI Slide:** https://canva.link/840ya322qx6ytve
- **NLP Slide:** https://canva.link/i0xl1006o17uc2g
- **Thực hành Machine Learning với Scikit-learn:** https://www.youtube.com/live/fm8nn374caw?si=Rz_JetT6dEmlPhM-

---

## Module 1: NLP Fundamentals

### 1. Nội dung học
- Tổng quan về NLP, các bài toán phổ biến và vai trò của NLP trong AI hiện đại.
- Lịch sử phát triển của NLP: từ các phương pháp truyền thống đến Machine Learning, Deep Learning và LLM.
- Các kỹ thuật xử lý văn bản cơ bản: tokenization, normalization, stopwords, stemming và lemmatization.
- Biểu diễn văn bản: Bag of Words, n-grams, TF-IDF và khái niệm word embeddings.
- Các bài toán NLP phổ biến: text classification, sentiment analysis, named entity recognition (NER), information extraction và text similarity.
- Xây dựng pipeline NLP cơ bản: thu thập dữ liệu, làm sạch văn bản, tiền xử lý, biểu diễn dữ liệu và đưa vào mô hình.
- Chia tập dữ liệu train/validation/test; hiểu data leakage và tầm quan trọng của dữ liệu chất lượng.
- Các chỉ số đánh giá cơ bản: accuracy, precision, recall và F1-score.
- Thực hành xử lý và phân tích một bộ dữ liệu văn bản; xây dựng mô hình phân loại văn bản đơn giản bằng TF-IDF và Logistic Regression.

### 2. Tài liệu học
- **Video bài giảng:** https://www.youtube.com/live/4O0HPXiDpTs?si=GOmMVvysstUO2zl1
- **NLTK Book:** https://www.nltk.org/book_1ed/
- **NLTK Documentation:** https://www.nltk.org/
- **Scikit-learn — Text Feature Extraction:** https://scikit-learn.org/stable/modules/feature_extraction.html
- **Scikit-learn — Text Classification:** https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

### 3. Bài tập thực hành
1. **Làm sạch và chuẩn hóa dữ liệu:** Xử lý dữ liệu rỗng, trùng lặp và chuẩn hóa ký tự văn bản.
2. **Tiền xử lý text:** Thực hành tokenization, loại bỏ stopwords, stemming và lemmatization.
3. **Biểu diễn đặc trưng:** Xây dựng ma trận Bag of Words và TF-IDF; so sánh kết quả biểu diễn.
4. **Mô hình hóa:** Xây dựng mô hình phân loại cảm xúc (sentiment) hoặc phân loại chủ đề bằng Logistic Regression.
5. **Đánh giá & Tránh Leakage:** Chia tập train/validation/test đúng chuẩn, đánh giá mô hình bằng accuracy, precision, recall, F1-score.
6. **Error Analysis:** Phân tích các mẫu dự đoán sai và viết báo cáo ngắn về pipeline, kết quả và hạn chế.

---

## Module 2: Deep Learning in NLP

### 1. Nội dung học
- Neural Network trong NLP và cách deep learning cải thiện khả năng biểu diễn ngôn ngữ.
- Kiến trúc tuần tự: nguyên lý hoạt động, ưu và nhược điểm của RNN, LSTM và GRU.
- Xử lý vấn đề vanishing/exploding gradient trong mạng tuần tự.
- Ứng dụng mô hình tuần tự vào bài toán phân loại văn bản (sequence classification).
- Thực hành xây dựng, huấn luyện và đánh giá mô hình NLP cơ bản bằng PyTorch.

### 2. Video & Tài liệu học
- **Video bài giảng:**
  - Sequence Models & RNN: https://www.youtube.com/live/pusntJk4tsw?si=eV0oiDSbunvAJ0Ct
  - LSTM & GRU Architecture: https://www.youtube.com/live/_J1tXiC1DLY?si=8SV6nDGYFBi97qJC
  - Playlist NLP Deep Learning: https://www.youtube.com/playlist?list=PLqL-7eLmqd9V3faivSAST76YQClS44dSz
- **Tài liệu tham khảo:**
  - PyTorch Sequence Models Tutorial: https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
  - PyTorch nn.RNN: https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html
  - PyTorch nn.LSTM: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
  - PyTorch nn.GRU: https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html
  - Dataset Stanford IMDB: https://huggingface.co/datasets/stanfordnlp/imdb

### 3. Bài tập thực hành

#### Bài 1 — Neural Network cơ bản (Embedding + Feed Forward)
- Chuyển văn bản thành token ID; xây dựng vocabulary và padding chuỗi về cùng độ dài.
- Xây dựng mô hình với kiến trúc: Embedding → Global Pooling (Average/Max) → Linear.
- Huấn luyện mô hình phân loại cảm xúc trên IMDB dataset.
- Theo dõi loss curve và đánh giá Accuracy, F1-score trên tập test.

#### Bài 2 — Phân loại văn bản với LSTM
- Thay lớp pooling bằng lớp `nn.LSTM`.
- Trích xuất hidden state của bước thời gian cuối (hoặc qua pooling các hidden states) để đưa vào classifier.
- Theo dõi loss/accuracy qua từng epoch, kiểm tra hiện tượng overfitting.
- Thử nghiệm dự đoán trên các câu đánh giá thực tế mới nhập vào.

#### Bài 3 — So sánh thực nghiệm RNN, LSTM và GRU
- Giữ nguyên cấu hình phân chia dữ liệu, vocabulary và hyperparameter huấn luyện.
- Huấn luyện độc lập 3 mô hình: Simple RNN, LSTM, GRU.
- Lập bảng so sánh: Accuracy, Macro F1-score, thời gian huấn luyện mỗi epoch, tốc độ suy luận (inference latency).
- Phân tích nguyên nhân tại sao LSTM/GRU vượt trội hơn RNN và phân loại các trường hợp mô hình đoán sai.

---

## Module 3: Transformer & LLMs

### 1. Nội dung học

**A. Nền tảng Transformer và LLM**
- Kiến trúc Transformer: Self-Attention, Multi-Head Attention, Positional Encoding, Feed-Forward layers và Residual Connections.
- So sánh sự khác biệt bản chất giữa Transformer và kiến trúc tuần tự (RNN/LSTM).
- Phân loại các nhánh kiến trúc: Encoder-only (BERT), Decoder-only (GPT, LLaMA), Encoder-Decoder (T5).
- Cơ chế huấn luyện trước (Pre-training), căn chỉnh (Alignment) và sinh văn bản (Autoregressive generation).
- Khái niệm token, context window, attention mask và giới hạn độ dài đầu vào.
- Tiêu chí lựa chọn mô hình: chất lượng, chi phí API, tài nguyên GPU và độ trễ.

**B. Prompt Engineering**
- Các kỹ thuật prompting cơ bản và nâng cao: Zero-shot, Few-shot, Instruction Prompting, Chain-of-Thought (CoT).
- Quản lý context window, tối ưu lượng token và kỹ thuật tạo prompt template có cấu trúc.
- Kiểm soát định dạng đầu ra (Structured Outputs: JSON schema, Markdown, key-value).

**C. Đánh giá mô hình và Benchmark**
- Thiết kế bộ dữ liệu kiểm thử chuẩn (Ground truth test set) và tiêu chí đánh giá định lượng.
- Các tiêu chí: độ chính xác, mức độ liên quan, tuân thủ hướng dẫn (instruction-following) và tính nhất quán.
- Đo lường và đối chiếu kết quả thực nghiệm giữa các prompt và mô hình khác nhau.

**D. Phân biệt RAG và Fine-tuning**
- Bản chất kỹ thuật, ưu nhược điểm và chi phí của RAG vs Fine-tuning:

| Tiêu chí | RAG (Retrieval-Augmented Generation) | Fine-tuning (PEFT / LoRA) |
| :--- | :--- | :--- |
| **Mục đích** | Bổ sung kiến thức động từ tài liệu ngoài | Căn chỉnh hành vi, phong cách, định dạng chuyên biệt |
| **Cập nhật dữ liệu** | Nhanh chóng, chỉ cần cập nhật kho vector/tài liệu | Phải huấn luyện lại hoặc fine-tune tiếp tục |
| **Chi phí ban đầu** | Thấp, tận dụng mô hình có sẵn | Đòi hỏi tài nguyên tính toán (GPU) và dữ liệu huấn luyện |
| **Phù hợp khi** | Hỏi đáp tài liệu, dữ liệu biến động liên tục, cần dẫn nguồn | Chuẩn hóa giọng văn, format output nghiêm ngặt, domain hẹp |

### 2. Video & Tài liệu học
- **Video bài giảng:**
  - Video 1 (Kiến trúc & Cơ chế Attention): https://www.youtube.com/live/VCq4YFYAj-Q?si=fQPAbUMXpS35tmMD
  - Video 2 (Playlist LLMs & Transformer): https://www.youtube.com/playlist?list=PLqL-7eLmqd9V3faivSAST76YQClS44dSz
- **Tài liệu chuyên sâu:**
  - Paper gốc *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
  - Hugging Face LLM Course: https://huggingface.co/learn/llm-course/
  - Hugging Face Transformers: https://huggingface.co/docs/transformers/
  - OpenAI Prompt Engineering Guide: https://developers.openai.com/api/docs/guides/prompt-engineering
  - Hugging Face PEFT (LoRA/QLoRA): https://huggingface.co/docs/peft/index
  - Hugging Face Transformers Trainer: https://huggingface.co/docs/transformers/main_classes/trainer
  - Hugging Face Evaluate: https://huggingface.co/docs/evaluate/

### 3. Bài tập thực hành

**Bài 1 – Thử nghiệm kỹ thuật Prompt Engineering**
- Chọn một bài toán thực tế (ví dụ: phân loại ticket hỗ trợ hoặc trích xuất thông tin khách hàng).
- Thiết kế 10 trường hợp kiểm thử và so sánh 3 phong cách: Zero-shot, Few-shot và Instruction Prompting kèm ràng buộc JSON.
- Đánh giá tỷ lệ tuân thủ định dạng và chất lượng câu trả lời.

**Bài 2 – Thí nghiệm Context Window & Token Budget**
- Thử nghiệm cung cấp context ở 3 mức: ngắn gọn, đầy đủ và context dài có chèn thông tin nhiễu (distractor).
- Thay đổi `max_tokens` và nhiệt độ `temperature`.
- Ghi nhận chất lượng đầu ra, thời gian sinh (latency) và chi phí token.

**Bài 3 – Xây dựng Benchmark kiểm thử LLM**
- Chuẩn bị bộ test 30–50 câu hỏi có đáp án chuẩn (ground truth).
- Xây dựng thang điểm đánh giá (accuracy, format compliance).
- Chạy đánh giá tự động và xuất báo cáo kết quả ra file CSV/JSON.

**Bài 4 – Xây dựng RAG nguyên bản (Prototype)**
- Tải một tập tài liệu PDF, thực hiện: Đọc PDF → Chunking → Tạo Embeddings → Vector search → Sinh câu trả lời với context.
- Yêu cầu mô hình bắt buộc trích dẫn nguồn (số trang, tên file).

**Bài 5 – Fine-tuning mô hình nhỏ với LoRA/PEFT**
- Lựa chọn mô hình nền nhỏ (ví dụ: Qwen2.5-0.5B / LLaMA-1B).
- Chuẩn bị tập dữ liệu huấn luyện theo định dạng instruction/output, chia train/val.
- Huấn luyện bằng thư viện PEFT/LoRA, theo dõi loss và đánh giá trước/sau fine-tuning.

**Bài 6 – Nghiên cứu so sánh RAG vs Fine-tuning**
- Áp dụng cả 2 phương pháp trên cùng một bài toán và cùng bộ câu hỏi kiểm thử.
- Lập bảng so sánh chi tiết: độ chính xác, chi phí tài nguyên, thời gian phản hồi, khả năng cập nhật tri thức mới.

### 4. Sản phẩm đầu ra
- Notebook hoàn chỉnh thực hành Prompt Engineering, Benchmark và Transformer.
- Mã nguồn pipeline RAG cơ bản có trích dẫn tài liệu.
- Script fine-tuning mô hình nhỏ với PEFT/LoRA kèm checkpoint và log đánh giá.
- Báo cáo phân tích so sánh RAG vs Fine-tuning chi tiết.

---

## Module 4: RAG (Retrieval-Augmented Generation)

### 1. Nội dung học
- Tổng quan kiến trúc RAG: Workflow chuẩn từ câu hỏi người dùng đến câu trả lời hoàn chỉnh.
- Các thành phần cốt lõi:
  - **Document Loaders & Preprocessing:** Đọc PDF, Word, Markdown, làm sạch dữ liệu.
  - **Chunking Strategies:** Fixed-size chunking, sentence splitting, semantic chunking; vai trò của chunk size và overlap.
  - **Embeddings:** Vector representation của văn bản, sự khác biệt giữa sparse embedding (BM25) và dense embedding.
  - **Vector Database:** Lưu trữ, lập chỉ mục (HNSW, IVFFlat) và tìm kiếm độ tương đồng (Cosine, Dot product, Euclidean).
  - **Retrieval & Reranking:** Top-k retrieval, lọc metadata, kết hợp Hybrid Search (BM25 + Dense).
  - **Generation:** Xây dựng system prompt, nhồi context, ràng buộc chống ảo giác (hallucination).

### 2. Video & Tài liệu học
- **Video bài giảng:**
  - Build Local RAG với Ollama: https://www.youtube.com/watch?v=KjGXSNHY-q4&t=217s
  - Đánh giá RAG với Ragas: https://www.youtube.com/watch?v=5fp6e5nhJRk
- **Tài liệu tham khảo:**
  - Paper gốc về RAG (Lewis et al.): https://arxiv.org/abs/2005.11401
  - Cài đặt Ollama: https://ollama.com/download
  - Embedding Model nomic-embed-text: https://ollama.com/library/nomic-embed-text
  - ChromaDB Documentation: https://docs.trychroma.com/docs/overview/introduction
  - LangChain RAG Guide: https://docs.langchain.com/oss/python/deepagents/rag
  - Ragas Framework: https://docs.ragas.io/en/latest/

### 3. Bài tập thực hành

#### Bài 1 — Local LLM & Prompt Constraint
- **Mục tiêu:** Nắm vững cách gọi LLM cục bộ bằng thư viện Python `ollama` và kiểm soát câu trả lời.
- **Thực hiện:**
  1. Khởi chạy mô hình cục bộ `qwen2.5:3b`.
  2. Đưa vào một đoạn quy định/nội quy mẫu.
  3. Thử nghiệm 3 biến thể prompt: Không hướng dẫn, Có gán vai trò (role-play), Có context kèm điều kiện nghiêm ngặt *"chỉ trả lời dựa trên context, không suy diễn"*.
- **Điều kiện hoàn thành:** Lưu 10 mẫu test vào CSV; chỉ rõ trường hợp prompt thành công chặn được ảo giác (hallucination).

#### Bài 2 — Pipeline RAG Local từ đầu (End-to-End)
- **Mục tiêu:** Tự triển khai đầy đủ pipeline RAG 100% offline, không dùng API trả phí.
- **Thực hiện:**
  1. Sử dụng `pypdf` để đọc và phân tích 3–5 file PDF tài liệu.
  2. Chia văn bản thành các chunks có độ dài phù hợp.
  3. Dùng mô hình `nomic-embed-text` qua Ollama để chuyển đổi chunks thành vectors.
  4. Lưu vectors, text chunks và metadata (file name, page number) vào ChromaDB.
  5. Khi nhận câu hỏi: vector search lấy top-3 chunks liên quan nhất và gửi kèm context đến `qwen2.5:3b`.
  6. Trả về câu trả lời có trích dẫn rõ nguồn (tên tài liệu, trang số).
- **Điều kiện hoàn thành:** Hệ thống trả lời chính xác theo tài liệu; khi không có dữ liệu trong context, hệ thống từ chối lịch sự ("Không đủ thông tin trong tài liệu").

#### Bài 3 — Tối ưu hóa Chunking và Retrieval
- **Mục tiêu:** Tìm cấu hình tối ưu thông qua thực nghiệm đo lường.
- **Thực nghiệm:** Thiết lập 3 cấu hình thử nghiệm:

| Cấu hình | Kích thước Chunk (Chunk Size) | Độ gối đầu (Overlap) | Top-k Retrieval |
| :---: | :---: | :---: | :---: |
| **A** | 300 tokens | 0 | 3 |
| **B** | 500 tokens | 50 tokens | 3 |
| **C** | 800 tokens | 50 tokens | 5 |

- Đánh giá trên 20 câu hỏi chuẩn:
  - Tỷ lệ tìm thấy đoạn chứa câu trả lời đúng (Hit Rate).
  - Số lượng chunk nhiễu/không liên quan bị kéo theo.
  - Thời gian truy xuất (Retrieval latency) và thời gian sinh câu trả lời (Generation latency).
- **Điều kiện hoàn thành:** Lập bảng so sánh số liệu và phân tích lý do chọn cấu hình tối ưu.

#### Bài 4 — Đánh giá chất lượng RAG chuyên sâu
- **Mục tiêu:** Kiểm tra các lỗi phổ biến trong bài toán doanh nghiệp.
- **Bộ câu hỏi kiểm thử:**
  - 10 câu có đáp án rõ ràng trong một đoạn.
  - 5 câu yêu cầu tổng hợp thông tin từ nhiều trang/đoạn khác nhau.
  - 5 câu bẫy không hề có thông tin trong tài liệu.
- **Các tiêu chí đánh giá:**
  - **Retrieval:** Tỷ lệ truy xuất đúng đoạn cần thiết.
  - **Faithfulness (Độ trung thực):** Mọi luận điểm trong câu trả lời có được tài liệu chứng minh không.
  - **Abstention (Khả năng từ chối):** Hệ thống có từ chối khi thiếu dữ liệu không.
  - **Latency:** Đo thời gian phản hồi từng bước.
- **Điều kiện hoàn thành:** Báo cáo phân tích ít nhất 5 trường hợp lỗi, nguyên nhân gốc rễ và giải pháp khắc phục (khuyến khích dùng Ragas).

### 4. Dự án tổng hợp: Local Enterprise RAG (Portfolio)
- Xây dựng ứng dụng hỏi đáp tài liệu doanh nghiệp chạy hoàn toàn cục bộ:
  - **Backend:** FastAPI với các endpoint chuẩn RESTful: `/ingest` (tải tài liệu), `/query` (hỏi đáp), `/health` (kiểm tra trạng thái).
  - **Core LLM & Embeddings:** Quản trị qua Ollama (`nomic-embed-text` + `qwen2.5:3b`).
  - **Vector Database:** ChromaDB lưu trữ persistent trên đĩa.
  - **Tính năng bắt buộc:** Trích dẫn nguồn (tên file, số trang, trích dẫn liên quan); xử lý lỗi file PDF hỏng, tài liệu rỗng; chống timeout.
  - **Chất lượng:** Logging chi tiết, bộ test suite 30 câu hỏi tự động, Dockerfile và Docker Compose.

---

## Module 5: AI Agents

### 1. Nội dung học
- Định nghĩa AI Agent và sự khác biệt giữa Chatbot thông thường, RAG Chatbot và AI Agent:
  - Chatbot: Chỉ giao tiếp phản hồi.
  - RAG: Tìm kiếm thông tin và tổng hợp.
  - AI Agent: Tự lập kế hoạch (Reasoning/Planning), gọi công cụ (Tool calling/Action), lưu nhớ ngữ cảnh (Memory) và lặp lại để đạt mục tiêu.
- Các kiến trúc Agent phổ biến: ReAct (Reasoning + Acting), Plan-and-Solve, Tool Use.
- Quản lý bộ nhớ (Memory): Short-term memory (trong hội thoại) và Long-term memory (lưu trữ vector/cơ sở dữ liệu).
- Xử lý lỗi khi gọi công cụ, kiểm soát an toàn (guardrails) và tránh lặp vô hạn (infinite loop).

### 2. Tài liệu học
- **Hugging Face Agents Course:** https://huggingface.co/learn/agents-course/en/unit1/introduction
- **Ollama Tool Calling Documentation:** https://docs.ollama.com/capabilities/tool-calling
- **LangGraph Persistence:** https://langchain-ai.github.io/langgraph/how-tos/persistence/
- **Dataset thực hành phân tích:** UCI Online Retail Dataset

### 3. Bài tập thực hành

#### Bài 1 — Tạo Tool Calling cơ bản cho Agent
- **Mục tiêu:** Xây dựng Agent chạy local có khả năng tự quyết định gọi công cụ Python.
- **Tập công cụ:**
  - `calculator(expression)`: Thực thi tính toán số học an toàn.
  - `search_notes(query)`: Tra cứu thông tin trong file ghi chú cục bộ.
  - `get_current_time()`: Lấy thời gian hiện tại của hệ thống.
- **Kiểm tra:** Yêu cầu người dùng cần kết hợp: tìm số liệu trong ghi chú → tính toán số học → trả lời ngày giờ báo cáo.
- **Đạt yêu cầu khi:** Agent gọi đúng công cụ với tham số chuẩn xác, dùng kết quả thực tế để trả lời, không bịa đặt số liệu.

#### Bài 2 — Agent có State & Conversation Memory
- **Mục tiêu:** Giữ ngữ cảnh hội thoại xuyên suốt nhiều lượt chat và phân tách độc lập các phiên người dùng.
- **Yêu cầu:**
  - Ghi nhớ tên dự án, mục tiêu và sở thích người dùng cung cấp ở các lượt trước.
  - Quản lý phiên hội thoại độc lập theo `session_id`.
  - Đảm bảo 2 session khác nhau không bị rò rỉ dữ liệu qua lại.
- **Đạt yêu cầu khi:** Agent truy hồi chính xác thông tin cũ của đúng phiên làm việc.

#### Bài 3 — Agent xử lý quy trình nhiều bước (Multi-step Task)
- **Tình huống:** Người dùng yêu cầu phân tích một file CSV dữ liệu bán hàng.
- **Workflow của Agent:**
  1. Kiểm tra file tồn tại và định dạng hợp lệ.
  2. Đọc và tiền xử lý dữ liệu bằng Pandas qua tool.
  3. Gọi tool tính toán doanh thu theo nhóm sản phẩm.
  4. Gọi tool tìm top sản phẩm bán chạy nhất.
  5. Tổng hợp dữ liệu thành báo cáo hoàn chỉnh dưới định dạng Markdown.
- **Đạt yêu cầu khi:** Mỗi bước có input/output minh bạch, xử lý ngoại lệ khi file lỗi và tuyệt đối không thực thi lệnh Python tùy ý không an toàn.

### 4. Dự án tổng hợp: Local AI Work Assistant (Portfolio)
- Xây dựng trợ lý ảo hỗ trợ công việc chạy cục bộ bằng Python, FastAPI và Ollama:
  - **Hỏi đáp tài liệu nội bộ:** Tích hợp RAG engine.
  - **Phân tích số liệu:** Bộ công cụ đọc bảng tính CSV, thống kê và vẽ biểu đồ.
  - **Bộ nhớ phiên:** Quản lý hội thoại đa phiên.
  - **Báo cáo tự động:** Xuất kết quả ra file Markdown/CSV.
  - **Tiêu chuẩn an toàn:** Giới hạn số lần gọi tool tối đa (max iterations = 5) để chặn vòng lặp vô hạn; kiểm tra quyền xác nhận người dùng trước các thao tác ghi/xóa dữ liệu; ghi log có cấu trúc (structured logging) theo dõi thời gian và trạng thái gọi tool.

---

## Module 6: AI Deployment in Production

### 1. Bối cảnh & Mục tiêu thực tế
- Chuyển giao ứng dụng AI từ bản thử nghiệm (prototype) sang dịch vụ production sẵn sàng phục vụ người dùng thực tế:
  - Cung cấp API chuẩn cho các client (Web, Mobile).
  - Khả năng chịu lỗi cao: Xử lý ngoại lệ khi LLM service gián đoạn, database mất kết nối hoặc tài liệu lỗi.
  - Đảm bảo chất lượng câu trả lời và trích dẫn.
  - Giám sát hiệu năng hệ thống, phát hiện điểm nghẽn và đo tải thực tế.

### 2. Yêu cầu kỹ thuật chi tiết

#### A. Xây dựng Server & Client (FastAPI + Ollama)
Thiết kế hệ thống API backend bằng FastAPI:

| Phương thức & Endpoint | Chức năng chính |
| :--- | :--- |
| `GET /health` | Kiểm tra trạng thái hoạt động của backend service (Liveness probe) |
| `GET /health/ready` | Kiểm tra tính sẵn sàng của các phụ thuộc: Ollama, Vector DB (Readiness probe) |
| `POST /documents` | Nhận upload, xác thực và xử lý ingest file PDF vào hệ thống |
| `GET /documents/{id}` | Truy vấn tiến độ và trạng thái xử lý của tài liệu |
| `POST /chat` | Nhận câu hỏi, truy xuất context và trả lời kèm trích dẫn nguồn |

Xây dựng kèm một Python Client mẫu để tương tác với API, render kết quả và bắt lỗi chuẩn xác.

#### B. Xử lý các tình huống thực tế (Edge cases & Resiliency)
Hệ thống phải có xử lý lỗi và HTTP status code tương ứng cho các trường hợp:
- File upload sai định dạng, file rỗng hoặc kích thước vượt ngưỡng cho phép (HTTP 400 / 413).
- Câu hỏi rỗng hoặc vượt quá giới hạn độ dài token (HTTP 400 / 422).
- Không tìm thấy dữ liệu liên quan trong kho tài liệu (Trả lời từ chối theo kịch bản, HTTP 200).
- Dịch vụ Ollama bị treo, timeout hoặc ngừng hoạt động (HTTP 503 / 504 kèm thông báo rõ ràng).
- Kết nối tới Vector Database bị đứt đoạn (HTTP 500 kèm logging cảnh báo).
- Xử lý hàng đợi khi có nhiều request đồng thời.

#### C. Kiểm thử & Đánh giá (Testing & Evaluation)
- **Software Testing:**
  - Unit test cho module tiền xử lý dữ liệu, trích xuất text và kiểm tra logic validation.
  - API testing tự động bằng `pytest` và `FastAPI TestClient`.
  - Integration test kiểm thử luồng nghiệp vụ hoàn chỉnh: Upload → Ingest → Truy xuất → Trả lời.
  - Áp dụng Mocking đối với các service LLM bên ngoài để test cô lập các tình huống lỗi.
- **AI Evaluation:**
  - Bộ benchmark tối thiểu 30 câu hỏi: câu hỏi có đáp án đơn, câu hỏi đa đoạn (multi-hop) và câu hỏi bẫy.
  - Đo lường định lượng: Retrieval Recall@k, Faithfulness, Answer Correctness, Citation Accuracy.
  - Sử dụng framework Ragas để tự động hóa quá trình đánh giá và lưu kết quả theo dõi.

#### D. Giám sát & Theo dõi hiệu năng (Observability & Performance)
- **Tracing:** Tích hợp OpenTelemetry thu thập trace chi tiết cho từng request (thời gian tiền xử lý, thời gian vector search, thời gian LLM sinh câu trả lời).
- **Chỉ số đo lường bắt buộc:**

| Chỉ số (Metric) | Ý nghĩa nghiệp vụ |
| :--- | :--- |
| **p50 & p95 Latency** | Thời gian phản hồi trung vị và thời gian phản hồi ở nhóm 95% request |
| **Error Rate** | Tỷ lệ phần trăm số request gặp lỗi trên tổng số request |
| **Throughput (RPS)** | Số lượng request hệ thống có thể xử lý thành công trong một giây |
| **Retrieval Latency** | Thời gian tìm kiếm và xếp hạng tài liệu trong Vector Database |
| **LLM Inference Latency** | Thời gian mô hình tạo token và hoàn thành câu trả lời |

- **Load Testing:** Sử dụng Locust mô phỏng tải ở các mức 1, 5 và 10 người dùng đồng thời. Tìm nút thắt hiệu năng (bottleneck), thử nghiệm thay đổi `chunk_size` hoặc `top_k` để đánh giá sự đánh đổi giữa tốc độ và độ chính xác.

#### E. Đóng gói & Tiêu chuẩn triển khai (Docker & CI)
- Đóng gói toàn bộ ứng dụng bằng Docker và Docker Compose (FastAPI backend, ChromaDB service).
- Quản lý biến môi trường an toàn qua file `.env.example`, không commit bí mật (secrets).
- Tích hợp pipeline kiểm thử tự động chạy qua CI trước khi build container.
- Hướng dẫn triển khai chi tiết từng bước trong file `README.md`.

### 3. Video & Tài liệu tham khảo
- **Video thực hành:**
  - FastAPI & Docker Compose: https://youtu.be/vH1q89WFOzg?si=p2KzP4rdKMz6hHiH
  - Hướng dẫn Docker nâng cao: https://youtu.be/pg19Z8LL06w?si=Afycp1PfGQOlC8sL
  - Kiểm thử phần mềm với Pytest: https://www.youtube.com/watch?v=gop9Or2V_80
  - Đánh giá chất lượng RAG với Ragas: https://www.youtube.com/watch?v=5fp6e5nhJRk
- **Tài liệu chính thức:**
  - FastAPI Tutorial & Testing: https://fastapi.tiangolo.com/tutorial/ | https://fastapi.tiangolo.com/tutorial/testing/
  - Docker Python Guide & Compose: https://docs.docker.com/guides/python/ | https://docs.docker.com/compose/
  - Pytest Documentation: https://docs.pytest.org/en/stable/getting-started.html
  - Ragas Metrics Guide: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
  - OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
  - Locust Load Testing: https://docs.locust.io/en/stable/
  - Prometheus & Grafana: https://prometheus.io/docs/introduction/overview/ | https://grafana.com/docs/

---

## Module 7: LLM Data Preparation & Model Evaluation

### 1. Nội dung học

**A. Xử lý và làm sạch dữ liệu văn bản (Data Cleaning & Quality Assurance)**
- Xử lý dữ liệu khuyết thiếu (missing values), chuẩn hóa encoding, định dạng văn bản và lọc bỏ nội dung nhiễu.
- Kỹ thuật lọc trùng lặp: Deduplication chính xác và Near-deduplication (sử dụng MinHash, LSH).
- Quy trình gán nhãn dữ liệu (Data Annotation) theo schema định sẵn và xây dựng hướng dẫn gán nhãn (labeling guideline).
- Kiểm tra chất lượng dữ liệu: phát hiện nhãn xung đột, mẫu gán nhãn sai, kiểm tra tính nhất quán và quy trình QA/QC trước training.
- Chiến lược phân chia train/validation/test nghiêm ngặt: loại bỏ rò rỉ dữ liệu (data leakage) giữa các tập.

**B. Chuẩn bị Dataset chuyên biệt cho Huấn luyện LLM**
- Quản lý nguồn gốc dữ liệu (data provenance), bản quyền và tính riêng tư.
- Chuẩn hóa cấu trúc dataset theo các kỹ thuật huấn luyện:
  - **SFT (Supervised Fine-Tuning):** Định dạng `instruction – input – output` hoặc định dạng hội thoại `messages` (`role: system/user/assistant`, `content`).
  - **DPO (Direct Preference Optimization):** Cấu trúc bộ ba `prompt – chosen – rejected` phục vụ căn chỉnh sở thích con người (alignment).
- Kiểm tra tính hợp lệ của schema, trường bắt buộc, dữ liệu rỗng và độ dài ký tự.
- Phân tích phân phối độ dài token (Token length distribution), xử lý các mẫu vượt ngưỡng context window của mô hình.
- Quản lý phiên bản dữ liệu (data versioning) và metadata bằng Hugging Face Datasets để đảm bảo tính tái lập thực nghiệm.

**C. Đánh giá mô hình & Xây dựng Benchmark**
- Thiết kế prompt template chuẩn phục vụ quy trình đánh giá nhất quán.
- Xây dựng bộ benchmark hoàn chỉnh: các ca kiểm thử điển hình, ca kiểm thử biên (edge cases) và đáp án tham chiếu.
- Lựa chọn metric phù hợp:
  - Bài toán phân loại / trích xuất: Accuracy, Precision, Recall, F1-score.
  - Bài toán sinh tự do: BLEU, ROUGE, độ chính xác ngữ nghĩa (Semantic similarity), tỷ lệ tuân thủ format.
  - Đánh giá bằng LLM-as-a-judge (sử dụng LLM mạnh hơn để chấm điểm theo rubric).
- Đánh giá đối đầu (A/B testing) giữa các checkpoint trước và sau training/fine-tuning.
- Quy trình Error Analysis: phân loại các nhóm lỗi chính của mô hình và đề xuất giải pháp cải thiện tập dữ liệu.

### 2. Tài liệu học
- **Hugging Face Datasets:** https://huggingface.co/docs/datasets/
- **Hugging Face TRL — SFT Trainer:** https://huggingface.co/docs/trl/sft_trainer
- **Hugging Face TRL — DPO Trainer:** https://huggingface.co/docs/trl/dpo_trainer
- **Hugging Face Tokenizers:** https://huggingface.co/docs/tokenizers/
- **Hugging Face Evaluate:** https://huggingface.co/docs/evaluate/

### 3. Bài tập thực hành

#### Bài 1 — Làm sạch và kiểm tra chất lượng Dataset
- Chuẩn bị một bộ dữ liệu văn bản thô chứa dữ liệu thiếu, trùng lặp, lỗi encoding và nhãn mâu thuẫn.
- Xây dựng script tiền xử lý tự động và xuất báo cáo thống kê số lượng mẫu trước và sau làm sạch.
- Thiết lập quy trình QA/QC tự động phát hiện các mẫu không đạt tiêu chuẩn chất lượng.

#### Bài 2 — Xây dựng Dataset chuẩn SFT
- Chuyển đổi dữ liệu đã làm sạch sang định dạng chuẩn SFT (`instruction – input – output` hoặc hội thoại `messages`).
- Viết script kiểm tra schema: loại bỏ các trường rỗng, kiểm tra tính hợp lệ của cú pháp JSON/Dictionary.
- Sử dụng tokenizer đo phân phối độ dài token, lọc hoặc cắt tỉa các mẫu vượt ngưỡng context window.
- Phân chia train/val/test và kiểm tra xác suất trùng lặp dữ liệu giữa các tập.

#### Bài 3 — Xây dựng Dataset chuẩn DPO
- Thiết kế bộ dữ liệu gồm các bộ ba `prompt – chosen – rejected`.
- Kiểm tra tính hợp lệ: cùng một prompt, phản hồi `chosen` phải thể hiện câu trả lời tốt hơn rõ ràng so với `rejected` dựa trên tiêu chí định sẵn (chính xác, hữu ích, không độc hại).
- Lưu và tải dataset bằng thư viện `datasets` của Hugging Face, kiểm tra tương thích với `DPOTrainer`.

#### Bài 4 — Thiết kế Benchmark đánh giá mô hình
- Xây dựng bộ test suite gồm 30–50 mẫu kiểm thử bao quát các tình huống thường gặp và câu hỏi bẫy.
- Xây dựng barem điểm tham chiếu và script chấm điểm tự động.
- Lưu lại toàn bộ output dự đoán và báo cáo đánh giá ra file CSV/JSON.

#### Bài 5 — Đánh giá so sánh trước và sau huấn luyện (Pre vs Post Fine-Tuning)
- Chạy đánh giá trên cùng một bộ benchmark cho mô hình nền (base model) và mô hình sau khi fine-tune.
- So sánh các chỉ số định lượng, tốc độ suy luận và mức độ khắc phục các ca lỗi điển hình.
- Viết báo cáo kỹ thuật tổng kết kết quả, các hạn chế còn tồn đọng và định hướng cải tiến chất lượng dữ liệu ở chu kỳ tiếp theo.