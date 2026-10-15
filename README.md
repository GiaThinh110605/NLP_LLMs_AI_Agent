- machine learning slide: https://canva.link/6xdb3n1gpijx1kw
- deep learning slide: https://canva.link/6xdb3n1gpijx1kw
- fastapi slide: https://canva.link/840ya322qx6ytve
- nlp slide: https://canva.link/i0xl1006o17uc2g
- thực hành machine learning sử dụng scikit learn: https://www.youtube.com/live/fm8nn374caw?si=Rz_JetT6dEmlPhM-

> Khi học chỉ tra google không nên sử dụng AI

---

## Module 1: NLP Fundamentals

### Nội dung học

- Tổng quan về NLP, các bài toán phổ biến và vai trò của NLP trong AI hiện đại.
- Lịch sử phát triển của NLP: từ các phương pháp truyền thống đến Machine Learning, Deep Learning và LLM.
- Các kỹ thuật xử lý văn bản cơ bản: tokenization, normalization, stopwords, stemming và lemmatization.
- Biểu diễn văn bản: Bag of Words, n-grams, TF-IDF và khái niệm word embeddings.
- Các bài toán NLP phổ biến: text classification, sentiment analysis, named entity recognition (NER), information extraction và text similarity.
- Xây dựng pipeline NLP cơ bản: thu thập dữ liệu, làm sạch văn bản, tiền xử lý, biểu diễn dữ liệu và đưa vào mô hình.
- Chia tập dữ liệu train/validation/test; hiểu data leakage và tầm quan trọng của dữ liệu chất lượng.
- Các chỉ số đánh giá cơ bản: accuracy, precision, recall và F1-score.
- Thực hành xử lý và phân tích một bộ dữ liệu văn bản; xây dựng mô hình phân loại văn bản đơn giản bằng TF-IDF và Logistic Regression.

### Tài liệu học

- Video tổng quan: https://www.youtube.com/live/4O0HPXiDpTs?si=GOmMVvysstUO2zl1
- NLTK Book: https://www.nltk.org/book_1ed/
- NLTK Documentation: https://www.nltk.org/
- Scikit-learn — Text Feature Extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
- Scikit-learn — Text Classification: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

### Bài tập thực hành

1. Làm sạch và chuẩn hóa dữ liệu văn bản, xử lý dữ liệu rỗng và trùng lặp.
2. Thực hành tokenization, loại bỏ stopwords, stemming và lemmatization.
3. Biểu diễn văn bản bằng Bag of Words và TF-IDF; so sánh kết quả.
4. Xây dựng mô hình phân loại cảm xúc hoặc phân loại văn bản bằng Logistic Regression.
5. Chia dữ liệu thành train/validation/test, tránh data leakage và đánh giá bằng accuracy, precision, recall, F1-score.
6. Phân tích các mẫu dự đoán sai và viết báo cáo ngắn về pipeline, kết quả và hạn chế của mô hình.

---

## Module 2: Deep Learning in NLP

- Neural Network trong NLP và cách deep learning cải thiện khả năng xử lý ngôn ngữ.
- Tìm hiểu các kiến trúc RNN, LSTM, GRU: nguyên lý hoạt động, ưu và nhược điểm.
- Ứng dụng các mô hình tuần tự vào bài toán xử lý văn bản.
- Thực hành xây dựng và huấn luyện mô hình NLP cơ bản bằng PyTorch.

### youtube:
- https://www.youtube.com/live/pusntJk4tsw?si=eV0oiDSbunvAJ0Ct
- https://www.youtube.com/live/_J1tXiC1DLY?si=8SV6nDGYFBi97qJC
- https://www.youtube.com/playlist?list=PLqL-7eLmqd9V3faivSAST76YQClS44dSz

### tài liệu:
- https://docs.pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html
- https://huggingface.co/datasets/stanfordnlp/imdb

### Bài tập:

#### Bài 1 — Neural Network cơ bản
- Chuyển văn bản thành token ID.
- Tạo vocabulary và padding chuỗi.
- Xây dựng mô hình embedding + pooling + Linear.
- Huấn luyện và đánh giá Accuracy, F1-score.

#### Bài 2 — LSTM
- Thay mô hình cơ bản bằng nn.LSTM.
- Sử dụng hidden state để phân loại cảm xúc.
- Theo dõi loss và kết quả trên tập test.
- Thử dự đoán các câu đánh giá mới.

#### Bài 3 — So sánh RNN, LSTM và GRU
- Giữ nguyên cách chia dữ liệu, vocabulary và các điều kiện huấn luyện.
- So sánh Accuracy, F1-score, thời gian huấn luyện.
- Giải thích ưu nhược điểm và các trường hợp mô hình dự đoán sai.

---

## Module 3: Transformer & LLMs

### 1. Nội dung học

**A. Nền tảng Transformer và LLM**
- Hiểu kiến trúc Transformer, Self-Attention, Positional Encoding và Embeddings.
- Phân biệt Transformer với RNN/LSTM; phân biệt Encoder-only, Decoder-only và Encoder-Decoder.
- Tìm hiểu cách LLM được huấn luyện và sinh văn bản; hiểu token, context window và giới hạn đầu vào.
- Tổng quan về các LLM phổ biến và tiêu chí lựa chọn mô hình theo chất lượng, chi phí và tài nguyên.

**B. Prompt Engineering**
- Học các kỹ thuật zero-shot, few-shot, instruction prompting và prompt template.
- Tìm hiểu ảnh hưởng của context, giới hạn token và cách viết prompt đến chất lượng đầu ra.
- Thực hành kiểm soát định dạng đầu ra bằng JSON và xây dựng prompt có cấu trúc.

**C. Đánh giá mô hình và Benchmark**
- Xây dựng bộ câu hỏi kiểm thử cùng đáp án tham chiếu và tiêu chí đánh giá.
- Đánh giá độ chính xác, mức độ liên quan, khả năng tuân thủ yêu cầu và tính nhất quán.
- So sánh các prompt hoặc mô hình trên cùng bộ dữ liệu kiểm thử.
- Ghi lại kết quả thực nghiệm, phân tích lỗi và đề xuất cải thiện.

**D. Phân biệt RAG và Fine-tuning**
- Hiểu nguyên lý hoạt động, ưu điểm và hạn chế của RAG và Fine-tuning.
- Biết khi nào nên dùng RAG, Fine-tuning hoặc kết hợp cả hai.
- Hiểu cơ bản về PEFT/LoRA để fine-tune mô hình nhỏ với tài nguyên hạn chế.
- Biết cách đánh giá và so sánh hai phương pháp theo cùng bài toán.

### 2. Video học tập

- Video 1: https://www.youtube.com/live/VCq4YFYAj-Q?si=fQPAbUMXpS35tmMD
- Video 2: https://www.youtube.com/playlist?list=PLqL-7eLmqd9V3faivSAST76YQClS44dSz

### 3. Tài liệu học

**Transformer và LLM**
- Paper gốc *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
- Hugging Face LLM Course: https://huggingface.co/learn/llm-course/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/

**Prompt Engineering**
- OpenAI Prompt Engineering: https://developers.openai.com/api/docs/guides/prompt-engineering

**Fine-tuning tiết kiệm tài nguyên**
- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- Hugging Face Transformers Trainer: https://huggingface.co/docs/transformers/main_classes/trainer

**RAG và đánh giá**
- Hugging Face RAG documentation: https://huggingface.co/docs/transformers/model_doc/rag
- Ragas – đánh giá hệ thống RAG: https://docs.ragas.io/en/latest/
- Hugging Face Evaluate: https://huggingface.co/docs/evaluate/

### 4. Bài tập thực hành

**Bài 1 – Thử nghiệm Prompt Engineering**
- Chọn một tác vụ, chẳng hạn phân loại phản hồi khách hàng hoặc hỏi đáp.
- Tạo 10 câu hỏi và thử nghiệm 3 cách prompting: zero-shot, few-shot và instruction prompting.
- Ghi lại kết quả, xác định prompt nào tốt hơn và giải thích nguyên nhân.

**Bài 2 – Thí nghiệm Context và Token**
- Thử nghiệm với context ngắn, context dài và context có thông tin nhiễu.
- Thay đổi độ dài đầu vào và giới hạn token đầu ra.
- Ghi nhận chất lượng câu trả lời, số token sử dụng và thời gian phản hồi.

**Bài 3 – Xây dựng Benchmark**
- Tạo bộ kiểm thử khoảng 30–50 câu hỏi có đáp án tham chiếu.
- Xây dựng tiêu chí đánh giá và lưu kết quả vào CSV/JSON.
- So sánh các prompt hoặc mô hình trên cùng bộ câu hỏi để bảo đảm kết quả có thể đối chiếu.

**Bài 4 – Xây dựng hệ thống RAG**
- Sử dụng một tập tài liệu PDF làm nguồn tri thức.
- Thực hiện pipeline: đọc tài liệu → chia chunk → tạo embeddings → truy xuất → sinh câu trả lời.
- Yêu cầu câu trả lời trích dẫn tài liệu nguồn.
- Đánh giá độ chính xác, khả năng truy xuất đúng đoạn và tính chính xác của trích dẫn.

**Bài 5 – Fine-tuning mô hình nhỏ**
- Chọn mô hình nhỏ phù hợp với tài nguyên máy và một tác vụ cụ thể.
- Chuẩn bị dataset theo định dạng phù hợp, chia train/validation và fine-tune bằng LoRA/PEFT.
- Đánh giá mô hình trước và sau fine-tuning trên cùng bộ kiểm thử.
- Ghi lại cấu hình, thời gian huấn luyện, kết quả và hạn chế.

**Bài 6 – So sánh RAG và Fine-tuning**
- Sử dụng cùng bài toán và bộ câu hỏi kiểm thử để so sánh hai phương pháp.
- So sánh độ chính xác, chi phí, độ trễ, khả năng cập nhật dữ liệu và mức độ phù hợp với yêu cầu.
- Tổng hợp kết quả bằng bảng và viết báo cáo kỹ thuật, đưa ra kết luận nên chọn phương pháp nào trong từng tình huống.

### 5. Sản phẩm đầu ra

- Notebook thực hành Prompt Engineering và Transformer/LLM.
- Bộ benchmark có câu hỏi, đáp án tham chiếu và kết quả đánh giá.
- Một hệ thống RAG hỏi đáp tài liệu PDF có trích dẫn nguồn.
- Một mô hình nhỏ đã fine-tune cùng kết quả đánh giá trước/sau huấn luyện.
- Báo cáo so sánh RAG và Fine-tuning, kèm mã nguồn và hướng dẫn chạy.s

---

## Module 4: RAG

- Hiểu khái niệm RAG và workflow cơ bản của một hệ thống truy xuất kết hợp sinh câu trả lời.
- Tìm hiểu các thành phần chính: chunking, embeddings, vector store, retrieval và generation.
- Cách chuẩn bị tài liệu và tối ưu dữ liệu đầu vào cho hệ thống RAG.
- Thực hành xây dựng một RAG chatbot cơ bản có thể trả lời theo tài liệu.

- build rag bằng ollama: https://www.youtube.com/watch?v=KjGXSNHY-q4&t=217s
- hướng dẫn chuẩn từ langchain build rag bằng api: https://docs.langchain.com/oss/python/deepagents/rag
- bài báo gốc rag: https://arxiv.org/abs/2005.11401

### Bài tập: 
- hướng dẫn tải ollama: https://ollama.com/download
- lưu vector local: https://docs.trychroma.com/docs/overview/introduction
- embedding model: https://ollama.com/library/nomic-embed-text

#### Bài 1 — Local LLM và Prompting
**Cấp độ 1 · 30–45 phút**  
**Mục tiêu:** Hiểu cách gọi LLM local và ảnh hưởng của prompt.

**Thực hiện**
1. Gọi qwen2.5:3b từ Python bằng thư viện ollama.
2. Đặt câu hỏi về một đoạn nội quy mẫu.
3. Thử ba prompt: không có hướng dẫn, có vai trò, có context và quy tắc chỉ trả lời theo context.
4. So sánh câu trả lời và thời gian phản hồi.

**Điều kiện hoàn thành:** Lưu ít nhất 10 câu hỏi và kết quả vào CSV; xác định trường hợp mô hình trả lời thông tin không có trong tài liệu.

#### Bài 2 — Xây dựng pipeline RAG local
**Cấp độ 2 · 2–3 giờ**  
**Mục tiêu:** Tự triển khai đầy đủ pipeline, không dùng API trả phí.

**Thực hiện**
1. Chuẩn bị 3–5 tài liệu PDF/TXT.
2. Dùng pypdf để đọc PDF.
3. Chia văn bản thành chunks.
4. Gọi nomic-embed-text để tạo embeddings.
5. Lưu vector, chunks và metadata vào ChromaDB.
6. Khi có câu hỏi, truy xuất top-3 chunks rồi gửi cho qwen2.5:3b.
7. Trả về câu trả lời kèm tên file và trang nguồn.

**Điều kiện hoàn thành:** Chatbot trả lời được câu hỏi dựa trên tài liệu; nếu không tìm thấy bằng chứng phù hợp thì báo không đủ thông tin.  
**Tham khảo:** Ollama Embeddings

#### Bài 3 — Tối ưu chunking và retrieval
**Cấp độ 3 · 2–3 giờ**  
**Mục tiêu:** Tìm ra cấu hình tốt hơn bằng thực nghiệm.

Thử các cấu hình sau:

| Thí nghiệm | Kích thước chunk | Overlap | Top-k |
| :--- | :--- | :--- | :--- |
| A | 300 tokens | 0 | 3 |
| B | 500 tokens | 50 | 3 |
| C | 800 tokens | 50 | 5 |

Với 20 câu hỏi có đáp án được xác định trước:
- Ghi lại chunks được truy xuất.
- Đánh giá có tìm thấy đoạn chứa đáp án hay không.
- Đếm số đoạn không liên quan.
- Đo thời gian truy xuất và thời gian sinh câu trả lời.

**Điều kiện hoàn thành:** Có bảng so sánh và giải thích tại sao chọn cấu hình cuối cùng. Lưu ý: đây là kích thước thử nghiệm ban đầu, không phải cấu hình tối ưu cho mọi tài liệu.

#### Bài 4 — Đánh giá chất lượng RAG
**Cấp độ 4 · 2 giờ**  
**Mục tiêu:** Kiểm tra các lỗi thường gặp trong ứng dụng doanh nghiệp.

Tạo bộ câu hỏi gồm:
- 10 câu có đáp án rõ ràng.
- 5 câu cần tổng hợp nhiều đoạn.
- 5 câu không có đáp án trong tài liệu.

Đánh giá:
- Retrieval: Có tìm đúng đoạn cần thiết không?
- Faithfulness: Câu trả lời có được hỗ trợ bởi tài liệu không?
- Abstention: Hệ thống có từ chối khi thiếu bằng chứng không?
- Latency: Mỗi câu mất bao lâu?

**Điều kiện hoàn thành:** Báo cáo ít nhất 5 lỗi, xác định nguyên nhân và đề xuất cách sửa. Có thể dùng Ragas để tự động hóa một phần đánh giá.

#### Bài 5 — Dự án tổng hợp: Local Enterprise RAG
**Dự án portfolio**

Xây dựng chatbot hỏi đáp tài liệu công ty chạy hoàn toàn local.

**Yêu cầu**
- Backend FastAPI với endpoint /ingest, /query và /health.
- Ollama chạy LLM và embedding model.
- ChromaDB lưu vector bền vững.
- Trả lời có trích dẫn tên file, trang và đoạn liên quan.
- Xử lý PDF lỗi, tài liệu rỗng, câu hỏi không liên quan và lỗi Ollama.
- Có logging, bộ kiểm thử 30 câu hỏi, README và Docker.
- Đo thời gian phản hồi và so sánh ít nhất hai cấu hình retrieval.

**Đầu ra GitHub:** Source code, tài liệu hướng dẫn, bộ câu hỏi kiểm thử, kết quả đánh giá và sơ đồ kiến trúc.

---

## Module 5: AI Agents

- Hiểu AI Agent là gì và cách agent khác với chatbot hoặc RAG chatbot.
- Tìm hiểu các thành phần cốt lõi của agent: model, memory, tools và workflow.
- Cách cho agent sử dụng công cụ để xử lý các tác vụ nhiều bước.
- Thực hành xây dựng một AI Agent đơn giản cho bài toán thực tế.

- https://huggingface.co/learn/agents-course/en/unit1/introduction
- https://docs.ollama.com/capabilities/tool-calling

### Bài 1 — Tạo tool cho agent
**Cơ bản · 1 giờ**

Dùng Ollama để xây dựng agent có ba công cụ Python:
- calculator(expression): tính toán biểu thức an toàn.
- search_notes(query): tìm kiếm trong file ghi chú cục bộ.
- get_current_time(): trả về thời gian hiện tại của máy.

**Bài kiểm tra:** Yêu cầu agent tìm một số liệu trong ghi chú, tính tổng rồi giải thích kết quả.  
**Đạt yêu cầu khi:** Agent thực sự gọi đúng công cụ, dùng kết quả công cụ trong câu trả lời và không tự bịa kết quả tính toán.  
**Tài liệu:** Ollama Tool Calling.

### Bài 2 — Agent có memory
**Trung bình · 1–2 giờ**

Xây dựng agent nhớ ngữ cảnh hội thoại qua nhiều lượt.

**Nhiệm vụ**
- Ghi nhớ tên dự án và mục tiêu do người dùng cung cấp.
- Trả lời câu hỏi tiếp nối dựa trên lịch sử.
- Lưu lịch sử theo session_id.
- Kiểm tra hai phiên người dùng không bị lẫn dữ liệu.

**Đạt yêu cầu khi:** Agent nhớ được thông tin trong đúng phiên và không nhầm giữa hai phiên.  
**Tài liệu:** LangGraph Persistence.

### Bài 3 — Agent xử lý công việc nhiều bước
**Nâng cao · 2–3 giờ**

**Tình huống:** Người dùng yêu cầu phân tích file CSV bán hàng.

Agent cần:
1. Kiểm tra file có tồn tại và đúng định dạng.
2. Đọc dữ liệu bằng Pandas.
3. Gọi tool tính doanh thu theo sản phẩm.
4. Gọi tool tìm sản phẩm bán chạy nhất.
5. Tổng hợp kết quả thành báo cáo Markdown.

**Đạt yêu cầu khi:** Mỗi bước có đầu vào và đầu ra rõ ràng, xử lý được file lỗi và không thực thi lệnh Python tùy ý do LLM tự sinh.  
**Dữ liệu mẫu:** UCI Online Retail.

### Bài 4 — Dự án tổng hợp: Local AI Work Assistant
**Portfolio**

Xây dựng trợ lý công việc local bằng Ollama, Python và FastAPI.

**Chức năng**
- Hỏi đáp trên tài liệu nội bộ bằng RAG.
- Đọc CSV và thống kê bằng các tool đã định nghĩa.
- Nhớ lịch sử theo phiên.
- Tạo báo cáo Markdown/CSV.
- Ghi log tool được gọi, lỗi và thời gian xử lý.

**Yêu cầu chất lượng**
- Có ít nhất 20 test case.
- Có giới hạn số lần gọi tool để tránh vòng lặp vô hạn.
- Có timeout và xử lý lỗi.
- Không tự ý xóa file, gửi dữ liệu ra ngoài hoặc thực hiện hành động ghi dữ liệu quan trọng khi chưa được xác nhận.
- Có README, sơ đồ workflow và hướng dẫn chạy local.

---

## Module 6: AI Deployment in Production

- Tổng quan về cách đưa chatbot hoặc AI Agent từ bản demo sang ứng dụng thực tế.
- Xây dựng server và client cơ bản để kết nối mô hình và xử lý request.
- Chạy thử, đánh giá các kịch bản sử dụng và theo dõi hiệu năng hệ thống.
- Tìm hiểu các lưu ý quan trọng khi triển khai AI trong môi trường production.

- https://docs.docker.com/guides/python/
- https://fastapi.tiangolo.com/tutorial/
- https://opentelemetry.io/docs/languages/python/

### 1. Bối cảnh thực tế
Một doanh nghiệp muốn xây dựng chatbot nội bộ giúp nhân viên hỏi đáp từ tài liệu PDF. Hệ thống phải cung cấp API cho ứng dụng client, trả lời dựa trên tài liệu, xử lý lỗi khi mô hình không hoạt động, kiểm tra chất lượng câu trả lời và theo dõi hiệu năng khi có nhiều request.

Bạn đóng vai trò AI Engineer phụ trách đưa ứng dụng từ prototype sang dịch vụ có thể triển khai và giám sát.

### 2. Yêu cầu chức năng

#### A. Xây dựng server và client
Sử dụng Python, FastAPI và Ollama.

| Endpoint | Yêu cầu |
| :--- | :--- |
| `GET /health` | Kiểm tra backend còn hoạt động |
| `GET /health/ready` | Kiểm tra các dependency cần thiết |
| `POST /documents` | Upload và xử lý PDF |
| `GET /documents/{id}` | Xem trạng thái xử lý tài liệu |
| `POST /chat` | Hỏi đáp và trả về nguồn trích dẫn |

Tạo một Python client có khả năng gọi API, hiển thị câu trả lời và xử lý response lỗi.

#### B. Xử lý các tình huống thực tế
Hệ thống cần xử lý tối thiểu:
- File không đúng định dạng, file rỗng hoặc vượt giới hạn kích thước.
- Câu hỏi thiếu nội dung hoặc quá dài.
- Không tìm thấy thông tin phù hợp trong tài liệu.
- Ollama timeout hoặc ngừng hoạt động.
- Database mất kết nối.
- Nhiều request được gửi cùng lúc.

Mỗi tình huống phải có hành vi mong đợi, HTTP status code phù hợp và test chứng minh.

### 3. Yêu cầu kiểm thử và đánh giá
Chia thành hai nhóm riêng biệt.

#### Nhóm 1 — Software testing
- Unit test cho xử lý tài liệu, retrieval và validation.
- API test bằng pytest và FastAPI TestClient.  
  *FastAPI*
- Integration test cho luồng upload → truy xuất → trả lời.
- Mock model khi kiểm thử lỗi để không phụ thuộc vào model đang chạy.
- Chạy toàn bộ test tự động trước khi đóng gói ứng dụng.

#### Nhóm 2 — AI evaluation
Tạo dataset gồm tối thiểu 30 câu hỏi và đáp án tham chiếu, bao gồm câu hỏi có đáp án, không có đáp án và câu hỏi dễ gây nhầm lẫn.

Đo các chỉ số:
- Retrieval Recall@k.
- Faithfulness.
- Answer correctness.
- Citation correctness.

Sử dụng Ragas khi phù hợp; lưu kết quả thành CSV hoặc JSON để so sánh các cấu hình retrieval khác nhau. Ragas có các metric phục vụ đánh giá retrieval và chất lượng câu trả lời.  
*Ragas*

### 4. Yêu cầu theo dõi hiệu năng
Tích hợp OpenTelemetry để ghi nhận trace cho từng request và các bước xử lý chính.  
*OpenTelemetry*

Các chỉ số bắt buộc:

| Chỉ số | Cần trả lời câu hỏi |
| :--- | :--- |
| p50 và p95 latency | Request mất bao lâu? |
| Error rate | Bao nhiêu phần trăm request thất bại? |
| Throughput | Xử lý được bao nhiêu request mỗi giây? |
| Retrieval latency | Tìm kiếm tài liệu mất bao lâu? |
| LLM latency | Model mất bao lâu để tạo câu trả lời? |

Sử dụng công cụ load testing như Locust để chạy thử ở mức 1, 5 và 10 người dùng đồng thời. Ghi rõ cấu hình máy và model, vì kết quả phụ thuộc vào tài nguyên local.

Bài thực hành bắt buộc: tìm ra nút thắt hiệu năng, thay đổi ít nhất một yếu tố như top_k, chunk size hoặc timeout, sau đó đo lại và so sánh kết quả.

### 5. Yêu cầu Docker và triển khai
Đóng gói ứng dụng bằng Docker, sử dụng Docker Compose để quản lý các service cần thiết. Thực hiện theo Docker Python Guide.  
*Docker Docs*

- youtube đánh giá bằng ragas: https://www.youtube.com/watch?v=5fp6e5nhJRk
- kiểm thử bằng pytest: https://www.youtube.com/watch?v=gop9Or2V_80
- fastapi và docker compose: https://youtu.be/vH1q89WFOzg?si=p2KzP4rdKMz6hHiH
- https://youtu.be/pg19Z8LL06w?si=Afycp1PfGQOlC8sL

Bản hoàn thành phải:
- Chạy được bằng hướng dẫn trong README.
- Có cấu hình mẫu .env.example, không commit secrets.
- Có health check và lưu trữ dữ liệu cần thiết.
- Chạy được test trong quy trình CI.
- Có thể triển khai API lên một máy chủ hoặc nền tảng cloud nếu có điều kiện.

Không bắt buộc deploy model lên cloud. Bạn có thể giữ Ollama chạy local trong giai đoạn đầu, miễn là tách lớp model provider để sau này thay đổi được.

- Xây dựng API: https://fastapi.tiangolo.com/tutorial/
- Kiểm thử API: https://fastapi.tiangolo.com/tutorial/testing/
- Đánh giá AI: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- Đo tải: https://docs.locust.io/en/stable/
- Giám sát OpenTelemetry: https://opentelemetry.io/docs/languages/python/
- Prometheus: https://prometheus.io/docs/introduction/overview/
- Grafana: https://grafana.com/docs/
- Đóng gói Docker: https://docs.docker.com/guides/python/
- Docker Compose: https://docs.docker.com/compose/
- Pytest: https://docs.pytest.org/en/stable/getting-started.html