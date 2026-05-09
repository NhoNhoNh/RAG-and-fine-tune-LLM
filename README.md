# 🚦 RAG & Fine-tune LLM — Hỏi đáp Luật Giao thông Đường bộ Việt Nam 2008

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)](https://www.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)](https://www.trychroma.com/)

Dự án xây dựng hệ thống **hỏi đáp thông minh** về Luật Giao thông Đường bộ Việt Nam (Luật số 23/2008/QH12) sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp với **Fine-tuning LLM**. Toàn bộ pipeline chạy trên **Google Colab** với GPU T4 miễn phí.

---

## 📋 Mục lục

- [Tổng quan dự án](#-tổng-quan-dự-án)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc Repository](#-cấu-trúc-repository)
- [Yêu cầu](#-yêu-cầu)
- [Hướng dẫn chạy trên Google Colab](#-hướng-dẫn-chạy-trên-google-colab)
- [Chi tiết từng Notebook](#-chi-tiết-từng-notebook)
- [Thứ tự thực thi](#-thứ-tự-thực-thi)
- [Cấu trúc dữ liệu trên Google Drive](#-cấu-trúc-dữ-liệu-trên-google-drive)
- [Mô hình sử dụng](#-mô-hình-sử-dụng)
- [Kết quả đánh giá](#-kết-quả-đánh-giá)
- [Thành viên nhóm](#-thành-viên-nhóm)

---

## 🎯 Tổng quan dự án

Hệ thống cho phép người dùng đặt câu hỏi bằng tiếng Việt về Luật Giao thông, ví dụ:
- *"Vượt đèn đỏ bị phạt bao nhiêu tiền?"*
- *"Nồng độ cồn tối đa cho phép khi lái xe là bao nhiêu?"*
- *"Điều kiện để được cấp bằng lái xe hạng B2 là gì?"*

Hệ thống sẽ tự động **tìm kiếm điều khoản liên quan** trong luật và **tạo câu trả lời chính xác** dựa trên ngữ cảnh pháp lý thực tế.

*Demo dự án:* https://youtu.be/vNufIKx0sno

### Điểm nổi bật
- ✅ **Parent-Child Chunking**: Chia văn bản luật theo cấp Điều/Khoản để tăng độ chính xác retrieval
- ✅ **Metadata-aware Search**: Lọc theo loại phương tiện, hành vi vi phạm, mức phạt
- ✅ **Fine-tuned LLM**: Model Qwen2.5-3B được fine-tune với dữ liệu hỏi đáp luật giao thông
- ✅ **Gradio Demo**: Giao diện web tương tác chạy ngay trên Colab
- ✅ **Offline-first**: Toàn bộ LLM và embedding chạy locally, không cần API bên ngoài

---

## 🏗️ Kiến trúc hệ thống

```
PDF Luật Giao thông 2008
        │
        ▼
┌─────────────────────┐
│  1. preprocessing   │  Trích xuất PDF → Làm sạch → Chunking
│     .ipynb          │  Parent (Điều) + Child (Khoản) + Metadata
└────────┬────────────┘
         │ parents_traffic_law.json
         │ children_traffic_law.json
         ▼
┌─────────────────────┐
│  2. embedding       │  Embedding: paraphrase-multilingual-MiniLM-L12-v2
│     .ipynb          │  Vector Store: ChromaDB (lưu trên Google Drive)
└────────┬────────────┘
         │ vector_db_parent_child/
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  3. finetune_LLM    │     │  4. evaluation       │
│     .ipynb          │     │     .ipynb           │
│  Qwen2.5-3B + LoRA  │     │  BLEU / ROUGE / F1   │
└────────┬────────────┘     └──────────────────────┘
         │ adapter weights
         ▼
┌─────────────────────┐
│  5. demo            │  Gradio UI: RAG Pipeline + Fine-tuned LLM
│     .ipynb          │  Custom Parent-Child Retriever + Qwen-3B
└─────────────────────┘
```

---

## 📁 Cấu trúc Repository

```
RAG-and-fine-tune-LLM/
├── preprocessing.ipynb     # Bước 1: Tiền xử lý PDF luật giao thông
├── embedding.ipynb         # Bước 2: Tạo vector embeddings & ChromaDB
├── finetune_LLM.ipynb      # Bước 3: Fine-tune Qwen2.5-3B với LoRA
├── evaluation.ipynb        # Bước 4: Đánh giá RAG pipeline & fine-tuned model
├── demo.ipynb              # Bước 5: Giao diện demo Gradio
└── README.md
```

---

## ✅ Yêu cầu

### Tài khoản & Công cụ
| Yêu cầu | Mô tả |
|---------|-------|
| **Google Account** | Để sử dụng Google Colab và Google Drive |
| **Google Colab** | GPU T4 (miễn phí) hoặc A100 (Colab Pro) |
| **Google Drive** | Tối thiểu **5GB** trống để lưu model và vector DB |
| **HuggingFace Token** | Cần thiết để tải model Qwen2.5 — [Tạo tại đây](https://huggingface.co/settings/tokens) |

### File dữ liệu cần chuẩn bị
- `23_2008_QH12_82203.pdf` — File PDF Luật Giao thông Đường bộ 2008 (tải về từ [cơ quan nhà nước](https://vbpl.vn))

---

## 🚀 Hướng dẫn chạy trên Google Colab

### Bước 1: Clone repository về Google Drive

Mở một notebook mới trên Colab, chạy lệnh sau:

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone repo vào Drive
%cd /content/drive/MyDrive
!git clone https://github.com/NhoNhoNh/RAG-and-fine-tune-LLM.git NLP
%cd NLP
```

### Bước 2: Tạo cấu trúc thư mục trên Drive

```python
import os

BASE_PATH = "/content/drive/MyDrive/NLP"
os.makedirs(f"{BASE_PATH}/data/raw", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data/processed", exist_ok=True)
os.makedirs(f"{BASE_PATH}/data/datasets", exist_ok=True)
os.makedirs(f"{BASE_PATH}/vector_db_parent_child", exist_ok=True)
os.makedirs(f"{BASE_PATH}/models/qwen_finetuned", exist_ok=True)

print("✅ Tạo cấu trúc thư mục thành công!")
```

### Bước 3: Upload file PDF luật lên Drive

Upload file `23_2008_QH12_82203.pdf` vào thư mục:
```
MyDrive/NLP/data/raw/23_2008_QH12_82203.pdf
```

### Bước 4: Cấu hình HuggingFace Token

Trong mỗi notebook có ô cần điền token:
```python
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"   # ← Thay bằng token của bạn
```

> ⚠️ **Bảo mật**: Không commit token thật lên GitHub. Dùng **Colab Secrets** thay thế:
> - Vào menu **🔑 Secrets** (biểu tượng chìa khóa bên trái Colab)
> - Thêm secret tên `HF_TOKEN` với giá trị là token của bạn
> - Sau đó trong notebook dùng:
> ```python
> from google.colab import userdata
> os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
> ```

### Bước 5: Bật GPU

Vào **Runtime → Change runtime type → T4 GPU** → Save

### Bước 6: Chạy theo thứ tự

Chạy lần lượt các notebook theo thứ tự dưới đây ↓

---

## 📓 Chi tiết từng Notebook

### 1️⃣ `preprocessing.ipynb` — Tiền xử lý dữ liệu

**Mục đích**: Biến đổi file PDF Luật Giao thông thành dữ liệu có cấu trúc.

**Các bước thực hiện**:
1. Cài đặt `pdfplumber`, kết nối Google Drive
2. Trích xuất toàn bộ văn bản từ PDF
3. Làm sạch văn bản: chuẩn hóa Unicode, loại bỏ header/footer, ký tự nhiễu
4. **Parent-Child Chunking**:
   - **Parent** = mỗi Điều luật (176 documents)
   - **Child** = mỗi Khoản trong Điều (707 chunks)
5. Gán metadata: loại điều khoản, phương tiện, hành vi vi phạm, mức phạt
6. Lưu ra `parents_traffic_law.json` và `children_traffic_law.json`

**Thư viện chính**: `pdfplumber`, `re`, `uuid`, `json`

**Output**:
```
data/datasets/parents_traffic_law.json   # 176 Parent Documents
data/datasets/children_traffic_law.json  # 707 Child Chunks
```

---

### 2️⃣ `embedding.ipynb` — Tạo Vector Embeddings

**Mục đích**: Tạo vector embeddings và lưu vào ChromaDB để phục vụ tìm kiếm ngữ nghĩa.

**Các bước thực hiện**:
1. Load dữ liệu JSON từ bước preprocessing
2. Khởi tạo model embedding: `paraphrase-multilingual-MiniLM-L12-v2`
3. Tạo **Parent-Child Retriever**:
   - Lập index Child Chunks vào ChromaDB (tìm kiếm nhanh)
   - Lưu Parent Documents vào InMemoryStore (trả về context đầy đủ)
4. Lưu vector DB xuống Google Drive

**Thư viện chính**: `langchain`, `chromadb`, `sentence-transformers`, `huggingface-hub`

**Output**:
```
vector_db_parent_child/   # ChromaDB với 707 child chunks đã được index
```

**Lưu ý**: Lần đầu chạy sẽ mất ~5-10 phút để index. Các lần sau sẽ load từ disk.

---

### 3️⃣ `finetune_LLM.ipynb` — Fine-tune Mô hình Ngôn ngữ

**Mục đích**: Fine-tune model Qwen2.5-3B-Instruct để chuyên biệt hóa cho lĩnh vực luật giao thông Việt Nam.

**Các bước thực hiện**:
1. Cài đặt `unsloth`, `transformers`, `trl`, `bitsandbytes`
2. Load Qwen2.5-3B-Instruct với **4-bit quantization**
3. Áp dụng **LoRA** (Low-Rank Adaptation):
   - `r = 16`, `lora_alpha = 16`
   - Target modules: attention layers
4. Chuẩn bị dataset hỏi đáp dạng instruction-tuning
5. Training với **SFTTrainer** (~313 mẫu train, ~50 mẫu validation)
6. Lưu adapter weights về Google Drive

**Thư viện chính**: `unsloth`, `transformers`, `trl`, `bitsandbytes`, `peft`

**Yêu cầu GPU**: T4 (16GB VRAM) — **bắt buộc bật GPU**

**Output**:
```
models/qwen_finetuned/   # LoRA adapter weights
```

**Thời gian chạy**: ~30-60 phút (tùy GPU)

---

### 4️⃣ `evaluation.ipynb` — Đánh giá Hệ thống

**Mục đích**: So sánh hiệu suất giữa RAG pipeline và fine-tuned LLM trên bộ câu hỏi kiểm thử.

**Các bước thực hiện**:
1. Load bộ câu hỏi kiểm thử về luật giao thông
2. Chạy inference với:
   - **Base Qwen2.5-3B** (không fine-tune, không RAG)
   - **RAG Pipeline** (có retriever, không fine-tune)
   - **Fine-tuned Qwen2.5-3B + RAG** (kết hợp đầy đủ)
3. Đánh giá theo các metrics:
   - **BLEU Score** — độ chính xác từ vựng
   - **ROUGE Score** — độ bao phủ thông tin
   - **F1 Score** — cân bằng precision/recall

**Thư viện chính**: `nltk`, `rouge-score`, `transformers`, `langchain`

---

### 5️⃣ `demo.ipynb` — Giao diện Demo

**Mục đích**: Chạy giao diện web tương tác để trải nghiệm hệ thống hỏi đáp.

**Các bước thực hiện**:
1. Load ChromaDB từ Google Drive
2. Khởi động Qwen2.5-3B-Instruct (hoặc fine-tuned version)
3. Xây dựng RAG pipeline hoàn chỉnh với LangChain
4. Khởi chạy giao diện **Gradio**

**Thư viện chính**: `gradio`, `langchain`, `transformers`, `chromadb`

**Sử dụng**:
- Nhập câu hỏi vào ô chat
- Nhấn **Submit** hoặc **Enter**
- Hệ thống sẽ hiển thị câu trả lời kèm theo nguồn tham chiếu từ luật

---

## 📊 Thứ tự thực thi

```
preprocessing.ipynb
       ↓
embedding.ipynb
       ↓
finetune_LLM.ipynb    ←── (Tùy chọn, có thể bỏ qua nếu chỉ cần RAG)
       ↓
evaluation.ipynb      ←── (Tùy chọn, để xem kết quả đánh giá)
       ↓
demo.ipynb            ←── Chạy để trải nghiệm hệ thống
```

> 💡 **Tip**: Nếu chỉ muốn trải nghiệm nhanh, chạy `preprocessing.ipynb` → `embedding.ipynb` → `demo.ipynb`.

---

## 🗂️ Cấu trúc dữ liệu trên Google Drive

Sau khi chạy đầy đủ, thư mục `MyDrive/NLP/` sẽ có cấu trúc:

```
MyDrive/NLP/
├── data/
│   ├── raw/
│   │   ├── 23_2008_QH12_82203.pdf        # File PDF gốc (bạn tự upload)
│   │   └── raw_content.txt               # Văn bản thô trích xuất từ PDF
│   ├── processed/
│   │   └── clean_content.txt             # Văn bản sau khi làm sạch
│   └── datasets/
│       ├── parents_traffic_law.json       # 176 Parent Documents (cấp Điều)
│       └── children_traffic_law.json     # 707 Child Chunks (cấp Khoản)
├── vector_db_parent_child/               # ChromaDB vector store
│   └── ...
└── models/
    └── qwen_finetuned/                   # Fine-tuned LoRA adapter weights
        └── ...
```

---

## 🤖 Mô hình sử dụng

| Mô hình | Mục đích | Kích thước |
|---------|---------|-----------|
| [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | LLM sinh câu trả lời | ~6GB (4-bit) |
| [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Embedding đa ngôn ngữ | ~470MB |

---

## 📈 Kết quả đánh giá

| Hệ thống | BLEU | ROUGE-L | F1 |
|----------|------|---------|-----|
| Base LLM (Qwen2.5-3B) | thấp | thấp | thấp |
| RAG Pipeline | trung bình | cao | cao |
| Fine-tuned + RAG | cao nhất | cao nhất | cao nhất |

> Chi tiết số liệu xem trong `evaluation.ipynb`.

---

## ⚠️ Lưu ý quan trọng

1. **Token bảo mật**: Không nhúng HuggingFace token trực tiếp vào notebook. Dùng Colab Secrets hoặc biến môi trường.
2. **GPU bắt buộc**: `finetune_LLM.ipynb` và `demo.ipynb` **cần GPU T4** (bật tại Runtime → Change runtime type).
3. **Drive storage**: Đảm bảo Google Drive còn đủ dung lượng (~5GB).
4. **Thứ tự chạy**: Phải chạy theo đúng thứ tự preprocessing → embedding → finetune → demo.
5. **Session timeout**: Nếu Colab ngắt session, chỉ cần chạy lại từ notebook hiện tại (dữ liệu đã được lưu trên Drive).


---

## 📄 Giấy phép

Dự án được thực hiện phục vụ mục đích học thuật — Môn Xử lý Ngôn ngữ Tự nhiên (NLP), Học kỳ 2 năm học 2025-2026.

---

<div align="center">
  <i>Được xây dựng với ❤️ bằng LangChain, HuggingFace Transformers và ChromaDB</i>
</div>
