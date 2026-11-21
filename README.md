# Machine Learning PyTorch

## Mô tả dự án

Project thực hành các bài toán học máy bằng PyTorch. Mã nguồn bao gồm mô hình,
dataset, hàm loss, vòng lặp huấn luyện và bộ kiểm tra tự động.

Dự án thực hiện các bài toán sau:

- **Perceptron:** phân loại nhị phân bằng cập nhật trọng số.
- **Regression:** xấp xỉ hàm `sin(x)` trong khoảng `[-2π, 2π]`.
- **Digit Classification:** phân loại chữ số MNIST thành 10 lớp.
- **Language ID:** nhận diện ngôn ngữ từ chuỗi ký tự.
- **Digit Convolution:** thực hiện convolution 3x3 rồi phân loại chữ số MNIST.
- **Causal Attention:** triển khai self-attention có mask nhân quả.

## Cấu trúc project

```text
.
├── models.py          # Các mô hình neural network và attention
├── train.py           # Vòng lặp huấn luyện
├── losses.py          # Các hàm tính loss
├── backend.py         # Dataset và hỗ trợ trực quan hóa
├── autograder.py      # Bộ kiểm tra tự động
├── gpt_model.py       # Kiến trúc Character-GPT
├── test.py            # Kiểm tra PyTorch, NumPy và Matplotlib
├── requirements.txt   # Danh sách thư viện Python
└── data/
    ├── mnist.npz
    └── lang_id.npz
```

## Yêu cầu

- Python 3
- PyTorch
- NumPy
- Matplotlib

## Hướng dẫn chạy lại chương trình

Chạy toàn bộ lệnh sau tại thư mục gốc của project.

### 1. Tạo môi trường ảo

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Cài đặt thư viện

Cài toàn bộ thư viện từ file requirements:

```powershell
python -m pip install -r requirements.txt
```

### 3. Kiểm tra môi trường

```powershell
python test.py
```

Script này sẽ in phiên bản PyTorch và NumPy, đồng thời mở một đồ thị Matplotlib.

### 4. Chạy bộ bài tập bằng autograder

Chạy toàn bộ:

```powershell
python autograder.py --no-graphics
```

Chạy riêng một câu hỏi:

```powershell
python autograder.py --no-graphics -q q1
```

Các câu hỏi được hỗ trợ: `q1`, `q2`, `q3`, `q4`, `q5` và `q6`.

Kiểm tra thư viện đã cài:

```powershell
python autograder.py --check-dependencies
```