// Dữ liệu cho panel E1-E3. E1 và E2 đã xong, E3 đang triển khai.
window.E1_RESULTS = {
  code: "E1",
  label: "BÀI TẬP",
  deadline: "02/11/2026",
  pdf: "reports/E1-E3.pdf",
  repoUrl: "https://github.com/truong1501/SDH_HK261_CO5085/tree/main/E1",
  title: "Fashion-MNIST: softmax, MLP, CNN",
  lead: "Xây dựng và so sánh ba bộ phân loại ảnh trên cùng một tập Fashion-MNIST (28×28 xám, 10 lớp): softmax tuyến tính trên vector flatten, MLP nhiều tầng, và CNN có convolution/pooling. Cả ba dùng chung một split và một vòng lặp huấn luyện tự viết bằng PyTorch.",
  checklist: [
    "3 bộ phân loại: softmax, MLP, CNN",
    "Training loop tự viết: forward → loss → backward() → optimizer.step()",
    "Huấn luyện và đánh giá trên validation/test",
    "So sánh: accuracy, số tham số, đường cong loss/accuracy, confusion matrix, mẫu lỗi",
  ],
  stats: [
    { value: "91,45%", unit: "CNN", note: "Test accuracy, cao nhất trong 3 mô hình" },
    { value: "87,39%", unit: "MLP", note: "+8,3 điểm phần trăm so với softmax" },
    { value: "79,05%", unit: "Softmax", note: "Baseline tuyến tính, 7.850 tham số" },
    { value: "10.000", unit: "ảnh test", note: "Đánh giá một lần, cùng split cho cả 3 model" },
  ],
  models: [
    { name: "Softmax", params: "7.850", testAcc: "79,05%", f1: "0,788", time: "4,9 phút" },
    { name: "MLP", params: "235.146", testAcc: "87,39%", f1: "0,874", time: "5,0 phút" },
    { name: "CNN", params: "421.834", testAcc: "91,45%", f1: "0,914", time: "3,9 phút" },
  ],
  errorNote:
    "Cả ba mô hình đều khó nhất ở lớp Shirt, hay nhầm với T-shirt/top, Pullover và Coat, cùng một cụm áo phần thân trên khó phân biệt ở ảnh xám 28×28. Softmax chỉ nhận đúng 40,9% ảnh Shirt, MLP 65,3%, CNN 69,6%. Tăng capacity và thêm convolution giúp thu hẹp nhầm lẫn nhưng không loại bỏ hoàn toàn, vì giới hạn này đến từ chính dữ liệu chứ không phải từ mô hình.",
  setup: [
    "Adam, learning rate 1e-3",
    "Batch size 128",
    "Seed 42",
    "20 epoch (softmax, MLP) · 15 epoch (CNN)",
    "Google Colab · GPU Tesla T4",
  ],
  metrics: [
    "Test accuracy",
    "Macro-F1",
    "Số tham số",
    "Confusion matrix",
    "Đường cong loss/accuracy",
    "Mẫu dự đoán sai",
  ],
};

window.E2_RESULTS = {
  code: "E2",
  label: "BÀI TẬP",
  deadline: "02/11/2026",
  repoUrl: "https://github.com/truong1501/SDH_HK261_CO5085/tree/main/E2",
  title: "Transformer: MSA tự viết và 4 cách tokenize",
  lead: "Hiện thực Multi-Head Self-Attention (MSA) từ đầu và bằng nn.MultiheadAttention của PyTorch để đối chiếu, rồi thử bốn cách biến ảnh Fashion-MNIST thành chuỗi token đưa vào cùng một encoder Transformer nông (4 khối, 4 head, chiều 64).",
  checklist: [
    "MSA tự viết (ScratchMSA): chỉ dùng Linear, einsum, softmax và ghép head",
    "MSA bằng PyTorch (nn.MultiheadAttention) để đối chiếu",
    "Bốn cách tokenize: patch 4x4, patch 7x7 chồng lấn, theo hàng, CNN-stem",
    "So sánh MSA tự viết với PyTorch, và so sánh các cách tokenize với nhau",
  ],
  stats: [
    { value: "92,25%", unit: "CNN-stem", note: "Test accuracy cao nhất trong 5 cấu hình, vượt cả CNN của E1" },
    { value: "0,0", unit: "Chênh lệch MSA", note: "Tự viết và PyTorch cho cùng output, attention, gradient" },
    { value: "88,50%", unit: "Patch 4x4", note: "Cách tokenize yếu nhất, token là pixel thô chưa qua xử lý" },
    { value: "5", unit: "cấu hình", note: "4 cách tokenize, cộng 1 bản MSA PyTorch để đối chiếu" },
  ],
  models: [
    { name: "Patch 4x4", attn: "MSA tự viết", params: "139.018", testAcc: "88,50%", f1: "0,884", time: "4,3 phút" },
    { name: "Patch 4x4", attn: "MSA PyTorch", params: "139.018", testAcc: "88,54%", f1: "0,884", time: "4,5 phút" },
    { name: "Patch 7x7 chồng lấn", attn: "MSA tự viết", params: "142.090", testAcc: "89,54%", f1: "0,895", time: "4,4 phút" },
    { name: "Theo hàng", attn: "MSA tự viết", params: "138.442", testAcc: "89,14%", f1: "0,891", time: "4,3 phút" },
    { name: "CNN-stem", attn: "MSA tự viết", params: "156.938", testAcc: "92,25%", f1: "0,922", time: "4,3 phút" },
  ],
  msaNote:
    "MSA tự viết và bản PyTorch cho đúng cùng một kết quả khi copy cùng trọng số: chênh lệch tối đa giữa hai bản là 0,0 ở output, ma trận attention và gradient, kiểm tra trên cả hai cấu hình (chiều 64, 4 head) và (chiều 32, 8 head). Accuracy lúc train cũng gần như nhau, 88,50% so với 88,54%, chênh lệch nhỏ đó đến từ nhiễu ngẫu nhiên của quá trình train chứ không phải do cài đặt sai.",
  errorNote:
    "CNN-stem là cách tokenize tốt nhất, đạt 92,25%, vượt cả CNN thuần của E1 (91,45%) dù ít tham số hơn nhiều (156.938 so với 421.834). Ảnh đã qua hai lớp convolution để trích đặc trưng cục bộ trước khi đưa vào attention, nên attention chỉ cần học quan hệ giữa các vùng đã được xử lý sẵn thay vì từ pixel thô. Patch 4x4 thuần là cách tokenize yếu nhất, 88,50%, còn kém hơn MLP của E1 (87,39%) dù có attention, vì mỗi token chỉ là một ô pixel chưa qua xử lý nào. Patch chồng lấn và chia theo hàng nhỉnh hơn patch 4x4 một chút nhưng vẫn chưa bằng CNN-stem hay CNN thuần của E1.",
  setup: [
    "AdamW, lr 1e-3, weight decay 0,05",
    "Warmup 2 epoch rồi cosine decay",
    "Batch 128, seed 42, 30 epoch",
    "Encoder: chiều 64, 4 khối, 4 head, FFN rộng gấp đôi, dropout 0,1",
    "Google Colab · GPU Tesla T4",
  ],
  metrics: [
    "Test accuracy",
    "Macro-F1",
    "Số tham số",
    "Kiểm chứng MSA (msa_check)",
    "Attention map",
    "Đường cong loss/accuracy",
  ],
  pending:
    "E3 (LSTM, GRU trên chuỗi ảnh) dùng chung dataset này và đang triển khai, hạn nộp 09/11/2026.",
};
