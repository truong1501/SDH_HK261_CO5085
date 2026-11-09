// Dữ liệu cho 3 panel riêng biệt E1, E2, E3 (xem e1-plan.js, e2-plan.js, e3-plan.js). Cả ba bài đã xong.
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
  pdf: "reports/E1-E3.pdf",
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
};

window.E3_RESULTS = {
  code: "E3",
  label: "BÀI TẬP",
  deadline: "09/11/2026",
  pdf: "reports/E1-E3.pdf",
  repoUrl: "https://github.com/truong1501/SDH_HK261_CO5085/tree/main/E3",
  title: "LSTM và GRU trên chuỗi ảnh",
  lead: "Mô hình hóa ảnh Fashion-MNIST như dữ liệu chuỗi: thử hai cách biến ảnh thành chuỗi (theo hàng, theo patch 4x4) cho cả LSTM và GRU, rồi so với baseline không hồi quy MLP/CNN đã có từ E1, trên cùng split.",
  checklist: [
    "LSTM và GRU, mỗi kiến trúc chạy trên 2 cách biến chuỗi (theo hàng, theo patch)",
    "Baseline MLP/CNN đánh giá lại từ checkpoint E1, không train lại",
    "So sánh: test accuracy, số tham số, thời gian train",
  ],
  stats: [
    { value: "88,95%", unit: "GRU · theo hàng", note: "Test accuracy cao nhất trong 4 cấu hình, vượt MLP" },
    { value: "86,75%", unit: "GRU · patch", note: "Cấu hình yếu nhất, thấp hơn cả MLP" },
    { value: "25%", unit: "GRU nhẹ hơn LSTM", note: "Cùng hidden size, ít tham số hơn nhờ 3 cổng thay vì 4" },
    { value: "91,45%", unit: "CNN (E1)", note: "Không cấu hình LSTM/GRU nào vượt qua được" },
  ],
  models: [
    { name: "LSTM", attn: "theo hàng", params: "82.186", testAcc: "88,55%", time: "5,2 phút" },
    { name: "GRU", attn: "theo hàng", params: "61.962", testAcc: "88,95%", time: "4,9 phút" },
    { name: "LSTM", attn: "patch 4x4", params: "76.042", testAcc: "86,89%", time: "4,9 phút" },
    { name: "GRU", attn: "patch 4x4", params: "57.354", testAcc: "86,75%", time: "4,9 phút" },
  ],
  errorNote:
    "Theo hàng luôn tốt hơn patch ở cả hai kiến trúc (chênh 1,66 đến 2,20 điểm phần trăm), vì mỗi bước của chuỗi theo hàng có ngữ cảnh một hàng đầy đủ (28 chiều), còn mỗi bước của patch chỉ có một ô cục bộ (16 chiều) nên mô hình phải tự ghép nối ngữ cảnh không gian qua nhiều bước hồi quy hơn. LSTM và GRU không đồng nhất: GRU tốt hơn ở chuỗi theo hàng, LSTM nhỉnh hơn một chút ở patch, chênh lệch nhỏ và chỉ chạy một seed nên chưa đủ để kết luận kiến trúc nào luôn tốt hơn. Hai cấu hình theo hàng vượt MLP (87,39%) dù ít tham số hơn nhiều lần, nhưng không cấu hình nào vượt CNN (91,45%): hồi quy tuần tự một chiều tích lũy ngữ cảnh tốt hơn MLP phẳng, nhưng chưa thay được inductive bias không gian hai chiều của convolution.",
  setup: [
    "Adam, learning rate 1e-3",
    "Batch size 128, seed 42, 20 epoch",
    "Google Colab · GPU Tesla T4",
  ],
  metrics: [
    "Test accuracy",
    "Số tham số",
    "Thời gian train",
    "Đường cong loss/accuracy",
  ],
};
