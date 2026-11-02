// Dữ liệu cho panel E1. Chỉ có E1, vì E2 và E3 vẫn đang triển khai, chưa đưa lên đây.
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
  pending:
    "E2 (MSA, tokenization) và E3 (LSTM, GRU) dùng chung dataset này và đang triển khai, hạn nộp lần lượt 02/11 và 09/11/2026.",
};
