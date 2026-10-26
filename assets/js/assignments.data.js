// Nội dung panel cho tab E1–E3 và A2 (tab A1 dùng a1-plan.data.js).
// Key trùng với id của thẻ tab trong index.html.
window.ASSIGNMENT_PANELS = {
  "e1-e3": {
    code: "E1–E3",
    label: "BÀI TẬP · NỀN TẢNG",
    deadline: "02/11 · 09/11/2026",
    title: "Nền tảng phân loại ảnh",
    lead: "Ba bài tập đi từ mô hình tuyến tính và CNN đến cơ chế attention và mạng hồi quy. Cả ba dùng chung một báo cáo PDF, cập nhật theo từng mốc nộp.",
    deliverables: {
      title: "Sản phẩm cần nộp",
      items: [
        "Code cho từng bài E1, E2, E3",
        "Báo cáo PDF chung E1–E3, cập nhật theo mốc",
        "Demo kết quả",
      ],
    },
    milestones: [
      {
        code: "E1",
        deadline: "02/11/2026",
        title: "Softmax, MLP và CNN",
        topics: ["Softmax", "MLP", "CNN"],
      },
      {
        code: "E2",
        deadline: "02/11/2026",
        title: "Attention và tokenization",
        topics: ["Multi-Head Self-Attention", "Tokenization"],
      },
      {
        code: "E3",
        deadline: "09/11/2026",
        title: "LSTM và GRU trên chuỗi ảnh",
        topics: ["LSTM", "GRU", "Chuỗi ảnh"],
      },
    ],
    resources: [
      ["Báo cáo chung E1–E3", "Sẽ cập nhật"],
      ["Demo", "Sẽ cập nhật"],
    ],
  },
  a2: {
    code: "A2",
    label: "BÀI TẬP LỚN · CHỦ ĐỀ TỰ CHỌN",
    deadline: "02/11 · 16/11/2026",
    title: "Chủ đề tự chọn & phân tích paper",
    lead: "Xây dựng pipeline thị giác máy tính cho một chủ đề tự chọn, so sánh các phương pháp và phân tích một bài báo liên quan.",
    deliverables: {
      title: "Sản phẩm cần nộp",
      items: [
        "Pipeline và mã nguồn",
        "So sánh phương pháp",
        "Phân tích paper",
        "Báo cáo và video demo",
      ],
    },
    milestones: [
      {
        code: "A2.1",
        deadline: "02/11/2026",
        title: "Chủ đề và kế hoạch",
        topics: ["Chủ đề", "Paper", "Dữ liệu", "Kế hoạch pipeline"],
      },
      {
        code: "A2.2",
        deadline: "16/11/2026",
        title: "Hoàn thiện bài tập lớn",
        topics: [
          "Pipeline",
          "So sánh phương pháp",
          "Phân tích paper",
          "Báo cáo",
          "Video demo",
        ],
      },
    ],
    resources: [
      ["Báo cáo", "Sẽ cập nhật"],
      ["Paper & demo", "Sẽ cập nhật"],
    ],
  },
};
