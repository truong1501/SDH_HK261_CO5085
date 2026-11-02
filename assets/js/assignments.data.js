// Nội dung panel cho tab A2 (tab A1 dùng a1-plan.data.js, tab E1–E3 dùng e1-plan.data.js).
// Key trùng với id của thẻ tab trong index.html.
window.ASSIGNMENT_PANELS = {
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
