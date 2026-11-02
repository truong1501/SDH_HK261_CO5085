// Render panel E1-E3 bang ket qua that: phan E1 va E2 da xong, E3 dang cho.
// Du lieu lay tu e1-plan.data.js (window.E1_RESULTS, window.E2_RESULTS).
(function () {
  "use strict";

  var root = document.querySelector('[data-plan="e1"]');
  var e1 = window.E1_RESULTS;
  var e2 = window.E2_RESULTS;
  if (!root || !e1 || !e2) return;

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch];
    });
  }

  function list(items, render) {
    return items.map(render).join("");
  }

  function block(index, title, body, extraClass) {
    return (
      '<div class="plan-block' +
      (extraClass ? " " + extraClass : "") +
      '">' +
      '<p class="plan-block-label"><span>' +
      index +
      "</span>" +
      esc(title) +
      "</p>" +
      body +
      "</div>"
    );
  }

  function renderHead() {
    return (
      '<div class="plan-head">' +
      '<div class="plan-tag"><span class="assignment-code">E1 + E2</span>' +
      '<span class="assignment-type">BÀI TẬP · E1, E2 ĐÃ XONG</span></div>' +
      '<div class="plan-meta"><span>Hạn nộp <strong>02/11/2026</strong></span>' +
      '<a class="plan-pdf" href="' +
      encodeURI(e1.pdf) +
      '" target="_blank" rel="noopener">Báo cáo PDF <span aria-hidden="true">↗</span></a>' +
      '<a class="plan-pdf" href="' +
      esc(e1.repoUrl) +
      '" target="_blank" rel="noopener">Mã nguồn E1 <span aria-hidden="true">↗</span></a>' +
      '<a class="plan-pdf" href="' +
      esc(e2.repoUrl) +
      '" target="_blank" rel="noopener">Mã nguồn E2 <span aria-hidden="true">↗</span></a>' +
      "</div>" +
      "</div>" +
      '<div class="plan-intro">' +
      '<div><h3 id="e1-plan-title">Fashion-MNIST: từ softmax đến Transformer</h3>' +
      '<p class="plan-lead">Cùng một tập Fashion-MNIST và cùng một split cho cả bài: ' +
      "E1 so sánh softmax, MLP và CNN; E2 hiện thực Multi-Head Self-Attention và thử " +
      "bốn cách biến ảnh thành chuỗi token cho Transformer.</p></div>" +
      '<div class="plan-questions"><p>Yêu cầu đã hoàn thành</p>' +
      '<ul class="plan-checklist">' +
      list(
        [
          "E1: 3 bộ phân loại softmax, MLP, CNN, có training loop tự viết",
          "E2: MSA tự viết và bằng PyTorch, đối chiếu khớp tuyệt đối",
          "E2: 4 cách tokenize ảnh, so sánh accuracy và chi phí",
          "Cả hai: bảng số liệu, đường cong loss/accuracy, phân tích lỗi",
        ],
        function (c) {
          return "<li>" + esc(c) + "</li>";
        },
      ) +
      "</ul></div>" +
      "</div>"
    );
  }

  function renderStats() {
    var stats = [
      { value: "92,25%", unit: "CNN-stem (E2)", note: "Test accuracy cao nhất trong toàn bộ E1 và E2" },
      { value: "91,45%", unit: "CNN (E1)", note: "Cao nhất trong 3 bộ phân loại không dùng attention" },
      { value: "0,0", unit: "Chênh lệch MSA", note: "MSA tự viết khớp tuyệt đối với bản PyTorch" },
      { value: "10.000", unit: "ảnh test", note: "Dùng chung cho mọi mô hình của E1 và E2" },
    ];
    return (
      '<div class="plan-stats">' +
      list(stats, function (s) {
        return (
          '<div class="plan-stat"><strong>' +
          esc(s.value) +
          "</strong><span>" +
          esc(s.unit) +
          "</span><small>" +
          esc(s.note) +
          "</small></div>"
        );
      }) +
      "</div>"
    );
  }

  function modelCard(m) {
    var chips = '<span>' + esc(m.params) + ' tham số</span>';
    if (m.attn) chips += '<span>' + esc(m.attn) + '</span>';
    return (
      '<li class="mode-card">' +
      '<div class="mode-top"><span class="mode-runs">' + chips + '</span></div>' +
      "<h4>" +
      esc(m.name) +
      '</h4><p class="mode-init">Test accuracy ' +
      esc(m.testAcc) +
      "</p>" +
      "<dl><div><dt>Macro-F1</dt><dd>" +
      esc(m.f1) +
      "</dd></div>" +
      "<div><dt>Thời gian train</dt><dd>" +
      esc(m.time) +
      "</dd></div></dl>" +
      "</li>"
    );
  }

  function renderE1() {
    var cards = '<ol class="mode-grid">' + list(e1.models, modelCard) + "</ol>";
    return (
      block("A", "E1: kết quả ba mô hình trên tập test", cards, "plan-block-wide") +
      block(
        "B",
        "E1: phân tích lỗi",
        '<p class="plan-note" style="margin-top:0">' + esc(e1.errorNote) + "</p>",
        "plan-block-wide",
      )
    );
  }

  function renderE2() {
    var cards = '<ol class="mode-grid milestone-grid">' + list(e2.models, modelCard) + "</ol>";
    return (
      block("C", "E2: kết quả MSA và tokenize trên tập test", cards, "plan-block-wide") +
      block(
        "D",
        "E2: kiểm chứng MSA tự viết",
        '<p class="plan-note" style="margin-top:0">' + esc(e2.msaNote) + "</p>",
        "plan-block-wide",
      ) +
      block(
        "E",
        "E2: phân tích tokenize",
        '<p class="plan-note" style="margin-top:0">' + esc(e2.errorNote) + "</p>",
        "plan-block-wide",
      )
    );
  }

  function chips(items) {
    return (
      '<ul class="plan-chips">' +
      list(items, function (c) {
        return "<li>" + esc(c) + "</li>";
      }) +
      "</ul>"
    );
  }

  function renderFooter() {
    return (
      '<div class="plan-footer">' +
      block("F", "E1: cấu hình huấn luyện", chips(e1.setup)) +
      block("G", "E1: kết quả & metric ghi lại", chips(e1.metrics)) +
      "</div>" +
      '<div class="plan-footer">' +
      block("H", "E2: cấu hình huấn luyện", chips(e2.setup)) +
      block("I", "E2: kết quả & metric ghi lại", chips(e2.metrics)) +
      "</div>"
    );
  }

  root.innerHTML =
    renderHead() +
    renderStats() +
    renderE1() +
    renderE2() +
    renderFooter() +
    '<p class="plan-note">' +
    esc(e2.pending) +
    "</p>";
  root.classList.add("is-ready");
})();
