// Render panel E1 bằng kết quả thật, dữ liệu lấy từ e1-plan.data.js.
(function () {
  "use strict";

  var root = document.querySelector('[data-plan="e1"]');
  var plan = window.E1_RESULTS;
  if (!root || !plan) return;

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
      '<div class="plan-tag"><span class="assignment-code">' +
      esc(plan.code) +
      "</span>" +
      '<span class="assignment-type">' +
      esc(plan.label) +
      "</span></div>" +
      '<div class="plan-meta"><span>Hạn nộp <strong>' +
      esc(plan.deadline) +
      "</strong></span>" +
      '<a class="plan-pdf" href="' +
      encodeURI(plan.pdf) +
      '" target="_blank" rel="noopener">Báo cáo PDF <span aria-hidden="true">↗</span></a>' +
      '<a class="plan-pdf" href="' +
      esc(plan.repoUrl) +
      '" target="_blank" rel="noopener">Mã nguồn E1 <span aria-hidden="true">↗</span></a>' +
      "</div>" +
      "</div>" +
      '<div class="plan-intro">' +
      '<div><h3 id="e1-plan-title">' +
      esc(plan.title) +
      '</h3><p class="plan-lead">' +
      esc(plan.lead) +
      "</p></div>" +
      '<div class="plan-questions"><p>Yêu cầu đã hoàn thành</p>' +
      '<ul class="plan-checklist">' +
      list(plan.checklist, function (c) {
        return "<li>" + esc(c) + "</li>";
      }) +
      "</ul></div>" +
      "</div>"
    );
  }

  function renderStats() {
    return (
      '<div class="plan-stats">' +
      list(plan.stats, function (s) {
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

  function renderModels() {
    var cards =
      '<ol class="mode-grid">' +
      list(plan.models, function (m) {
        return (
          '<li class="mode-card">' +
          '<div class="mode-top"><span class="mode-runs"><span>' +
          esc(m.params) +
          ' tham số</span></span></div>' +
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
      }) +
      "</ol>";
    return block("A", "Kết quả ba mô hình trên tập test", cards, "plan-block-wide");
  }

  function renderErrorAnalysis() {
    return block(
      "B",
      "Phân tích lỗi",
      '<p class="plan-note" style="margin-top:0">' + esc(plan.errorNote) + "</p>",
      "plan-block-wide",
    );
  }

  function renderFooter() {
    function chips(items) {
      return (
        '<ul class="plan-chips">' +
        list(items, function (c) {
          return "<li>" + esc(c) + "</li>";
        }) +
        "</ul>"
      );
    }
    return (
      '<div class="plan-footer">' +
      block("C", "Cấu hình huấn luyện", chips(plan.setup)) +
      block("D", "Kết quả & metric ghi lại", chips(plan.metrics)) +
      "</div>"
    );
  }

  root.innerHTML =
    renderHead() +
    renderStats() +
    renderModels() +
    renderErrorAnalysis() +
    renderFooter() +
    '<p class="plan-note">' +
    esc(plan.pending) +
    "</p>";
  root.classList.add("is-ready");
})();
