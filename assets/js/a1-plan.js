(function () {
  "use strict";

  var root = document.querySelector('[data-plan="a1"]');
  var plan = window.A1_PLAN;
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
      '" target="_blank" rel="noopener">Bản kế hoạch PDF <span aria-hidden="true">↗</span></a></div>' +
      "</div>" +
      '<div class="plan-intro">' +
      '<div><h3 id="a1-plan-title">' +
      esc(plan.title) +
      '</h3><p class="plan-lead">' +
      esc(plan.lead) +
      "</p></div>" +
      '<div class="plan-questions"><p>Câu hỏi cần trả lời</p><ol>' +
      list(plan.questions, function (q) {
        return "<li>" + esc(q) + "</li>";
      }) +
      "</ol></div>" +
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

  function renderData() {
    var split = plan.split;
    var bar =
      '<div class="split-bar" role="img" aria-label="Tỷ lệ train, validation, test">' +
      list(split.parts, function (p) {
        return (
          '<span class="split-' +
          p.key +
          '" style="flex-grow:' +
          p.ratio +
          '"></span>'
        );
      }) +
      "</div>";
    var legend =
      '<ul class="split-legend">' +
      list(split.parts, function (p) {
        return (
          '<li><span class="split-dot split-' +
          p.key +
          '" aria-hidden="true"></span>' +
          "<strong>" +
          esc(p.name) +
          '</strong><span class="split-count">' +
          esc(p.count) +
          "</span>" +
          "<small>" +
          p.perClass +
          " ảnh/lớp · " +
          esc(p.use) +
          "</small></li>"
        );
      }) +
      "</ul>";
    var pipeline =
      '<dl class="plan-pipeline">' +
      list(split.pipeline, function (row) {
        return (
          "<div><dt>" + esc(row[0]) + "</dt><dd>" + esc(row[1]) + "</dd></div>"
        );
      }) +
      "</dl>";
    return block(
      "A",
      "Dữ liệu & split",
      bar +
        legend +
        '<p class="plan-note">' +
        esc(split.note) +
        "</p>" +
        pipeline,
    );
  }

  function renderModels() {
    var models = plan.models;
    var main =
      '<div class="model-pair">' +
      list(models.main, function (m, i) {
        return (
          (i ? '<span class="model-vs" aria-hidden="true">vs</span>' : "") +
          '<div class="model-card"><span class="model-kind">' +
          esc(m.kind) +
          "</span>" +
          "<strong>" +
          esc(m.name) +
          "</strong>" +
          "<dl><div><dt>Tham số</dt><dd>" +
          esc(m.params) +
          "</dd></div>" +
          "<div><dt>GFLOPs</dt><dd>" +
          esc(m.gflops) +
          "</dd></div>" +
          "<div><dt>Top-1 IN-1k</dt><dd>" +
          esc(m.top1) +
          "</dd></div></dl>" +
          "<code>" +
          esc(m.timm) +
          "</code></div>"
        );
      }) +
      "</div>";
    var extra =
      '<div class="model-extra"><span>Cặp mở rộng · tùy chọn</span><p>' +
      list(models.extra, function (m, i) {
        return (
          (i ? '<em aria-hidden="true">vs</em>' : "") +
          "<strong>" +
          esc(m.name) +
          "</strong> <small>" +
          esc(m.kind) +
          " · " +
          esc(m.params) +
          "</small>"
        );
      }) +
      "</p></div>";
    return block(
      "B",
      "Mô hình",
      main + '<p class="plan-note">' + esc(models.note) + "</p>" + extra,
    );
  }

  function renderModes() {
    var cards =
      '<ol class="mode-grid">' +
      list(plan.modes, function (m, i) {
        return (
          '<li class="mode-card">' +
          '<div class="mode-top"><span class="mode-index">0' +
          (i + 1) +
          "</span>" +
          '<span class="mode-runs">' +
          list(m.runs, function (r) {
            return "<span>" + esc(r) + "</span>";
          }) +
          "</span></div>" +
          "<h4>" +
          esc(m.name) +
          '</h4><p class="mode-init">' +
          esc(m.init) +
          "</p>" +
          "<dl><div><dt>Phần train</dt><dd>" +
          esc(m.train) +
          "</dd></div>" +
          "<div><dt>Trainable</dt><dd>" +
          esc(m.trainable) +
          "</dd></div>" +
          "<div><dt>Lịch train</dt><dd>" +
          esc(m.schedule) +
          "</dd></div></dl>" +
          '<ul class="plan-chips small">' +
          list(m.aug, function (a) {
            return "<li>" + esc(a) + "</li>";
          }) +
          "</ul>" +
          "</li>"
        );
      }) +
      "</ol>";
    var legend =
      '<p class="plan-note mode-legend">Mã run: R1–R4 là ResNet-50, R5–R8 là DeiT-S. Số trainable ghi theo thứ tự ResNet-50 · DeiT-S.</p>';
    return block(
      "C",
      "Ma trận thí nghiệm · 2 mô hình × 4 chế độ",
      cards + legend,
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
      block("D", "Cấu hình chung", chips(plan.setup)) +
      block("E", "Metric ghi lại", chips(plan.metrics)) +
      "</div>"
    );
  }

  root.innerHTML =
    renderHead() +
    renderStats() +
    '<div class="plan-columns">' +
    renderData() +
    renderModels() +
    "</div>" +
    renderModes() +
    renderFooter();
  root.classList.add("is-ready");
})();
