// Tab cho mục 02 / Nội dung học phần.
// - Desktop (>= 768px): thẻ lớn, nghiêng nhẹ khi hover, lật thẻ khi chọn,
//   panel hiện bên dưới lưới thẻ với hiệu ứng pop-up.
// - Mobile (< 768px): tab nhỏ xếp dọc kiểu accordion, panel trượt mở ngay
//   dưới tab được chọn; tab đang mở sẽ đóng lại khi chọn tab khác.
(function () {
  "use strict";

  var tablist = document.querySelector('[data-tabs="assignments"]');
  var panelsHome = document.querySelector(".assignment-panels");
  if (!tablist || !panelsHome) return;

  var tabs = [].slice.call(tablist.querySelectorAll(".assignment-card"));
  var data = window.ASSIGNMENT_PANELS || {};
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var accordionQuery = window.matchMedia("(max-width: 767px)");
  var FLIP_MS = 640;
  var OPEN_MS = 440;
  var CLOSE_MS = 360;
  var EASE = "cubic-bezier(.25,.8,.25,1)";
  var TILT_X = 5;
  var TILT_Y = 7;
  var DEFAULT_TAB = document.getElementById("a1") || tabs[0];
  var current = null;
  var accordion = false;

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

  function renderOverview(item) {
    return (
      '<div class="plan-head">' +
      '<div class="plan-tag"><span class="assignment-code">' +
      esc(item.code) +
      '</span><span class="assignment-type">' +
      esc(item.label) +
      "</span></div>" +
      '<div class="plan-meta"><span>Hạn nộp <strong>' +
      esc(item.deadline) +
      "</strong></span></div>" +
      "</div>" +
      '<div class="plan-intro">' +
      "<div><h3>" +
      esc(item.title) +
      '</h3><p class="plan-lead">' +
      esc(item.lead) +
      "</p></div>" +
      '<div class="plan-questions"><p>' +
      esc(item.deliverables.title) +
      '</p><ul class="plan-checklist">' +
      list(item.deliverables.items, function (d) {
        return "<li>" + esc(d) + "</li>";
      }) +
      "</ul></div>" +
      "</div>" +
      '<ol class="mode-grid milestone-grid">' +
      list(item.milestones, function (m) {
        return (
          '<li class="mode-card">' +
          '<div class="mode-top"><span class="mode-runs"><span>' +
          esc(m.code) +
          '</span></span><span class="milestone-date">' +
          esc(m.deadline) +
          "</span></div>" +
          "<h4>" +
          esc(m.title) +
          '</h4><p class="mode-init">Hạn nộp 23:59 (GMT+7)</p>' +
          '<ul class="plan-chips small">' +
          list(m.topics, function (t) {
            return "<li>" + esc(t) + "</li>";
          }) +
          "</ul></li>"
        );
      }) +
      "</ol>" +
      '<div class="plan-block plan-block-wide">' +
      '<p class="plan-block-label"><span>↗</span>Liên kết sản phẩm</p>' +
      '<dl class="resource-list">' +
      list(item.resources, function (r) {
        return "<div><dt>" + esc(r[0]) + "</dt><dd>" + esc(r[1]) + "</dd></div>";
      }) +
      "</dl></div>"
    );
  }

  function panelOf(tab) {
    return document.getElementById(tab.getAttribute("aria-controls"));
  }

  tabs.forEach(function (tab) {
    tab.classList.add("is-tab");
    var panel = panelOf(tab);
    if (panel && panel.hasAttribute("data-overview") && data[tab.id]) {
      panel.innerHTML = renderOverview(data[tab.id]);
    }
  });

  function animated() {
    return !reduceMotion.matches && typeof Element.prototype.animate === "function";
  }

  // ---------- Desktop: lật thẻ + pop-up panel ----------

  // Lật quanh trục Y; đổi màu đúng lúc thẻ quay ngang (giữa animation).
  function flip(tab, active) {
    clearTimeout(tab._flipTimer);
    if (!animated()) {
      tab.classList.toggle("is-active", active);
      return;
    }
    tab.classList.remove("is-flipping");
    void tab.offsetWidth;
    tab.classList.add("is-flipping");
    tab._flipTimer = setTimeout(function () {
      tab.classList.toggle("is-active", active);
    }, FLIP_MS / 2);
  }

  function popIn(panel) {
    panel.hidden = false;
    panel.classList.remove("is-entering");
    if (!animated()) return;
    void panel.offsetWidth;
    panel.classList.add("is-entering");
  }

  // ---------- Mobile: accordion trượt mở / đóng ----------

  function stopSlide(panel) {
    if (panel._slide) {
      panel._slide.onfinish = null;
      panel._slide.cancel();
      panel._slide = null;
    }
  }

  function slideOpen(panel) {
    var from = panel.hidden ? 0 : panel.offsetHeight;
    stopSlide(panel);
    panel.hidden = false;
    panel.classList.remove("is-sliding");
    if (!animated()) return;
    var style = getComputedStyle(panel);
    var to = panel.offsetHeight;
    void panel.offsetWidth;
    panel.classList.add("is-sliding");
    panel._slide = panel.animate(
      [
        {
          height: from + "px",
          paddingTop: from ? style.paddingTop : "0px",
          paddingBottom: from ? style.paddingBottom : "0px",
          opacity: from ? 1 : 0,
        },
        {
          height: to + "px",
          paddingTop: style.paddingTop,
          paddingBottom: style.paddingBottom,
          opacity: 1,
        },
      ],
      { duration: OPEN_MS, easing: EASE },
    );
    panel._slide.onfinish = function () {
      panel._slide = null;
    };
  }

  function slideClose(panel) {
    if (panel.hidden) return;
    var from = panel.offsetHeight;
    stopSlide(panel);
    panel.classList.remove("is-sliding");
    if (!animated()) {
      panel.hidden = true;
      return;
    }
    var style = getComputedStyle(panel);
    panel._slide = panel.animate(
      [
        {
          height: from + "px",
          paddingTop: style.paddingTop,
          paddingBottom: style.paddingBottom,
          opacity: 1,
        },
        { height: "0px", paddingTop: "0px", paddingBottom: "0px", opacity: 0 },
      ],
      { duration: CLOSE_MS, easing: EASE },
    );
    panel._slide.onfinish = function () {
      panel._slide = null;
      panel.hidden = true;
    };
  }

  // Khi panel phía trên đang đóng, giữ tab vừa chọn đứng yên trên màn hình.
  function pinInView(tab, duration) {
    if (!animated()) return;
    var anchor = tab.getBoundingClientRect().top;
    var end = performance.now() + duration + 40;
    (function frame() {
      var delta = tab.getBoundingClientRect().top - anchor;
      if (Math.abs(delta) > 0.5) {
        window.scrollTo({ top: window.scrollY + delta, behavior: "instant" });
      }
      if (performance.now() < end) requestAnimationFrame(frame);
    })();
  }

  // ---------- Trạng thái chung ----------

  function syncAria() {
    tabs.forEach(function (t) {
      var on = t === current;
      if (accordion) {
        t.setAttribute("aria-expanded", String(on));
        t.removeAttribute("aria-selected");
        t.tabIndex = 0;
      } else {
        t.setAttribute("aria-selected", String(on));
        t.removeAttribute("aria-expanded");
        t.tabIndex = on ? 0 : -1;
      }
    });
  }

  function select(tab, options) {
    options = options || {};

    if (accordion) {
      var previous = current;
      if (tab === previous) {
        if (options.keepOpen) return;
        current = null;
        tab.classList.remove("is-active");
        slideClose(panelOf(tab));
      } else {
        current = tab;
        if (previous) {
          previous.classList.remove("is-active");
          slideClose(panelOf(previous));
          if (tabs.indexOf(previous) < tabs.indexOf(tab)) pinInView(tab, CLOSE_MS);
        }
        tab.classList.add("is-active");
        slideOpen(panelOf(tab));
      }
      syncAria();
      if (options.focus) tab.focus({ preventScroll: true });
      return;
    }

    if (tab === current) return;
    var prev = current;
    current = tab;
    tabs.forEach(function (t) {
      if (t !== tab) panelOf(t).hidden = true;
    });
    if (prev) flip(prev, false);
    flip(tab, true);
    popIn(panelOf(tab));
    syncAria();
    if (options.focus) tab.focus({ preventScroll: true });
  }

  // Đổi bố cục giữa tab (desktop) và accordion (mobile), không animation.
  function applyMode() {
    accordion = accordionQuery.matches;
    tablist.classList.toggle("is-accordion", accordion);

    if (accordion) {
      tablist.removeAttribute("role");
      panelsHome.hidden = true;
    } else {
      tablist.setAttribute("role", "tablist");
      panelsHome.hidden = false;
      if (!current) current = DEFAULT_TAB;
    }

    tabs.forEach(function (t) {
      var panel = panelOf(t);
      var on = t === current;
      stopSlide(panel);
      clearTimeout(t._flipTimer);
      t.classList.remove("is-flipping");
      panel.classList.remove("is-entering", "is-sliding");
      t.classList.toggle("is-active", on);
      t.setAttribute("role", accordion ? "button" : "tab");
      panel.setAttribute("role", accordion ? "region" : "tabpanel");
      if (accordion) t.after(panel);
      else panelsHome.appendChild(panel);
      panel.hidden = !on;
    });
    syncAria();
  }

  function tabFromHash() {
    var id = decodeURIComponent(location.hash.slice(1));
    if (!id) return null;
    for (var i = 0; i < tabs.length; i++) {
      if (tabs[i].id === id || tabs[i].getAttribute("aria-controls") === id) {
        return tabs[i];
      }
    }
    return null;
  }

  tabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () {
      select(tab);
    });

    tab.addEventListener("keydown", function (event) {
      var key = event.key;
      if (key === "Enter" || key === " ") {
        event.preventDefault();
        select(tab, { focus: true });
        return;
      }
      var next = null;
      if (key === "ArrowRight" || key === "ArrowDown")
        next = tabs[(index + 1) % tabs.length];
      else if (key === "ArrowLeft" || key === "ArrowUp")
        next = tabs[(index - 1 + tabs.length) % tabs.length];
      else if (key === "Home") next = tabs[0];
      else if (key === "End") next = tabs[tabs.length - 1];
      if (!next) return;
      event.preventDefault();
      // Accordion: mũi tên chỉ di chuyển focus; tab ngang: chọn luôn.
      if (accordion) next.focus();
      else select(next, { focus: true });
    });

    tab.addEventListener("animationend", function (event) {
      if (event.animationName === "card-flip") tab.classList.remove("is-flipping");
    });

    // Nghiêng nhẹ theo vị trí chuột; bỏ qua cảm ứng, accordion và thẻ đang chọn.
    var rect = null;
    tab.addEventListener("pointerenter", function (event) {
      rect = event.pointerType === "mouse" ? tab.getBoundingClientRect() : null;
    });
    tab.addEventListener("pointermove", function (event) {
      if (!rect || accordion || tab === current || reduceMotion.matches) return;
      var x = (event.clientX - rect.left) / rect.width - 0.5;
      var y = (event.clientY - rect.top) / rect.height - 0.5;
      tab.style.setProperty("--ry", (x * 2 * TILT_Y).toFixed(2) + "deg");
      tab.style.setProperty("--rx", (-y * 2 * TILT_X).toFixed(2) + "deg");
    });
    tab.addEventListener("pointerleave", function () {
      rect = null;
      tab.style.removeProperty("--rx");
      tab.style.removeProperty("--ry");
    });
  });

  window.addEventListener("hashchange", function () {
    var tab = tabFromHash();
    if (tab) select(tab, { keepOpen: true });
  });

  if (accordionQuery.addEventListener) {
    accordionQuery.addEventListener("change", applyMode);
  } else {
    accordionQuery.addListener(applyMode);
  }

  current = tabFromHash() || (accordionQuery.matches ? null : DEFAULT_TAB);
  applyMode();
  tablist.classList.add("is-ready");
})();
