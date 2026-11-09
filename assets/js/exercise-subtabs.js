// Sub-tab E1 / E2 / E3 ben trong panel "e1-e3". Doc lap voi assignment-tabs.js.
(function () {
  "use strict";

  var nav = document.querySelector("[data-subtabs]");
  if (!nav) return;

  var buttons = [].slice.call(nav.querySelectorAll(".subtab-btn"));
  if (!buttons.length) return;

  function select(target, focus) {
    buttons.forEach(function (btn) {
      var active = btn === target;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", String(active));
      btn.tabIndex = active ? 0 : -1;
      var panel = document.getElementById(btn.getAttribute("aria-controls"));
      if (panel) panel.hidden = !active;
    });
    if (focus) target.focus();
  }

  buttons.forEach(function (btn, index) {
    btn.addEventListener("click", function () {
      select(btn, false);
    });
    btn.addEventListener("keydown", function (event) {
      var next = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        next = buttons[(index + 1) % buttons.length];
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        next = buttons[(index - 1 + buttons.length) % buttons.length];
      } else if (event.key === "Home") {
        next = buttons[0];
      } else if (event.key === "End") {
        next = buttons[buttons.length - 1];
      }
      if (next) {
        event.preventDefault();
        select(next, true);
      }
    });
  });
})();
