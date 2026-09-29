(() => {
  const panel = document.querySelector("[data-inspector-panel]");
  const content = document.querySelector("[data-inspector-content]");
  if (!panel || !content) return;
  const setContent = (title, html) => {
    const heading = panel.querySelector("[data-inspector-title]");
    if (heading) heading.textContent = title || "Inspector";
    content.innerHTML = html;
    panel.classList.add("is-open");
    panel.setAttribute("aria-live", "polite");
  };
  document.addEventListener("click", async (event) => {
    const target = event.target.closest("[data-inspector-target]");
    if (!target) return;
    if (target.dataset.inspectUrl) {
      event.preventDefault();
      try {
        const response = await fetch(target.dataset.inspectUrl, { headers: { "X-Console-Inspect": "1" } });
        if (!response.ok) throw new Error("inspector unavailable");
        const documentCopy = new DOMParser().parseFromString(await response.text(), "text/html");
        const fragment = documentCopy.querySelector("[data-inspector-fragment]");
        if (!fragment) throw new Error("inspector fragment missing");
        setContent(target.dataset.inspectTitle, fragment.innerHTML);
      } catch (_) { setContent("Unavailable", "<p class=\"muted\">这个 artifact 当前无法打开检查面板。</p>"); }
      return;
    }
    const fragment = target.querySelector("[data-inspector-fragment]") || document.getElementById(target.dataset.fragmentId || "");
    const html = fragment ? fragment.innerHTML : `<p>${escapeText(target.dataset.inspectSummary || target.textContent.trim())}</p>`;
    setContent(target.dataset.inspectTitle || "Episode context", html);
  });
  panel.querySelector("[data-inspector-close]")?.addEventListener("click", () => panel.classList.remove("is-open"));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") panel.classList.remove("is-open"); });
  function escapeText(value) { const el = document.createElement("span"); el.textContent = value; return el.innerHTML; }
})();
