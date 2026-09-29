(() => {
  const dialog = document.querySelector("[data-command-dialog]");
  const input = document.querySelector("[data-command-input]");
  const results = document.querySelector("[data-command-results]");
  if (!dialog || !input || !results) return;
  let items = [];
  let active = 0;
  try { items = JSON.parse(document.getElementById("command-index")?.textContent || "[]"); } catch (_) {}
  const open = () => { if (!dialog.open) dialog.showModal(); input.value = ""; render(); input.focus(); };
  const close = () => { if (dialog.open) dialog.close(); };
  document.querySelector("[data-command-open]")?.addEventListener("click", open);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); open(); }
    if (event.key === "Escape" && dialog.open) close();
  });
  input.addEventListener("input", () => { active = 0; render(); });
  input.addEventListener("keydown", (event) => {
    const visible = filtered();
    if (event.key === "ArrowDown") { event.preventDefault(); active = Math.min(active + 1, visible.length - 1); render(); }
    if (event.key === "ArrowUp") { event.preventDefault(); active = Math.max(active - 1, 0); render(); }
    if (event.key === "Enter" && visible[active]) { event.preventDefault(); location.href = visible[active].url; }
  });
  dialog.addEventListener("click", (event) => { if (event.target === dialog) close(); });
  function filtered() {
    const query = input.value.trim().toLocaleLowerCase();
    if (!query) return items.slice(0, 10);
    return items.filter((item) => `${item.title} ${item.kind} ${item.summary || ""}`.toLocaleLowerCase().includes(query)).slice(0, 20);
  }
  function render() {
    const found = filtered(); results.replaceChildren();
    if (!found.length) { const empty = document.createElement("div"); empty.className = "empty-state"; empty.textContent = "没有匹配的已加载 artifact。"; results.append(empty); return; }
    found.forEach((item, index) => {
      const button = document.createElement("button"); button.type = "button"; button.className = "command-result";
      button.setAttribute("role", "option"); button.setAttribute("aria-selected", String(index === active));
      const title = document.createElement("span"); title.textContent = item.title;
      const detail = document.createElement("small"); detail.textContent = item.kind;
      button.append(title, detail);
      button.addEventListener("mouseenter", () => { active = index; render(); });
      button.addEventListener("click", () => { location.href = item.url; });
      results.append(button);
    });
  }
})();
