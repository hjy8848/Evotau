(() => {
  const root = document.documentElement;
  const safeGet = (key) => { try { return localStorage.getItem(key); } catch (_) { return null; } };
  const safeSet = (key, value) => { try { localStorage.setItem(key, value); } catch (_) {} };
  const applyMode = (mode) => {
    const value = mode === "research" ? "research" : "simple";
    root.dataset.displayMode = value;
    document.querySelectorAll("[data-display-mode-button]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.displayModeButton === value));
    });
  };
  const storedMode = safeGet("evotau-display-mode");
  applyMode(storedMode === "research" ? "research" : "simple");
  document.querySelectorAll("[data-display-mode-button]").forEach((button) => button.addEventListener("click", () => {
    const mode = button.dataset.displayModeButton;
    safeSet("evotau-display-mode", mode);
    applyMode(mode);
  }));
  const setTheme = (theme) => {
    const value = ["dark", "light", "auto"].includes(theme) ? theme : "auto";
    if (value === "auto") root.removeAttribute("data-theme");
    else root.dataset.theme = value;
    root.dataset.themeMode = value;
    const toggle = document.querySelector("[data-theme-toggle]");
    if (toggle) toggle.setAttribute("aria-label", `当前主题：${value === "auto" ? "跟随系统" : value === "dark" ? "深色" : "浅色"}；点击切换`);
  };
  setTheme(safeGet("evotau-theme") || "auto");
  document.querySelector("[data-theme-toggle]")?.addEventListener("click", () => {
    const current = root.dataset.themeMode || "auto";
    const next = current === "auto" ? "dark" : current === "dark" ? "light" : "auto";
    safeSet("evotau-theme", next);
    setTheme(next);
  });
})();
