(() => {
  const timeline = document.querySelector("[data-live-timeline]");
  if (!timeline || !window.EventSource) return;
  const labels = {
    run_started: "Run started", run_finished: "Run completed", run_failed: "Run stopped",
    generation_committed: "Generation committed", episode_started: "Episode started",
    episode_finished: "Episode completed", evaluation_finished: "τ-bench evaluation completed",
    tool_call: "Service called a tool", tool_result: "Tool returned a result",
    service_strategy_proposed: "Service strategy proposed", service_strategy_selected: "Service proposal compared",
    budget_updated: "Provider budget updated", customer_message: "Customer message",
    agent_message: "Service response",
  };
  const feed = new EventSource(window.location.pathname.replace(/\/$/, "") + "/events");
  let seenEventIds = new Set();
  try { seenEventIds = new Set(JSON.parse(timeline.dataset.seenEventIds || "[]")); } catch (_) { /* keep the live view available */ }
  feed.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data);
      if (event.event_id && seenEventIds.has(event.event_id)) return;
      if (event.event_id) seenEventIds.add(event.event_id);
      const payload = event.payload || {};
      const rowTitle = humanTitle(event.event_type, event, payload);
      const time = new Date(event.timestamp);
      const timeText = Number.isNaN(time.getTime()) ? "—" : time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const timeNode = document.createElement("div"); timeNode.className = "event-time"; timeNode.textContent = timeText;
      const copy = document.createElement("div"); copy.className = "event-copy";
      const strong = document.createElement("strong"); strong.textContent = rowTitle; copy.append(strong);
      if (event.episode_id) {
        const link = document.createElement("a");
        link.href = `${window.location.pathname.replace(/\/$/, "")}/episodes/${encodeURIComponent(event.episode_id)}`;
        link.textContent = ` · Episode ${event.episode_id}`; copy.append(link);
      }
      const raw = document.createElement("span"); raw.className = "research-inline mono";
      raw.textContent = ` · ${event.event_type} · gen ${event.generation ?? "—"}`; copy.append(raw);
      timeline.prepend(timeNode, copy);
    } catch (_) { /* malformed observational rows are omitted from this live view */ }
  };
  function humanTitle(kind, event, payload) {
    if (kind === "generation_committed") return `Generation ${event.generation} committed`;
    if (kind === "episode_started") return `Episode started · ${event.episode_id || "current episode"}`;
    if (kind === "episode_finished") return `Episode completed · Task ${payload.task_id || "Unavailable"}`;
    if (kind === "tool_call") return `Tool call · ${payload.name || payload.tool_name || "tool"}`;
    if (kind === "tool_result") return `Tool result · ${payload.tool_name || "tool"}`;
    if (kind === "evaluation_finished" && typeof payload.task_success === "boolean") {
      return payload.task_success ? "τ-bench task succeeded" : "τ-bench task was not successful";
    }
    if (kind === "service_strategy_selected") return `Service proposal ${payload.accepted ? "accepted" : "not accepted"}`;
    return labels[kind] || "Run record updated";
  }
})();
