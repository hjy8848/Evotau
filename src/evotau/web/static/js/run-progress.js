(() => {
  const root = document.querySelector('[data-run-progress]');
  if (!root) return;
  const url = root.dataset.progressUrl;
  const runUrl = window.location.pathname.replace(/\/$/, '');
  const knownStates = new Set(['COMPLETE', 'REUSED', 'RUNNING', 'FAILED', 'STOPPED', 'WAITING', 'NOT_STARTED', 'SKIPPED']);
  let timer;
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function latestText(call) {
    if (!call) return '未记录';
    let label = call.state === 'pending' ? '等待响应' : call.transport_error ? 'Transport error' : `HTTP ${call.http_status}`;
    if (typeof call.elapsed_seconds === 'number') label += ` / ${call.elapsed_seconds.toFixed(1)}s`;
    return label;
  }
  function fact(list, label, value, mono = false) {
    const row = el('div'); row.append(el('dt', '', label), el('dd', mono ? 'mono' : '', value ?? '未记录')); list.append(row);
  }
  function render(p) {
    const opened = new Set([...root.querySelectorAll('[data-skill-id][open]')].map(node => node.dataset.skillId));
    const frag = document.createDocumentFragment();
    const header = el('header', 'progress-header'), title = el('div');
    title.append(el('div', 'eyebrow', 'Experiment monitor'), el('h2', '', `${p.domain} E${p.e_size} · G${p.generations ?? '—'} · P${p.parallelism}`));
    header.append(title, el('span', `status status-${p.status} progress-status`, `${p.status.toUpperCase()} · ${p.status_label}`)); frag.append(header);
    const context = el('div', 'progress-context');
    context.append(el('strong', '', `第 ${p.generation_number} 轮（Gen ${p.generation}） / 共 ${p.generations ?? '—'} 轮`), el('span', '', `已提交 ${p.completed_generations} 轮`), el('span', 'mono', `Source: ${p.source_commit ? p.source_commit.slice(0, 8) : '未记录'}`));
    if (p.source_parent) {
      const lineage = el('span', '', `Resume from parent · 复用 ${p.imported_episodes ?? '—'} 场`);
      lineage.title = p.source_parent; context.append(lineage);
    }
    frag.append(context);
    const stage = el('div', 'progress-stage'); stage.append(el('span', 'label-caps', 'Stage'), el('strong', '', p.stage)); frag.append(stage);
    const panels = el('div', 'progress-panels table-scroll'), table = el('table'), head = el('thead'), heading = el('tr'), body = el('tbody');
    table.append(el('caption', '', `Gen ${p.generation} progress · 未完成的 panel 不计算 accuracy`));
    ['Phase', 'Condition / Progress', 'Accuracy', 'Status'].forEach(text => heading.append(el('th', '', text)));
    head.append(heading); table.append(head);
    p.rows.forEach(row => {
      const tr = el('tr'), label = el('th', '', row.label); label.scope = 'row';
      const progress = el('td'), value = el('span', 'mono', row.condition); progress.append(value);
      if (typeof row.count === 'number') {
        const count = el('div', 'progress-count'); count.append(el('strong', '', `${row.count} / ${row.total}`));
        if (row.total > 0) {
          const bar = el('progress'); bar.max = row.total; bar.value = row.count; bar.setAttribute('aria-label', `${row.label} completed episodes`); count.append(bar);
        }
        if (row.reused) count.append(el('span', 'small muted', `${row.reused} reused`)); progress.append(count);
      }
      const accuracy = typeof row.accuracy === 'number' ? `${row.successes}/${row.total} = ${(row.accuracy * 100).toFixed(0)}%` : '—';
      const status = el('td'), state = knownStates.has(row.state) ? row.state : 'NOT_STARTED';
      status.append(el('span', `progress-phase-state state-${state.toLowerCase()}`, state));
      if (state === 'WAITING') status.append(el('div', 'small muted', '等待前阶段'));
      if (state === 'NOT_STARTED') status.append(el('div', 'small muted', '前阶段已停止'));
      tr.append(label, progress, el('td', 'mono', accuracy), status); body.append(tr);
    });
    table.append(body); panels.append(table); frag.append(panels);
    const details = el('div', 'progress-detail-grid'), episode = el('section', 'progress-detail'), current = p.current_episode;
    episode.append(el('h3', '', current ? current.label : '当前 episode'));
    if (current) {
      const facts = el('dl', 'progress-facts'); fact(facts, 'Task', current.task_id ?? '等待 task telemetry'); fact(facts, 'Panel', current.panel_name, true); fact(facts, '最后记录的 Turn', current.turn); episode.append(facts);
      if (current.episode_id) {
        const link = el('a', '', '打开对话与工具轨迹 →'); link.href = `${runUrl}/episodes/${encodeURIComponent(current.episode_id)}`; episode.append(link);
      }
    } else episode.append(el('p', 'small muted', p.status === 'running' ? '尚无已落盘的当前 episode' : '没有正在执行的 episode'));
    const latest = el('div', 'progress-latest'); latest.append(el('span', 'label-caps', 'Latest provider call'), el('strong', '', latestText(p.provider.latest_call))); episode.append(latest);
    const active = el('section', 'progress-detail'); active.append(el('h3', '', 'Current active state'));
    [['Customer', p.active_customer, 'customer'], ['Service', p.active_service, 'service']].forEach(([label, strategy, side]) => {
      const row = el('div', 'progress-active'); row.append(el('span', `badge badge-${side}`, `${label} ${strategy.label}`), el('span', 'mono small muted', strategy.strategy_id || '未记录')); active.append(row);
    });
    active.append(el('p', 'small', `Service SkillMemory: ${p.active_service.skills.length} skill(s) · 仅展示已提交状态`));
    p.active_service.skills.forEach(skill => {
      const detail = el('details', 'progress-skill'); detail.dataset.skillId = skill.skill_id; detail.open = opened.has(skill.skill_id);
      const summary = el('summary'); summary.append(el('strong', '', skill.skill_id), el('span', '', skill.trigger)); detail.append(summary);
      [['Trigger', skill.trigger], ['Guidance', skill.guidance]].forEach(([label, text]) => {
        const para = el('p'); para.append(el('strong', '', label), el('br'), document.createTextNode(text)); detail.append(para);
      }); active.append(detail);
    });
    details.append(episode, active); frag.append(details);
    const health = el('section', 'progress-health'), stats = el('div', 'progress-health-stats'); health.append(el('h3', '', 'Provider health'));
    const healthValues = [['Calls', p.provider.calls ?? '—'], ...Object.entries(p.provider.statuses).map(([code, count]) => [code === 'transport error' ? code : `HTTP ${code}`, count]), ['Flash thinking-off', p.provider.flash_requests === null ? '未记录' : `${p.provider.flash_disabled}/${p.provider.flash_requests}`], ['Peak request concurrency', p.provider.peak_requests ?? '未记录']];
    healthValues.forEach(([label, value]) => { const item = el('div'); item.append(el('span', '', label), el('strong', '', value)); stats.append(item); }); health.append(stats);
    if (!p.provider.coverage_complete) health.append(el('p', 'small muted', 'HTTP/thinking/concurrency 来自可用 wire observations；日志不完整时不推断缺失请求。'));
    frag.append(health);
    if (p.failure_message) {
      const failure = el('div', 'progress-failure'); failure.setAttribute('role', 'status'); failure.append(el('strong', '', '运行已停止，当前轮尚未提交'), el('p', '', p.failure_message)); frag.append(failure);
    }
    const footer = el('footer', 'progress-footer'), time = el('span', '', `快照 ${new Date(p.as_of).toLocaleString()}`); time.dataset.progressFreshness = '';
    footer.append(el('span', '', '只读 artifact 监控 · 不会启动或恢复实验'), time); frag.append(footer);
    root.replaceChildren(frag);
    const badge = document.querySelector('.run-context .status');
    if (badge) { badge.className = `status status-${p.status}`; badge.textContent = p.status_label; }
  }
  async function refresh() {
    clearTimeout(timer);
    if (document.hidden) return;
    try {
      const response = await fetch(url, { cache: 'no-store', headers: { Accept: 'application/json' } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      render(await response.json());
    } catch (error) {
      const stale = root.querySelector('[data-progress-freshness]');
      if (stale) { stale.classList.add('stale'); stale.textContent = `刷新失败（${error.message}）；此处为上次快照，请刷新页面核对`; }
    }
    timer = setTimeout(refresh, 5000);
  }
  document.addEventListener('visibilitychange', () => { clearTimeout(timer); if (!document.hidden) refresh(); });
  timer = setTimeout(refresh, 5000);
})();
