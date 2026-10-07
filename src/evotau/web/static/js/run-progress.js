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
  function extras(p, frag, detailState) {
    function section(title) {
      const node = el('section', 'progress-extra'); node.append(el('h3', '', title)); frag.append(node); return node;
    }
    function detail(parent, id, title, open = false) {
      const node = el('details', 'progress-proposal'); node.dataset.monitorDetail = id;
      node.open = detailState.has(id) ? detailState.get(id) : open;
      node.append(el('summary', '', title)); parent.append(node); return node;
    }
    function paragraph(parent, label, text) {
      const node = el('p'); node.append(el('strong', '', label), el('br'), document.createTextNode(text || '—')); parent.append(node);
    }
    function stats(parent, values) {
      const node = el('div', 'progress-health-stats');
      values.forEach(([label, value, tasks]) => {
        const item = el('div'); item.append(el('span', '', label), el('strong', '', value ?? '未记录'));
        if (tasks) item.append(el('span', 'mono small', `Tasks: ${tasks.length ? tasks.join(', ') : '—'}`)); node.append(item);
      }); parent.append(node);
    }
    if (p.active_episodes.length > 1) {
      const active = section(`Active episodes · ${p.active_episodes.length}`);
      p.active_episodes.forEach(ep => active.append(el('p', 'small mono', `Task ${ep.task_id} · ${ep.panel_name} · Turn ${ep.turn ?? '未记录'} · ${ep.role ?? '未记录'} · ${ep.tool_name ?? '—'} · ${ep.updated_at}`)));
    }
    const proposals = section('正在进化什么');
    p.customer_proposals.forEach(proposal => {
      const node = detail(proposals, `customer-${proposal.strategy_id}`, `Customer candidate ${proposal.index} · ${proposal.committed ? 'committed generation proposal' : 'PENDING · 尚未成为 active Customer'} · ${proposal.strategy_id}`, true);
      node.append(el('p', '', proposal.text));
    });
    if (!p.customer_proposals.length) proposals.append(el('p', 'small muted', '本轮尚无 Customer proposal。'));
    const mutation = p.service_mutation;
    if (mutation) {
      const node = detail(proposals, 'service-mutation', `Service ${mutation.operation} · ${mutation.committed ? '已完成决策' : 'PENDING · 尚未成为 active SkillMemory'}${mutation.target_skill_id ? ` · ${mutation.target_skill_id}` : ''}`, true);
      if (mutation.skill) { paragraph(node, 'Trigger', mutation.skill.trigger); paragraph(node, 'Guidance', mutation.skill.guidance); }
      paragraph(node, 'Analysis', mutation.analysis);
      if (mutation.operation === 'NO_OP') node.append(el('p', '', 'NO_OP · 不生成新 skill，不重复 candidate replay。'));
      if (mutation.committed) node.append(el('p', '', `Accepted: ${mutation.accepted}`));
    } else proposals.append(el('p', 'small muted', '本轮尚无 Service mutation。'));
    const pairs = section('Paired transitions · 同 task / seed');
    p.paired_transitions.forEach(pair => {
      const node = detail(pairs, `paired-${pair.label}`, `${pair.label} · ${pair.paired_count}/${pair.total} paired · ${pair.provisional ? 'PROVISIONAL · Not fitness' : 'COMPLETE'}`);
      stats(node, Object.entries(pair.counts).map(([label, count]) => [label, count, pair.task_ids[label]]));
    });
    if (!p.paired_transitions.length) pairs.append(el('p', 'small muted', '尚无配对观察。'));
    const continuation = section('Continuation provenance · 当前可见 E/V episodes'), c = p.continuation;
    continuation.append(el('p', 'small mono', `Parent: ${c.parent || '无'}`));
    stats(continuation, [['Imported complete', c.imported_complete], ['Reused panel references', c.reused_references], ['New complete', c.new_complete], ['Incomplete attempts', c.incomplete_attempts], ['Imported incomplete', c.imported_incomplete], ['New incomplete', c.new_incomplete]]);
    continuation.append(el('p', 'small muted', 'Reuse 统计 panel 引用；imported/new/incomplete 统计 episode attempts，单位不同。缺少来源清单时不猜测 new 数量。'));
    if (p.failure) {
      const f = p.failure, node = el('section', 'progress-failure'), facts = el('dl', 'progress-facts'); node.setAttribute('role', 'status'); node.append(el('h3', '', '运行已停止 · Failure'));
      [['Stage', f.stage], ['Type', f.type], ['Task', f.task_id], ['Panel', f.panel_name], ['Resume-safe', f.resume_safe], ['Checkpoint preserved', f.checkpoint_preserved]].forEach(([label, value]) => fact(facts, label, value));
      node.append(facts, el('p', '', f.message)); frag.append(node);
    }
  }
  function render(p) {
    const opened = new Set([...root.querySelectorAll('[data-skill-id][open]')].map(node => node.dataset.skillId));
    const detailState = new Map([...root.querySelectorAll('[data-monitor-detail]')].map(node => [node.dataset.monitorDetail, node.open]));
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
    table.append(el('caption', '', `Gen ${p.generation} progress · Fitness 仅在完整 panel 上计算；partial 为 PROVISIONAL`));
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
      const accuracyCell = el('td', 'mono', accuracy);
      if (row.provisional) {
        accuracyCell.replaceChildren(el('span', 'provisional', 'PROVISIONAL'), el('div', '', `${row.observed_successes}/${row.count} = ${(row.observed_accuracy * 100).toFixed(0)}%`), el('div', 'small muted', 'Observed so far · Not fitness'));
      }
      tr.append(label, progress, accuracyCell, status); body.append(tr);
    });
    table.append(body); panels.append(table); frag.append(panels);
    const details = el('div', 'progress-detail-grid'), episode = el('section', 'progress-detail'), current = p.current_episode;
    episode.append(el('h3', '', current ? current.label : '当前 episode'));
    if (current) {
      const facts = el('dl', 'progress-facts'); fact(facts, 'Task', current.task_id ?? '等待 task telemetry'); fact(facts, 'Panel', current.panel_name, true); fact(facts, '最后记录的 Turn', current.turn);
      fact(facts, 'Role / Tool', `${current.role ?? '未记录'} / ${current.tool_name ?? '—'}`); fact(facts, 'Next actor', current.next_role); fact(facts, 'Activity', current.activity ?? '已停止'); fact(facts, 'Updated', current.updated_at); episode.append(facts);
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
    extras(p, frag, detailState);
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
