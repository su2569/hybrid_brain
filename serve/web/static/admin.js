// ===== Tab =====
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('tab-active'));
    t.classList.add('tab-active');
    document.querySelectorAll('.tab-content').forEach(x => x.classList.add('hidden'));
    document.getElementById('tab-' + t.dataset.tab).classList.remove('hidden');
    refreshCurrentTab();
  };
});

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (r.status === 401) { location.href = '/admin/login'; throw new Error('未登录'); }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

async function logout() {
  await fetch('/admin/logout', {method: 'POST'});
  location.href = '/admin/login';
}

// ===== 实时刷新（mtime 轮询） =====
let _lastFP = {};
const POLL_MS = 3000;
async function pollFingerprint() {
  try {
    const {fp} = await fetchJSON('/admin/api/db/fingerprint');
    let changed = false;
    for (const [k, v] of Object.entries(fp)) {
      if (_lastFP[k] && _lastFP[k] !== v) changed = true;
      _lastFP[k] = v;
    }
    if (changed) refreshCurrentTab();
  } catch (e) {}
}
setInterval(pollFingerprint, POLL_MS);

function refreshCurrentTab() {
  const active = document.querySelector('.tab-active')?.dataset.tab;
  ({dashboard: loadDashboard, users: loadUsers, kb: loadKB,
    db: loadDB, memory: loadMemory, graph: loadGraph})[active]?.();
}

// ===== 概览 =====
async function loadDashboard() {
  const health = await fetchJSON('/health').catch(() => ({}));
  const kbStats = await fetchJSON('/admin/api/kb/stats').catch(() => ({}));
  const dbs = await fetchJSON('/admin/api/db/list').catch(() => ({dbs: []}));
  const users = await fetchJSON('/admin/api/users/list').catch(() => ({users: []}));
  document.getElementById('status').textContent =
    health.status === 'ok' ? '● 服务正常' : '● 加载中';
  const cards = [
    {label: 'KB 条目', value: kbStats.total || 0, color: 'text-purple-400'},
    {label: '用户', value: users.users?.length || 0, color: 'text-green-400'},
    {label: '数据库', value: (dbs.dbs || []).filter(d => d.exists).length, color: 'text-blue-400'},
    {label: '模型', value: health.model_loaded ? '已加载' : '加载中', color: 'text-yellow-400'},
  ];
  document.getElementById('statCards').innerHTML = cards.map(c => `
    <div class="bg-slate-800 rounded p-4">
      <div class="text-sm text-slate-400 mb-1">${c.label}</div>
      <div class="text-2xl font-bold ${c.color}">${c.value}</div>
    </div>`).join('');
  document.getElementById('sysInfo').textContent = JSON.stringify({
    kb_total: kbStats.total, emb_dim: kbStats.emb_dim,
    model_loaded: health.model_loaded,
  }, null, 2);
}

// ===== 用户 =====
async function loadUsers() {
  const d = await fetchJSON('/admin/api/users/list');
  const users = d.users || [];
  document.getElementById('userCount').textContent = `共 ${users.length} 个用户`;
  document.getElementById('userBody').innerHTML = users.map(u => `
    <tr class="border-b border-slate-700 hover:bg-slate-700/30">
      <td class="p-3 text-blue-400 font-mono">${escapeHtml(u.user_id)}</td>
      <td class="p-3">${escapeHtml(u.display_name)}</td>
      <td class="p-3 text-slate-400 font-mono text-xs">${escapeHtml(u.system_id)}</td>
      <td class="p-3 text-slate-500 text-xs">${escapeHtml(u.note || '')}</td>
      <td class="p-3">
        <button onclick='editUser(${JSON.stringify(u)})' class="px-2 py-1 bg-yellow-600 rounded text-xs">编辑</button>
        <button onclick="delUser('${u.user_id}')" class="px-2 py-1 bg-red-600 rounded text-xs">删除</button>
      </td>
    </tr>`).join('') || '<tr><td colspan="5" class="p-4 text-slate-500 text-center">无用户</td></tr>';
}

let _editUserMode = null;
function showAddUser() {
  _editUserMode = null;
  document.getElementById('userModalTitle').textContent = '新建用户';
  document.getElementById('mUserId').value = '';
  document.getElementById('mUserId').disabled = false;
  document.getElementById('mDisplayName').value = '';
  document.getElementById('mSystemId').value = '';
  document.getElementById('mNote').value = '';
  document.getElementById('userModal').classList.remove('hidden');
  document.getElementById('userModal').classList.add('flex');
}
function editUser(u) {
  _editUserMode = u.user_id;
  document.getElementById('userModalTitle').textContent = '编辑用户';
  document.getElementById('mUserId').value = u.user_id;
  document.getElementById('mUserId').disabled = true;
  document.getElementById('mDisplayName').value = u.display_name || '';
  document.getElementById('mSystemId').value = u.system_id || '';
  document.getElementById('mNote').value = u.note || '';
  document.getElementById('userModal').classList.remove('hidden');
  document.getElementById('userModal').classList.add('flex');
}
function closeUserModal() {
  document.getElementById('userModal').classList.add('hidden');
  document.getElementById('userModal').classList.remove('flex');
}
async function saveUser() {
  const body = {
    user_id: document.getElementById('mUserId').value.trim(),
    display_name: document.getElementById('mDisplayName').value.trim(),
    system_id: document.getElementById('mSystemId').value.trim(),
    note: document.getElementById('mNote').value.trim(),
  };
  try {
    if (_editUserMode) {
      await fetchJSON('/admin/api/users/update', {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
    } else {
      await fetchJSON('/admin/api/users/create', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
    }
    closeUserModal();
    loadUsers();
  } catch (e) {
    alert('保存失败: ' + e.message);
  }
}
async function delUser(uid) {
  if (!confirm(`删除用户 ${uid}?`)) return;
  await fetchJSON('/admin/api/users/delete?user_id=' + encodeURIComponent(uid), {method: 'DELETE'});
  loadUsers();
}
async function autoRegister() {
  const r = await fetchJSON('/admin/api/users/auto_register', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({}),
  });
  alert(`新增 ${r.added} 个用户`);
  loadUsers();
}

// ===== 知识库 =====
let kbOffset = 0;
async function loadKB() {
  const q = document.getElementById('kbSearch').value.trim();
  const url = `/admin/api/kb/list?offset=${kbOffset}&limit=20${q ? '&q=' + encodeURIComponent(q) : ''}`;
  const d = await fetchJSON(url);
  document.getElementById('kbBody').innerHTML = d.items.map(x => `
    <tr class="border-b border-slate-700 hover:bg-slate-700/50">
      <td class="p-3 text-slate-400">${x.id}</td>
      <td class="p-3">${escapeHtml(x.text)}</td>
    </tr>`).join('');
  document.getElementById('kbMeta').textContent = `共 ${d.total} 条`;
}
function kbPage(delta) {
  kbOffset = Math.max(0, kbOffset + delta * 20);
  loadKB();
}

// ===== 数据库（三栏：库 → 表 → 数据）=====
let currentDb = null;
let currentTable = null;

async function loadDB() {
  const d = await fetchJSON('/admin/api/db/list');
  document.getElementById('dbList').innerHTML = d.dbs.map(x => `
    <li>
      <button onclick="selectDb('${x.name}')"
              class="db-item w-full text-left px-3 py-2 rounded hover:bg-slate-700 ${x.exists ? '' : 'text-slate-500 opacity-60'} ${x.name === currentDb ? 'db-item-active' : ''}">
        <div class="font-medium">${x.name}</div>
        <div class="text-xs text-slate-400">${(x.size/1024).toFixed(1)} KB</div>
      </button>
    </li>`).join('');
  if (currentDb) selectDb(currentDb, true);
}

async function selectDb(name, keepTable = false) {
  currentDb = name;
  if (!keepTable) currentTable = null;
  // 高亮
  document.querySelectorAll('.db-item').forEach(b => {
    b.classList.toggle('db-item-active', b.textContent.trim().startsWith(name));
  });
  const d = await fetchJSON(`/admin/api/db/${name}/tables`);
  document.getElementById('tableList').innerHTML = d.tables.map(t => `
    <li>
      <button onclick="showTable('${t.name}')"
              class="table-btn w-full text-left px-3 py-2 rounded hover:bg-slate-700 ${t.name === currentTable ? 'table-btn-active' : ''}">
        <div class="font-medium">${t.name}</div>
        <div class="text-xs text-slate-400">${t.count} 行</div>
      </button>
    </li>`).join('');
  if (keepTable && currentTable) showTable(currentTable);
  else {
    document.getElementById('crumbTable').textContent = '';
    document.querySelector('#tab-db thead').innerHTML = '';
    document.getElementById('dbBody').innerHTML =
      '<tr><td class="p-4 text-slate-500 text-center">← 选择左侧的表</td></tr>';
  }
}

async function showTable(table) {
  if (!currentDb) return;
  currentTable = table;
  // 高亮表
  document.querySelectorAll('.table-btn').forEach(b => {
    b.classList.toggle('table-btn-active', b.textContent.trim().startsWith(table));
  });
  const d = await fetchJSON(`/admin/api/db/${currentDb}/table/${table}?limit=50`);
  document.getElementById('crumbDb').textContent = currentDb;
  document.getElementById('crumbTable').textContent = table;
  document.getElementById('tableMeta').textContent =
    `共 ${d.total} 行 · 显示前 ${d.items.length}`;

  if (!d.items.length) {
    document.querySelector('#tab-db thead').innerHTML = '';
    document.getElementById('dbBody').innerHTML =
      '<tr><td class="p-4 text-slate-500 text-center">空表</td></tr>';
    return;
  }
  const cols = Object.keys(d.items[0]);
  document.querySelector('#tab-db thead').innerHTML =
    '<tr>' + cols.map(c => `<th class="p-2 text-left">${c}</th>`).join('') +
    '<th class="p-2 text-left w-16">操作</th></tr>';
  document.getElementById('dbBody').innerHTML = d.items.map(row =>
    '<tr class="border-b border-slate-700 hover:bg-slate-700/50">' +
    cols.map(c => `<td class="p-2">${escapeHtml(String(row[c] ?? '').slice(0, 200))}</td>`).join('') +
    `<td class="p-2"><button onclick='editRow(${JSON.stringify(row).replace(/'/g,"&#39;")})' class="px-2 py-0.5 bg-yellow-600 rounded">编辑</button></td>` +
    '</tr>').join('');
  document.getElementById('dbBody').dataset.columns = JSON.stringify(cols);
}

async function editRow(row) {
  const cols = JSON.parse(document.getElementById('dbBody').dataset.columns || '[]');
  const pk = cols.includes('id') ? 'id' : cols[0];
  const editable = cols.filter(c => c !== pk);
  const vals = {};
  for (const c of editable) {
    const v = prompt(`修改 ${c}\n当前: ${row[c] ?? ''}`, row[c] ?? '');
    if (v === null) return;
    vals[c] = v;
  }
  const url = `/admin/api/db/${currentDb}/table/${currentTable}/update`
            + `?pk=${pk}&pk_val=${encodeURIComponent(row[pk])}`;
  try {
    await fetchJSON(url, {
      method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(vals),
    });
    showTable(currentTable);
  } catch (e) {
    alert('更新失败: ' + e.message);
  }
}

// ===== 记忆库 =====
async function loadMemory() {
  const [l1, l2, sess] = await Promise.all([
    fetchJSON('/admin/api/memory/l1/users').catch(() => ({users: []})),
    fetchJSON('/admin/api/memory/l2/items?limit=20').catch(() => ({items: []})),
    fetchJSON('/admin/api/memory/l1/sessions').catch(() => ({items: []})),
  ]);
  document.getElementById('l1Users').innerHTML = (l1.users || []).map(u => `
    <li>
      <button onclick="loadUserFacts('${u.user_id}')" class="w-full text-left px-2 py-1 rounded hover:bg-slate-700">
        ${u.user_id} <span class="text-slate-500 text-xs">(${u.count})</span>
      </button>
    </li>`).join('') || '<li class="text-slate-500">无</li>';
  document.getElementById('l2Items').innerHTML = (l2.items || []).map(it => `
    <li class="border-b border-slate-700 pb-2">
      <div class="text-xs text-slate-400">${it.user_id}</div>
      <div>${escapeHtml(it.text.slice(0, 100))}</div>
    </li>`).join('') || '<li class="text-slate-500">无</li>';
  document.getElementById('sessions').innerHTML = (sess.items || []).map(s => `
    <div class="bg-slate-700/50 rounded p-2">
      <div class="text-xs text-slate-400">${s.session_id.slice(0, 16)}</div>
      <div>${s.n} 条 · ${s.user_id}</div>
    </div>`).join('') || '<div class="text-slate-500">无</div>';
}
async function loadUserFacts(uid) {
  const d = await fetchJSON(`/admin/api/memory/l1/facts?user_id=${uid}`);
  document.getElementById('l1Facts').innerHTML = d.items.map(f => `
    <tr class="border-b border-slate-700">
      <td class="p-2 text-slate-400">${f.key}</td>
      <td class="p-2">${escapeHtml(f.value)}</td>
    </tr>`).join('') || '<tr><td colspan="2" class="text-slate-500 p-3">无</td></tr>';
}

// ===== 图谱 =====
let network = null;
async function loadGraph() {
  const scope = document.getElementById('graphScope').value;
  const user = document.getElementById('graphUser').value.trim();
  const url = `/admin/api/graph/overview?scope=${scope}`
            + (user ? `&user_id=${encodeURIComponent(user)}` : '');
  const d = await fetchJSON(url);

  const nodes = d.nodes.map(n => ({
    id: n.id, label: n.label, title: n.title, group: n.domain,
    size: n.size || 20,
    borderWidth: n.is_center ? 5 : 2,
    shadow: n.is_center ? {enabled: true, color: 'rgba(239,68,68,0.5)', size: 30} : false,
    font: {size: n.is_center ? 16 : 12, color: '#e2e8f0', strokeWidth: 2, strokeColor: '#0f172a'},
  }));
  const EDGE_STYLE = {
    semantic:    {color: '#22c55e', dashes: [5, 5], width: 1.5},
    knowledge:   {color: '#3b82f6', dashes: false,  width: 1.5},
    information: {color: '#f97316', dashes: false,  width: 2.5},
  };
  const edges = d.edges.map((e, i) => {
    const st = EDGE_STYLE[e.type] || EDGE_STYLE.knowledge;
    return {id: 'e' + i, from: e.from, to: e.to,
            color: {color: st.color, opacity: 0.65}, dashes: st.dashes, width: st.width};
  });
  const options = {
    nodes: {shape: 'dot', scaling: {min: 10, max: 60}},
    groups: {
      preference: {color: {background: '#f59e0b', border: '#b45309'}},
      identity:   {color: {background: '#8b5cf6', border: '#6d28d9'}},
      location:   {color: {background: '#06b6d4', border: '#0e7490'}},
      habit:      {color: {background: '#10b981', border: '#047857'}},
      note:       {color: {background: '#3b82f6', border: '#1d4ed8'}},
      session:    {color: {background: '#64748b', border: '#334155'}},
      center:     {color: {background: '#ef4444', border: '#b91c1c'}},
    },
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {gravitationalConstant: -80, centralGravity: 0.02,
                          springLength: 130, springConstant: 0.08, damping: 0.5},
      stabilization: {iterations: 300},
    },
    interaction: {hover: true, tooltipDelay: 80, navigationButtons: true,
                  keyboard: {enabled: true, bindToWindow: false}},
  };
  const container = document.getElementById('graphCanvas');
  if (network) network.destroy();
  network = new vis.Network(container,
    {nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges)}, options);

  network.on('click', p => {
    if (p.nodes.length === 0) return;
    const nid = p.nodes[0];
    const connected = network.getConnectedNodes(nid);
    const ce = network.getConnectedEdges(nid);
    const an = network.body.data.nodes;
    const ae = network.body.data.edges;
    an.update(an.get().map(n => ({id: n.id,
      opacity: (n.id === nid || connected.includes(n.id)) ? 1 : 0.15})));
    ae.update(ae.get().map(e => ({id: e.id,
      color: ce.includes(e.id) ? {color: '#fbbf24', opacity: 1} : {opacity: 0.05}})));
  });
  network.on('doubleClick', () => {
    const an = network.body.data.nodes;
    const ae = network.body.data.edges;
    an.update(an.get().map(n => ({id: n.id, opacity: 1})));
    ae.update(ae.get().map(e => ({id: e.id, opacity: undefined})));
  });
  const s = d.stats;
  document.getElementById('graphStats').innerHTML =
    `节点 <b>${s.nodes}</b> · 边 <b>${s.edges}</b>`;
  renderLegend(d.legend);
}

function renderLegend(legend) {
  const el = document.getElementById('graphLegend');
  const dh = Object.entries(legend.domains).map(([k, v]) =>
    `<span class="flex items-center gap-1">
       <span style="background:${v.color}" class="w-3 h-3 rounded-full inline-block"></span>${v.label}
     </span>`).join('');
  const eh = Object.entries(legend.edges).map(([k, v]) =>
    `<span class="flex items-center gap-1">
       <svg width="24" height="8"><line x1="0" y1="4" x2="24" y2="4"
         stroke="${v.color}" stroke-width="2"
         stroke-dasharray="${k === 'semantic' ? '4,3' : ''}"/></svg>${v.label}
     </span>`).join('');
  el.className = 'bg-slate-800 rounded p-3 mt-3 text-xs flex flex-wrap gap-4';
  el.innerHTML =
    `<div class="flex gap-4 flex-wrap"><b class="text-slate-400">节点：</b>${dh}</div>` +
    `<div class="flex gap-4 flex-wrap mt-2"><b class="text-slate-400">边：</b>${eh}</div>`;
}

// ===== 工具 =====
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// ===== 初始化 =====
loadDashboard();
