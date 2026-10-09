// herde serve's page: plain JavaScript, no outside assets. Everything a worker reports (names, voice labels)
// goes into the page as text, never as HTML: a worker on someone else's computer must not write into it.
'use strict';

const KEY = 'herde-serve-token';
const $ = (id) => document.getElementById(id);
let token = '';
let state = null;
const rendered = new Map(); // job id → { sig, li }: a playing clip is not torn down by the next refresh
const clips = new Map(); // job id → blob URL

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : String(c));
  return e;
}

async function api(method, path, body) {
  const res = await fetch('/api/' + path, {
    method,
    headers: { 'X-Herde-Token': token, ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) lock();
  const kind = res.headers.get('Content-Type') || '';
  const data = kind.startsWith('application/json') ? await res.json() : await res.blob();
  if (!res.ok) throw new Error((data && data.error) || 'HTTP ' + res.status);
  return data;
}

function lock() {
  try { localStorage.removeItem(KEY); } catch { /* private window */ }
  $('locked').hidden = false;
  $('main').hidden = true;
}

function start() {
  // The token arrives once in the address's fragment (never sent to a server, never in a log), then lives in this browser.
  const m = /token=([A-Za-z0-9_-]+)/.exec(location.hash);
  if (m) {
    token = m[1];
    try { localStorage.setItem(KEY, token); } catch { /* private window: this tab only */ }
    history.replaceState(null, '', location.pathname);
  } else {
    try { token = localStorage.getItem(KEY) || ''; } catch { token = ''; }
  }
  if (!token) return lock();
  $('main').hidden = false;
  wire();
  refresh();
  setInterval(refresh, 2000);
}

async function refresh() {
  try {
    state = await api('GET', 'state');
  } catch {
    return;
  }
  $('name').textContent = state.name && state.name !== 'herde serve' ? '· ' + state.name : 'serve';
  renderWorkers();
  renderInvites();
  renderChoices();
  renderJobs();
}

// --- workers ---------------------------------------------------------------------------------------

function engines(w) {
  return (w.engines || []).map((e) => e.kind === 'tts' ? `speech: ${e.model} (${(e.voices || []).length} voices)` : `${e.kind}: ${e.model}`).join(', ');
}

function renderWorkers() {
  const t = $('workers');
  t.replaceChildren();
  $('no-workers').hidden = state.workers.length > 0;
  if (!state.workers.length) return;
  t.append(el('tr', {}, el('th', {}, 'Computer'), el('th', {}, 'State'), el('th', {}, 'Offers'), el('th', {}, '')));
  for (const w of state.workers) {
    const mark = !w.active ? 'switched off' : w.online ? 'online' : w.state === 'paused' ? 'paused' : w.state === 'loading' ? 'loading its models' : 'offline';
    t.append(el('tr', {},
      el('td', {}, w.name, el('div', { class: 'muted small' }, [w.version, w.platform && w.platform.accelerator].filter(Boolean).join(' · '))),
      el('td', {}, el('span', { class: 'pill ' + (w.online ? 'good' : 'warn') }, mark)),
      el('td', { class: 'small' }, engines(w) || '—'),
      el('td', {},
        el('button', { type: 'button', class: 'ghost', onclick: () => act('PATCH', 'workers/' + w.id, { active: !w.active }) }, w.active ? 'Switch off' : 'Switch on'),
        el('button', { type: 'button', class: 'ghost', onclick: () => confirm(`Remove ${w.name}? Its key stops working.`) && act('DELETE', 'workers/' + w.id) }, 'Remove'))));
  }
}

function renderInvites() {
  const ul = $('invites');
  ul.replaceChildren(...state.invites.map((i) => el('li', {},
    `Open invite: ${i.name || 'a computer'}, valid until ${new Date(i.expires * 1000).toLocaleString()} `,
    el('button', { type: 'button', class: 'ghost', onclick: () => act('DELETE', 'invites/' + i.id) }, 'Withdraw'))));
}

async function act(method, path, body) {
  try {
    await api(method, path, body);
  } catch (e) {
    alertError(e);
  }
  refresh();
}

// --- new jobs ----------------------------------------------------------------------------------------

function online() {
  return state.workers.filter((w) => w.online);
}

/** What the online workers can do: voices with their languages, and text models. */
function offered() {
  const voices = new Map();
  const models = new Set();
  for (const w of online()) {
    for (const e of w.engines || []) {
      if (e.kind === 'tts') for (const v of e.voices || []) {
        const known = voices.get(v.id) || { label: v.label || v.id, langs: new Set(), model: e.model };
        for (const l of v.langs || []) known.langs.add(l);
        voices.set(v.id, known);
      }
      if (e.kind === 'text') models.add(e.model);
    }
  }
  return { voices, models };
}

function fill(select, options, keep) {
  const current = keep ?? select.value;
  select.replaceChildren(...options.map(([value, label]) => el('option', { value }, label)));
  if (options.some(([v]) => v === current)) select.value = current;
}

function renderChoices() {
  const { voices, models } = offered();
  const f = $('tts-form');
  fill(f.voice, [...voices.entries()].map(([id, v]) => [id, v.label]));
  const v = voices.get(f.voice.value);
  fill(f.lang, [...(v ? v.langs : [])].sort().map((l) => [l, l]));
  $('tts-hint').textContent = voices.size ? '' : 'No worker with speech is online yet.';
  fill($('text-form').model, [...models].sort().map((m) => [m, m]));
  $('text-hint').textContent = models.size ? '' : 'No worker with a text model is online yet.';
}

function wire() {
  for (const b of document.querySelectorAll('[data-tab]')) {
    b.addEventListener('click', () => {
      for (const o of document.querySelectorAll('[data-tab]')) o.setAttribute('aria-selected', String(o === b));
      $('tts-form').hidden = b.dataset.tab !== 'tts';
      $('text-form').hidden = b.dataset.tab !== 'text';
    });
  }
  $('tts-form').voice.addEventListener('change', () => renderChoices());
  $('invite-form').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    try {
      const inv = await api('POST', 'invites', { name: ev.target.name.value });
      $('invite-title').textContent = inv.name ? `Invite for ${inv.name}` : 'Invite';
      $('invite-code').textContent = inv.code;
      $('invite-join').textContent = inv.join;
      $('invite-box').hidden = false;
      ev.target.reset();
      refresh();
    } catch (e) {
      alertError(e);
    }
  });
  $('invite-done').addEventListener('click', () => { $('invite-box').hidden = true; });
  $('tts-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const voice = offered().voices.get(f.voice.value);
    submit({ kind: 'tts', text: f.text.value, voice: f.voice.value, lang: f.lang.value, model: voice ? voice.model : undefined,
      instruct: f.instruct.value, temperature: Number(f.temperature.value) });
  });
  $('text-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const job = { kind: 'text', model: f.model.value, prompt: f.prompt.value, max_tokens: Number(f.max_tokens.value) };
    if (f.system.value.trim()) job.system = f.system.value;
    if (f.schema.value.trim()) {
      try {
        job.json_schema = JSON.parse(f.schema.value);
      } catch {
        return showError('The JSON schema is not valid JSON.');
      }
    }
    submit(job);
  });
}

async function submit(job) {
  $('job-error').hidden = true;
  try {
    await api('POST', 'jobs', job);
    refresh();
  } catch (e) {
    alertError(e);
  }
}

function showError(text) {
  $('job-error').textContent = text;
  $('job-error').hidden = false;
}

function alertError(e) {
  showError('Not done: ' + (e && e.message ? e.message : e));
}

// --- jobs ------------------------------------------------------------------------------------------

function seconds(a, b) {
  return a && b ? ` in ${Math.max(0, b - a)} s` : '';
}

function renderJobs() {
  const ul = $('jobs');
  $('no-jobs').hidden = state.jobs.length > 0;
  const seen = new Set();
  let before = null;
  for (const j of state.jobs) {
    seen.add(j.id);
    const sig = [j.state, j.finished, j.error, j.worker, j.failures].join('|');
    let entry = rendered.get(j.id);
    if (!entry || entry.sig !== sig) {
      const li = jobItem(j);
      if (entry) entry.li.replaceWith(li);
      entry = { sig, li };
      rendered.set(j.id, entry);
    }
    // Newest first: each after the one before it.
    if (before ? before.nextSibling !== entry.li : ul.firstChild !== entry.li) {
      if (before) before.after(entry.li);
      else ul.prepend(entry.li);
    }
    before = entry.li;
  }
  for (const [id, entry] of rendered) {
    if (!seen.has(id)) {
      entry.li.remove();
      rendered.delete(id);
    }
  }
}

function jobItem(j) {
  const kind = j.kind === 'tts' ? `Speech · ${j.voice} · ${j.lang}` : `Text · ${j.model}`;
  const where = j.worker ? ` on ${j.worker}` : '';
  const status = {
    queued: 'waiting for a worker',
    leased: 'being made' + where,
    done: 'done' + seconds(j.started, j.finished) + where,
    failed: 'failed' + where + (j.error ? `: ${j.error}` : ''),
    cancelled: 'cancelled' + (j.error && j.error !== 'cancelled' ? ` (${j.error})` : ''),
  }[j.state] || j.state;
  const li = el('li', { class: 'card' },
    el('div', { class: 'row between' },
      el('strong', {}, `#${j.id} ${kind}`),
      el('span', { class: 'pill ' + (j.state === 'done' ? 'good' : j.state === 'failed' || j.state === 'cancelled' ? 'bad' : 'warn') }, status)),
    el('p', { class: 'muted small quote' }, j.input || ''));
  if (j.state === 'queued' || j.state === 'leased') {
    li.append(el('button', { type: 'button', class: 'ghost', onclick: () => act('DELETE', 'jobs/' + j.id) }, 'Cancel'));
  }
  if (j.kind === 'tts' && j.state === 'done') {
    const holder = el('div', {});
    const play = el('button', { type: 'button', onclick: async () => {
      try {
        if (!clips.has(j.id)) clips.set(j.id, URL.createObjectURL(await api('GET', `jobs/${j.id}/audio`)));
        holder.replaceChildren(el('audio', { controls: true, autoplay: true, src: clips.get(j.id) }),
          el('a', { href: clips.get(j.id), download: `herde-${j.id}.mp3` }, 'Download'));
      } catch (e) {
        alertError(e);
      }
    } }, `Play (${(j.ms / 1000).toFixed(1)} s)`);
    holder.append(play);
    li.append(holder);
  }
  if (j.kind === 'text' && j.state === 'done' && j.output) {
    let text = j.output.content;
    try { text = JSON.stringify(JSON.parse(text), null, 2); } catch { /* plain text */ }
    li.append(el('pre', {}, text), el('p', { class: 'muted small' },
      `${j.output.usage.input_tokens} tokens read, ${j.output.usage.output_tokens} written` + (j.output.finish === 'length' ? ' — cut at the limit' : '')));
  }
  return li;
}

start();
