(function() {
  'use strict';

  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${protocol}://${window.location.host}/ws`;

  let socket = null;
  let reconnectAttempts = 0;
  const MAX_RECONNECT = 8;
  let reconnectTimer = null;
  let currentVideoSource = 'normal_video';
  let demoMode = false;
  let devMode = false;

  const DISEASES = {
    coccidiosis: {
      name: 'Кокцидиоз',
      symptoms: 'Кровавый понос, вялость, растрепанные перья, потеря аппетита и веса.',
      cure: 'Обратитесь к ветеринару. Антикокцидиальные препараты, например ампролиум.',
      prevention: 'Поддерживайте чистоту, обеспечьте чистую воду, избегайте скученности, используйте кокцидиостатики в корме.'
    },
    botulism: {
      name: 'Ботулизм',
      symptoms: 'Прогрессирующий паралич от ног, опущенные крылья, затруднённое дыхание, «вялая шея».',
      cure: 'Немедленно устраните источник токсина. Поддерживающая терапия, включая гидратацию.',
      prevention: 'Держите курятник в чистоте, не допускайте доступа к гниющей растительности или тушкам, боритесь с мухами.'
    },
    newcastle: {
      name: 'Болезнь Ньюкасла',
      symptoms: 'Респираторные признаки (хрипы, насморк), диарея, нервные проявления (скрученная шея), снижение яйценоскости.',
      cure: 'Лечения нет. Антибиотики могут помочь при вторичных инфекциях. Поддерживающий уход крайне важен.',
      prevention: 'Вакцинация — основная профилактика. Строгие меры биобезопасности.'
    },
    chickenpox: {
      name: 'Куриная оспа',
      symptoms: 'Сухая оспа (корки на гребне, сережках) или мокрая оспа (язвы во рту), снижение яйценоскости.',
      cure: 'Удалите корки и обработайте антисептиком. При мокрой оспе обратитесь к ветеринару для поддерживающего ухода.',
      prevention: 'Доступна вакцина. Контролируйте комаров, обрабатывайте раны.'
    },
    lice_and_mites: {
      name: 'Вши и клещи',
      symptoms: 'Видимые паразиты на коже и перьях, повреждение оперения, чрезмерное чесание, анемия в тяжёлых случаях.',
      cure: 'Используйте пыль или спрей на основе перметрина. Повторите обработку через 7–10 дней.',
      prevention: 'Регулярная уборка курятника, пылевые ванны с диатомовой землёй, периодические осмотры.'
    }
  };

  // --- Утилиты ---
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const escapeHtml = (str) => {
    if (typeof str !== 'string') return '';
    return str.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
  };

  // --- WebSocket ---
  function connectWebSocket() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
    clearTimeout(reconnectTimer);
    try { socket = new WebSocket(wsUrl); } catch (e) { scheduleReconnect(); return; }
    const thisSocket = socket;
    thisSocket.binaryType = 'arraybuffer';

    thisSocket.onopen = () => {
      reconnectAttempts = 0;
      subscribeToCameraFeeds();
    };

    thisSocket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) { handleBinaryFrame(event.data); return; }
      let msg;
      try { msg = JSON.parse(event.data); } catch (e) { return; }
      handleJsonMessage(msg);
    };

    thisSocket.onerror = () => {};

    thisSocket.onclose = (event) => {
      if (socket !== thisSocket) return;
      socket = null;
      if (!event.wasClean) scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (reconnectAttempts >= MAX_RECONNECT) return;
    reconnectAttempts++;
    const delay = Math.min(2000 * Math.pow(1.5, reconnectAttempts - 1), 30000);
    reconnectTimer = setTimeout(connectWebSocket, delay);
  }

  function sendMessage(msg) {
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
  }

  function subscribeToCameraFeeds() {
    sendMessage({ action: 'subscribe', stream: currentVideoSource });
    sendMessage({ action: 'subscribe', stream: 'thermal_video' });
    sendMessage({ action: 'subscribe', stream: 'debug_stream' });
  }

  function unsubscribeFromStream(stream) {
    sendMessage({ action: 'unsubscribe', stream });
  }

  // --- Бинарные кадры ---
  const feedObjects = { 1: null, 2: null };

  function handleBinaryFrame(data) {
    if (data.byteLength < 2) return;
    const view = new Uint8Array(data);
    const streamId = view[0];
    const jpeg = new Blob([view.slice(1)], { type: 'image/jpeg' });
    const url = URL.createObjectURL(jpeg);

    let targetImg;
    if (streamId === 1 || streamId === 3) targetImg = $('#normal-feed-img');
    else if (streamId === 2) targetImg = $('#thermal-feed-img');

    if (targetImg) {
      const old = feedObjects[streamId];
      if (old) URL.revokeObjectURL(old);
      feedObjects[streamId] = url;
      targetImg.src = url;
      targetImg.alt = streamId === 2 ? 'Тепловизор' : 'Камера';
      targetImg.classList.remove('feed-placeholder');
    } else {
      URL.revokeObjectURL(url);
    }
  }

  // --- JSON сообщения ---
  const tempHistory = [];
  const MAX_TEMP_HISTORY = 40;

  function handleJsonMessage(msg) {
    if (msg.type === 'alert') {
      addAlertToCameraList(msg.data || msg);
      incrementBadge();
      showToast(msg.data?.message || msg.message || 'Новое оповещение');
    } else if (msg.type === 'history_load') {
      const history = Array.isArray(msg.data) ? msg.data : [];
      $('#history-count').textContent = String(history.length);
      renderHistory(history);
    } else if (msg.type === 'debug_update') {
      updateDebugInfo(msg.data);
      renderChickenList(msg.data);
      if (msg.data && typeof msg.data.flock_avg_movement === 'number') {
        const mv = msg.data.flock_avg_movement.toFixed(2);
        $('#flock-movement').textContent = mv;
        $('#dash-birds').textContent = String(Object.keys(msg.data.tracked_objects || {}).length);
      }
      const simTemp = 36 + Math.random() * 2;
      $('#avg-temp').textContent = simTemp.toFixed(1) + '°C';
      $('#dash-temp').textContent = simTemp.toFixed(1) + '°C';
      tempHistory.push(simTemp);
      if (tempHistory.length > MAX_TEMP_HISTORY) tempHistory.shift();
      drawSparkline();
    }
  }

  // --- Спарклайн ---
  function drawSparkline() {
    const canvas = $('#temp-sparkline');
    if (!canvas || tempHistory.length < 2) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const pad = 4;
    ctx.clearRect(0, 0, w, h);
    const min = 30, max = 40;
    const range = max - min;

    ctx.beginPath();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    tempHistory.forEach((v, i) => {
      const x = (i / (MAX_TEMP_HISTORY - 1)) * (w - pad * 2) + pad;
      const y = h - pad - ((v - min) / range) * (h - pad * 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.lineTo(w - pad, h);
    ctx.lineTo(pad, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(245,158,11,0.12)';
    ctx.fill();
  }

  // --- Оповещения ---
  let alertCount = 0;
  function incrementBadge() {
    alertCount++;
    $('#alert-count').textContent = String(alertCount);
    $('#camera-alert-badge').textContent = String(alertCount);
    $('#dash-alerts').textContent = String(alertCount);
    const farm = $('#farm-status');
    const dot = $('#farm-status-dot');
    if (farm && alertCount > 0) {
      farm.textContent = 'Требует внимания';
      farm.style.color = 'var(--accent-amber)';
      if (dot) { dot.classList.remove('green'); dot.classList.add('amber'); }
    }
  }

  function addAlertToCameraList(data) {
    const list = $('#camera-alerts-list');
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = 'report-item status-border-danger';
    const time = data.timestamp ? new Date(data.timestamp * 1000).toLocaleTimeString('ru-RU') : new Date().toLocaleTimeString('ru-RU');
    const demo = data.is_demo ? '<span class="demo-tag">Демо</span>' : '';
    const alertType = data.alert_type || data.type;
    const statusLabel = alertType === 'behavior_alert' ? 'ПОВЕДЕНИЕ' : (alertType === 'disease_alert' ? 'ЗАБОЛЕВАНИЕ' : 'ОПОВЕЩЕНИЕ');
    const statusClass = alertType === 'behavior_alert' ? 'status-warning' : 'status-danger';
    div.innerHTML = `<div class="report-meta">${escapeHtml(time)} ${demo}</div>
      <span class="report-status ${statusClass}">${escapeHtml(statusLabel)}</span>
      <div>${escapeHtml(data.message || '')}</div>`;
    list.prepend(div);
    while (list.children.length > 50) list.lastElementChild.remove();
  }

  function showToast(message) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    try { new Notification('Januya', { body: message }); } catch (e) {}
  }

  // --- Список куриц ---
  function renderChickenList(data) {
    const container = $('#camera-chickens-list');
    const tracked = data?.tracked_objects || {};
    const ids = Object.keys(tracked);

    if (!ids.length) {
      container.innerHTML = '<p class="empty-state">Нет отслеживаемых куриц</p>';
      return;
    }

    const frag = document.createDocumentFragment();
    ids.forEach(id => {
      const t = tracked[id];
      const div = document.createElement('div');
      div.className = 'chicken-item';

      let statusClass = 'healthy';
      let statusText = 'Здорова';
      const dz = t.dz_history || [];
      const lastDz = dz.length ? dz[dz.length - 1] : null;
      if (lastDz && lastDz !== 'Low Confidence') {
        statusClass = 'detected';
        statusText = 'Обнаружено: ' + lastDz;
      }
      // Note: we can't know active_alert from debug state, but we can infer from disease history density
      const diseaseCount = dz.filter(d => d && d !== 'Low Confidence').length;
      if (diseaseCount >= 2) {
        statusClass = 'warning';
        statusText = 'Подозрение: ' + lastDz;
      }

      const movement = typeof t.movement === 'number' ? t.movement.toFixed(2) : '--';

      div.innerHTML = `
        <span class="chicken-status ${statusClass}"></span>
        <div class="chicken-info">
          <div class="chicken-id">Курица #${escapeHtml(String(id))}</div>
          <div class="chicken-meta">
            <span class="tag">${escapeHtml(statusText)}</span>
            <span class="tag">🏃 ${escapeHtml(movement)} px/кадр</span>
            <span class="tag">🔬 ${t.clf_count || 0}/5 сканов</span>
          </div>
        </div>
      `;
      frag.appendChild(div);
    });
    container.innerHTML = '';
    container.appendChild(frag);
  }

  // --- История / Отчёты ---
  function renderHistory(items) {
    const container = $('#history-content');
    if (!items.length) { container.innerHTML = '<p class="empty-state">Нет записей в истории.</p>'; return; }
    const frag = document.createDocumentFragment();
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'report-item';
      const time = item.timestamp ? new Date(item.timestamp * 1000).toLocaleString('ru-RU') : 'Неизвестно';
      const demo = item.is_demo ? '<span class="demo-tag">Демо</span>' : '';
      const atype = item.alert_type || item.type;
      const status = atype === 'disease_alert' ? 'status-danger' : 'status-warning';
      const label = atype === 'disease_alert' ? 'ЗАБОЛЕВАНИЕ' : 'ПОВЕДЕНИЕ';
      div.innerHTML = `<div class="report-meta">${escapeHtml(time)} ${demo}</div>
        <span class="report-status ${status}">${escapeHtml(label)}</span>
        <div>${escapeHtml(item.message || '')}</div>`;
      frag.appendChild(div);
    });
    container.innerHTML = '';
    container.appendChild(frag);
  }

  function renderReports() {
    const container = $('#reports-content');
    const today = new Date().toLocaleDateString('ru-RU');
    container.innerHTML = `
      <div class="report-item">
        <div class="report-meta">${today}</div>
        <span class="report-status status-healthy">Здорово</span>
        <div>Аномальное поведение сегодня не обнаружено.</div>
      </div>
      <div class="report-item">
        <div class="report-meta">${today}</div>
        <span class="report-status status-info">Инфо</span>
        <div>Система успешно инициализирована. Все датчики онлайн.</div>
      </div>`;
  }

  // --- Вкладки ---
  function switchTab(tabId) {
    $$('.tab-pane').forEach(p => p.classList.remove('active'));
    const target = $(`#${tabId}`);
    if (target) target.classList.add('active');
    $$('.sidebar-nav .nav-item').forEach(n => n.classList.remove('active'));
    const nav = $(`.sidebar-nav .nav-item[data-tab="${tabId}"]`);
    if (nav) nav.classList.add('active');

    const titles = { 'home-pane': 'Панель управления', 'camera-pane': 'Мониторинг', 'info-pane': 'Заболевания' };
    $('#page-title').textContent = titles[tabId] || 'Januya';
  }

  // --- Оверлеи ---
  let activeOverlay = null;
  function openOverlay(id) {
    const el = $(id);
    if (!el) return;
    activeOverlay = el;
    el.classList.add('active');
    $('#overlay-backdrop').classList.add('active');
  }
  function closeOverlays() {
    if (activeOverlay) { activeOverlay.classList.remove('active'); activeOverlay = null; }
    $('#overlay-backdrop').classList.remove('active');
  }

  // --- Детали заболевания ---
  function showDiseaseDetails(id) {
    const d = DISEASES[id];
    if (!d) return;
    $('#disease-detail-title').textContent = d.name;
    $('#disease-symptoms').textContent = d.symptoms;
    $('#disease-cure').textContent = d.cure;
    $('#disease-prevention').textContent = d.prevention;
    openOverlay('#disease-details');
  }

  // --- Источник видео ---
  function cycleVideoSource() {
    const sources = ['normal_video', 'static_video', 'demo_image'];
    const idx = sources.indexOf(currentVideoSource);
    const next = sources[(idx + 1) % sources.length];

    unsubscribeFromStream(currentVideoSource);
    sendMessage({ action: 'subscribe', stream: next });

    currentVideoSource = next;
    $('#toggle-feed-btn').textContent = 'Источник: ' + sourceLabel(next);
  }

  function sourceLabel(src) {
    const map = {
      'normal_video': 'Веб-камера',
      'static_video': 'Видеофайл',
      'demo_image': 'Демо-изображение'
    };
    return map[src] || src;
  }

  // --- Демо-режим ---
  function toggleDemoMode() {
    demoMode = !demoMode;
    sendMessage({ action: 'set_source', source: currentVideoSource });
    const btn = $('#demo-mode-toggle');
    btn.textContent = demoMode ? 'Демо-режим: ВКЛ' : 'Демо-режим: ВЫКЛ';
    btn.classList.toggle('active', demoMode);
    if (!demoMode) {
      alertCount = 0;
      $('#alert-count').textContent = '0';
      $('#camera-alert-badge').textContent = '0';
      $('#dash-alerts').textContent = '0';
      const farm = $('#farm-status');
      const dot = $('#farm-status-dot');
      if (farm) { farm.textContent = 'Здорово'; farm.style.color = ''; }
      if (dot) { dot.classList.remove('amber'); dot.classList.add('green'); }
    }
  }

  // --- Переключатель Оповещения / Все курицы ---
  function switchAlertsView(view) {
    const alertsList = $('#camera-alerts-list');
    const chickensList = $('#camera-chickens-list');
    const title = $('#alerts-panel-title');

    if (view === 'alerts') {
      alertsList.style.display = '';
      chickensList.style.display = 'none';
      if (title) title.textContent = '⚠️ Оповещения';
    } else {
      alertsList.style.display = 'none';
      chickensList.style.display = '';
      if (title) title.textContent = '🐔 Все курицы';
    }

    $$('.panel-toggle-group .toggle-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === view);
    });
  }

  // --- Панель разработчика ---
  function openDevMode() {
    openOverlay('#dev-mode-overlay');
  }

  function triggerAlert(type, message) {
    sendMessage({ action: 'store_alert', alert: { type, message, is_demo: true } });
  }

  function injectDisease(diseaseName) {
    sendMessage({ action: 'inject_disease_demo', disease: diseaseName });
  }

  // --- События ---
  function initEvents() {
    // Вкладки
    $$('.sidebar-nav .nav-item[data-tab]').forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Кнопки шапки
    $('#history-btn').addEventListener('click', () => openOverlay('#history-list'));
    $('#reports-bell').addEventListener('click', () => { renderReports(); openOverlay('#reports-list'); });
    $('#theme-toggle-btn').addEventListener('click', () => {
      const html = document.documentElement;
      html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
    });

    // Панель разработчика
    $('#dev-mode-trigger').addEventListener('click', openDevMode);

    // Управление трансляцией
    $('#toggle-feed-btn').addEventListener('click', cycleVideoSource);
    $('#demo-mode-toggle').addEventListener('click', toggleDemoMode);

    // Переключатель оповещения / курицы
    $$('.panel-toggle-group .toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => switchAlertsView(btn.dataset.view));
    });

    // Карточки заболеваний
    $$('.disease-card').forEach(card => {
      card.addEventListener('click', (e) => {
        if (e.target.closest('.btn-small')) { showDiseaseDetails(card.dataset.diseaseId); }
        else { showDiseaseDetails(card.dataset.diseaseId); }
      });
    });

    // Закрытие оверлеев
    $$('.close-overlay-btn').forEach(btn => btn.addEventListener('click', closeOverlays));
    $('#overlay-backdrop').addEventListener('click', closeOverlays);

    // Триггеры оповещений в dev-панели
    $$('#alert-trigger-controls .dev-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        if (action === 'inject') {
          injectDisease(btn.dataset.disease);
        } else if (action === 'lethargy') {
          triggerAlert('behavior_alert', 'Курица #12 проявляет признаки летаргии.');
        }
      });
    });

    $('#lethargy-demo-btn').addEventListener('click', () => {
      sendMessage({ action: 'toggle_lethargy_demo' });
      $('#lethargy-demo-btn').classList.toggle('active');
    });

    // Горячая клавиша Ctrl+Shift+D для dev-панели
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        openDevMode();
      }
    });

    // Разрешение уведомлений
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().catch(() => {});
    }
  }

  // --- Инициализация ---
  function updateDebugInfo(data) {
    const pre = $('#debug-info-content');
    if (!pre) return;
    pre.textContent = JSON.stringify(data, null, 2);
  }

  document.addEventListener('DOMContentLoaded', () => {
    initEvents();
    connectWebSocket();
    renderReports();
    $('#toggle-feed-btn').textContent = 'Источник: ' + sourceLabel(currentVideoSource);
    for (let i = 0; i < MAX_TEMP_HISTORY; i++) tempHistory.push(36 + Math.random() * 2);
    drawSparkline();
  });
})();
