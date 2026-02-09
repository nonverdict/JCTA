document.addEventListener('DOMContentLoaded', () => {
    const htmlElement = document.documentElement;
    const overlayBackdrop = document.getElementById('overlay-backdrop');
    const navButtons = document.querySelectorAll('.nav-button');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const normalFeedImg = document.getElementById('normal-feed-img');
    const thermalFeedImg = document.getElementById('thermal-feed-img');
    const diseaseList = document.getElementById('disease-list');
    const diseaseDetailsPane = document.getElementById('disease-details');
    const diseaseDetailTitle = document.getElementById('disease-detail-title');
    const diseaseSymptoms = document.getElementById('disease-symptoms');
    const diseaseCure = document.getElementById('disease-cure');
    const diseasePrevention = document.getElementById('disease-prevention');
    const reportsBell = document.getElementById('reports-bell');
    const reportsListPane = document.getElementById('reports-list');
    const reportsContent = document.getElementById('reports-content');
    const toggleFeedBtn = document.getElementById('toggle-feed-btn');
    const showAllCamerasBtn = document.getElementById('show-all-cameras-btn');
    const themeToggleButton = document.getElementById('theme-toggle-btn');
    const historyBtn = document.getElementById('history-btn');
    const historyListPane = document.getElementById('history-list');
    const historyContent = document.getElementById('history-content');
    const historyCount = document.getElementById('history-count');


    // --- DEV/DEMO MODE ELEMENTS ---
    const devModeOverlay = document.getElementById('dev-mode-overlay');
    const demoModeToggle = document.getElementById('demo-mode-toggle');
    const switchToVideoBtn = document.getElementById('switch-to-video-btn');
    const devInfoPanel = document.getElementById('dev-info-panel');
    const debugInfoContent = document.getElementById('debug-info-content');
    const lethargyDemoBtn = document.getElementById('lethargy-demo-btn');
    const alertTriggerControls = document.getElementById('alert-trigger-controls');

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

    let socket = null;
    let currentVideoUrl = null;
    let currentThermalUrl = null;
    let activeTabId = document.querySelector('.tab-pane.active')?.id || 'home-pane';
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelayBase = 3000;
    let reconnectTimer = null;
    let isLiveFeedActive = true;
    let isTransitioningTabs = false;
    let isDemoModeActive = false;
    let isMulticamCombinedActive = false;
    let currentViewMode = 'normal';
    const animationDuration = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--duration-medium') || '0.35s') * 1000;
    let alertHistory = [];
    let historyError = null; // Variable to store any history loading errors


    const diseaseData = {
        coccidiosis: { name: "Кокцидиоз", symptoms: "Кровавый понос, вялость, взъерошенность оперения, снижение аппетита и веса.", cure: "Применение кокцидиостатиков. Поддерживающая терапия.", prevention: "Соблюдение гигиены, регулярная чистка, качественные корма." },
        botulism: { name: "Ботулизм", symptoms: "Прогрессирующий паралич, начинающийся с ног и переходящий на крылья, шею и веки. Другие признаки включают опущенные крылья, затрудненное дыхание и взъерошенные перья.", cure: "Промывание системы раствором мелассы или эпсомовской соли. Может потребоваться антитоксин от ветеринара. Обеспечение поддерживающего ухода.", prevention: "Содержите курятник в чистоте и сухости. Обеспечьте свежую еду и воду, своевременно убирайте мертвых животных. Избегайте источников стоячей воды." },
        newcastle: { name: "Болезнь Ньюкасла", symptoms: "Респираторные признаки, диарея, нервные явления, снижение яйценоскости.", cure: "Специфического лечения нет. Больных птиц уничтожают.", prevention: "Вакцинация. Карантин. Санитарный контроль." },
        chickenpox: { name: "Куриная оспа", symptoms: "Две формы: Сухая оспа (корочки на гребне, сережках, лице) и Влажная оспа (желтоватые образования во рту и горле, затрудненное дыхание). Может вызывать снижение яйценоскости и потерю веса.", cure: "Лечения нет, но можно управлять симптомами. Изолируйте зараженных птиц, обеспечьте поддерживающий уход и обрабатывайте поражения разбавленным раствором йода.", prevention: "Ключевым фактором является вакцинация. Соблюдайте строгую биозащиту, помещайте новых птиц в карантин и контролируйте популяцию комаров." },
        lice_and_mites: { name: "Вши и клещи", symptoms: "Видимые вши или клещи, особенно в области клоаки. Повреждение перьев, чрезмерное почесывание, грязноватые на вид перья, корочки и покраснения. Тяжелые инфестации могут вызывать анемию и потерю веса.", cure: "Обрабатывайте кур специальными порошками или спреями. Тщательно очистите и обработайте курятник, удалив и заменив всю подстилку.", prevention: "Регулярные осмотры, чистый курятник, предоставление пылевых ванн и карантин для новых птиц имеют решающее значение для профилактики." }
    };

    // --- THEME LOGIC ---
    function applyTheme(theme) {
        htmlElement.setAttribute('data-theme', theme === 'dark' ? 'dark' : 'default');
        localStorage.setItem('theme', theme);
    }
    if (themeToggleButton) {
        themeToggleButton.addEventListener('click', () => {
            const isCurrentlyDark = htmlElement.getAttribute('data-theme') === 'dark';
            applyTheme(isCurrentlyDark ? 'default' : 'dark');
        });
    }
    applyTheme(localStorage.getItem('theme') || 'default');

    // --- WEBSOCKET LOGIC ---
    function connectWebSocket() {
        if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        setVideoFeedStatus("Подключение WebSocket...");
        
        try { socket = new WebSocket(wsUrl); } 
        catch (e) { console.error("WebSocket instantiation error:", e); setVideoFeedStatus("Ошибка WebSocket URL."); return; }

        socket.binaryType = 'arraybuffer';

        socket.onopen = () => {
            console.log("[WS] Connection established.");
            reconnectAttempts = 0;
            // Clear transient reports and bell on successful reconnect
            reportsContent.innerHTML = '<p style="text-align:center; padding: 20px; color: var(--text-placeholder);">Нет доступных отчетов.</p>';
            if(reportsBell) reportsBell.classList.remove('has-new-report');

            if (activeTabId === 'camera-pane') {
                subscribeToCameraFeeds();
            }
        };

        socket.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                if (activeTabId !== 'camera-pane') return;
                const view = new Uint8Array(event.data);
                if (view.length < 2) return;
                const streamId = view[0];
                const imageData = event.data.slice(1);
                const newUrl = URL.createObjectURL(new Blob([imageData], { type: 'image/jpeg' }));

                if (streamId === 1) { // Normal Video
                    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
                    normalFeedImg.src = newUrl;
                    currentVideoUrl = newUrl;
                    normalFeedImg.classList.remove('feed-placeholder');
                } else if (streamId === 2) { // Thermal Video
                    if (currentThermalUrl) URL.revokeObjectURL(currentThermalUrl);
                    thermalFeedImg.src = newUrl;
                    currentThermalUrl = newUrl;
                    thermalFeedImg.classList.remove('feed-placeholder');
                } else if (streamId === 5) { // Multicam individual stream
                    const cameraIdx = view[1];
                    console.log(`Received frame from camera ${cameraIdx}`);
                    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
                    normalFeedImg.src = newUrl;
                    currentVideoUrl = newUrl;
                    normalFeedImg.classList.remove('feed-placeholder');
                } else if (streamId === 6) { // Multicam Combined View
                    if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
                    normalFeedImg.src = newUrl;
                    currentVideoUrl = newUrl;
                    normalFeedImg.classList.remove('feed-placeholder');
                }
            } else if (typeof event.data === 'string') {
                 try {
                    const data = JSON.parse(event.data);
                    if (data.type === "history_load") {
                        if (Array.isArray(data.data)) {
                            alertHistory = data.data;
                            historyError = null;
                        } else {
                            console.error("Received malformed history data:", data.data);
                            alertHistory = [];
                            historyError = "Ошибка: Неверный формат истории получен от сервера.";
                        }
                        populateHistoryPane(); // Populate history pane initially so count is correct
                    } else if (data.type === "disease_alert" || data.type === "behavior_alert") {
                        addAlertToReports(data);
                        historyError = null; // A new alert means history is working.
                        alertHistory.unshift(data); // Add to our local history state
                        if(historyCount) historyCount.textContent = alertHistory.length;
                    }
                     else if (data.type === "debug_update") {
                        updateDebugInfo(data);
                    }
                     else if (data.type === "multicam_available_cameras") {
                        console.log("Available cameras:", data.cameras);
                        if (data.cameras && data.cameras.length > 0) {
                            // Show camera selection UI
                            sendWebSocketMessage({ action: "subscribe", stream: `multicam_${data.cameras[0]}` });
                        }
                    }
                    else if (data.type === "multicam_ready") {
                        console.log("Multicam ready, cameras:", data.cameras);
                        if (data.cameras && data.cameras.length > 0) {
                            sendWebSocketMessage({ action: "subscribe", stream: `multicam_${data.cameras[0]}` });
                        }
                    }
                    else if (data.type === "multicam_combined_ready") {
                        console.log("Multicam combined ready, subscribing...");
                        sendWebSocketMessage({ action: "subscribe", stream: "multicam_combined" });
                    }
                } catch (e) { console.warn("Received non-JSON message or failed to parse:", event.data, e); }
            }
        };

        socket.onerror = (error) => { 
            console.error("[WS] WebSocket error:", error);
            setVideoFeedStatus("Ошибка WebSocket.");
        };

        socket.onclose = (event) => {
            console.log(`[WS] Connection closed: Code=${event.code}`);
            socket = null;
            if (currentVideoUrl) { URL.revokeObjectURL(currentVideoUrl); currentVideoUrl = null; }
            if (currentThermalUrl) { URL.revokeObjectURL(currentThermalUrl); currentThermalUrl = null; }
            setVideoFeedStatus(`WebSocket отключен (Код: ${event.code})`);
            
            if (event.code !== 1000 && event.code !== 1005 && reconnectAttempts < maxReconnectAttempts) {
                reconnectAttempts++;
                const delay = reconnectDelayBase * Math.pow(1.5, reconnectAttempts - 1);
                setVideoFeedStatus(`Переподключение #${reconnectAttempts}...`);
                reconnectTimer = setTimeout(connectWebSocket, delay);
            } else if (reconnectAttempts >= maxReconnectAttempts) {
                setVideoFeedStatus("Не удалось подключиться.");
            }
        };
    }

    function sendWebSocketMessage(message) {
        if (socket && socket.readyState === WebSocket.OPEN) {
            try { socket.send(JSON.stringify(message)); } 
            catch (e) { console.error("[WS] Error sending message:", e); }
        }
    }

    function setVideoFeedStatus(statusText) {
        if (normalFeedImg && (!normalFeedImg.src || normalFeedImg.classList.contains('feed-placeholder'))) {
            normalFeedImg.alt = statusText;
            if (!normalFeedImg.classList.contains('feed-placeholder')) {
                normalFeedImg.classList.add('feed-placeholder');
            }
        }
    }
    
    // --- UI & NAVIGATION ---
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            if (isTransitioningTabs) return;
            const targetPaneId = button.getAttribute('data-tab');
            if (button.classList.contains('active') || !targetPaneId) return;

            isTransitioningTabs = true;
            const currentActivePane = document.querySelector('.tab-pane.active');
            const targetPane = document.getElementById(targetPaneId);

            navButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            if (currentActivePane) currentActivePane.classList.add('exiting');
            
            if (targetPane) {
                targetPane.classList.remove('exiting');
                targetPane.scrollTop = 0;
                targetPane.classList.add('active');
                animateTabContent(targetPaneId);
            }
            
            const oldActiveTabId = activeTabId;
            activeTabId = targetPaneId;
            
            if (targetPaneId === 'camera-pane' && oldActiveTabId !== 'camera-pane') {
                subscribeToCameraFeeds();
            }
            else if (targetPaneId !== 'camera-pane' && oldActiveTabId === 'camera-pane') {
                unsubscribeFromCameraFeeds();
            }
            
            hideAllOverlays();

            setTimeout(() => {
                if(currentActivePane) currentActivePane.classList.remove('active', 'exiting');
                isTransitioningTabs = false;
            }, animationDuration);
        });
    });

    function animateTabContent(paneId) {
        const pane = document.getElementById(paneId);
        if (!pane) return;
        const elementsToAnimate = pane.querySelectorAll('.animatable-on-tab-load');
        elementsToAnimate.forEach((el, index) => {
            el.style.animation = 'none'; void el.offsetWidth;
            el.style.animation = `fadeInSlideUp 0.6s var(--transition-swift) ${0.1 + index * 0.1}s forwards`;
        });
        
        if (paneId === 'info-pane' && diseaseList) {
            const items = diseaseList.querySelectorAll('.disease-item');
            items.forEach((item, index) => {
                item.style.animation = 'none';
                item.style.opacity = '0'; // Start invisible
                void item.offsetWidth; // Trigger reflow
                item.style.animation = `itemPopIn var(--duration-medium) var(--transition-bounce) forwards`;
                item.style.animationDelay = `${0.2 + index * 0.07}s`;
                item.addEventListener('animationend', () => item.classList.add('animation-finished'), { once: true });
            });
        }
    }

    function showOverlay(overlayElement) {
        if (!overlayElement) return;
        hideAllOverlays();
        overlayElement.classList.add('active');
        if (overlayBackdrop && window.innerWidth >= 768) overlayBackdrop.classList.add('active');
    }
    
    function hideAllOverlays() {
        document.querySelectorAll('.details-overlay.active').forEach(el => el.classList.remove('active'));
        if (overlayBackdrop) overlayBackdrop.classList.remove('active');
    }

    function setupOverlayClosers() {
        document.body.addEventListener('click', (event) => {
            if (event.target.closest('.close-overlay-btn') || event.target.id === 'overlay-backdrop') {
                hideAllOverlays();
            }
        });
        document.addEventListener('keydown', (event) => { if (event.key === 'Escape') hideAllOverlays(); });
    }
    
    if (diseaseList) {
        diseaseList.addEventListener('click', (event) => {
            const listItem = event.target.closest('.disease-item');
            if (!listItem) return;
            const diseaseId = listItem.getAttribute('data-disease-id');
            const data = diseaseData[diseaseId];
            if (data && diseaseDetailsPane) {
                diseaseDetailTitle.textContent = data.name;
                diseaseSymptoms.textContent = data.symptoms;
                diseaseCure.textContent = data.cure;
                diseasePrevention.textContent = data.prevention;
                showOverlay(diseaseDetailsPane);
            }
        });
    }

    if (reportsBell) {
        reportsBell.addEventListener('click', () => {
            showOverlay(reportsListPane);
            reportsBell.classList.remove('has-new-report');
        });
    }

    if (historyBtn) {
        historyBtn.addEventListener('click', () => {
            populateHistoryPane(); // Re-populate to ensure it's up-to-date
            showOverlay(historyListPane);
        });
    }

    // --- STREAM SUBSCRIPTION LOGIC ---
    function subscribeToCameraFeeds() {
        unsubscribeFromCameraFeeds();

        if (isMulticamCombinedActive) {
            setVideoFeedStatus(`Запрос multicam_combined потока...`);
            sendWebSocketMessage({ action: "set_source", source: "multicam_combined" });
            // Backend will send multicam_available_cameras, then we'll subscribe
        } else {
            const streamName = isDemoModeActive ? "demo_image" : (isLiveFeedActive ? "normal_video" : "static_video");
            setVideoFeedStatus(`Запрос ${streamName} потока...`);
            sendWebSocketMessage({ action: "set_source", source: streamName });
            
            // For live video, we'll use multicam system
            if (isLiveFeedActive && !isDemoModeActive) {
                // Subscribe to first available multicam stream
                sendWebSocketMessage({ action: "subscribe", stream: "multicam_1" });
            } else {
                const videoStreamToSubscribe = isLiveFeedActive ? "normal_video" : "static_video";
                sendWebSocketMessage({ action: "subscribe", stream: videoStreamToSubscribe });
            }
        }
        
        sendWebSocketMessage({ action: "subscribe", stream: "thermal_video" });
        if (isDemoModeActive) sendWebSocketMessage({ action: "subscribe", stream: "debug_stream" });

        if (!socket || socket.readyState !== WebSocket.OPEN) {
            connectWebSocket();
        }
    }

    function unsubscribeFromCameraFeeds() {
        sendWebSocketMessage({ action: "unsubscribe", stream: "normal_video" });
        sendWebSocketMessage({ action: "unsubscribe", stream: "static_video" });
        sendWebSocketMessage({ action: "unsubscribe", stream: "thermal_video" });
        sendWebSocketMessage({ action: "unsubscribe", stream: "debug_stream" });
        sendWebSocketMessage({ action: "unsubscribe", stream: "multicam_combined" });
        // Unsubscribe from all multicam streams
        for (let i = 0; i <= 10; i++) {
            sendWebSocketMessage({ action: "unsubscribe", stream: `multicam_${i}` });
        }
        [normalFeedImg, thermalFeedImg].forEach(img => {
            if (img && img.src) { 
                img.src = ""; 
                img.classList.add('feed-placeholder'); 
            }
        });
        if (currentVideoUrl) { URL.revokeObjectURL(currentVideoUrl); currentVideoUrl = null; }
        if (currentThermalUrl) { URL.revokeObjectURL(currentThermalUrl); currentThermalUrl = null; }
    }

    // --- REPORTING & HISTORY LOGIC ---
    function addAlertToReports(alertData) {
        const status = alertData.type === 'disease_alert' ? 'ОПАСНОСТЬ' : 'ВНИМАНИЕ';
        const statusClass = alertData.type === 'disease_alert' ? 'status-danger' : 'status-warning';
        
        addReportToDOM(reportsContent, {
            timestamp: new Date(alertData.timestamp * 1000).toLocaleString('ru-RU'),
            details: alertData.message,
            status: status,
            statusClass: statusClass,
            isDemo: alertData.is_demo || false
        });
        if (reportsBell) reportsBell.classList.add('has-new-report');
    }
    
    function addReportToDOM(container, reportData) {
        if (!container) return;
        const reportDiv = document.createElement('div');
        reportDiv.className = 'report-item';
        
        let demoLabel = reportData.isDemo ? '<span class="demo-tag">DEMO</span>' : '';
        
        reportDiv.innerHTML = `
            <div class="report-meta">${reportData.timestamp} ${demoLabel}</div>
            <div><span class="report-status ${reportData.statusClass}">${reportData.status}</span></div>
            <div>${reportData.details}</div>
        `;
        
        const placeholder = container.querySelector('p');
        if (placeholder) placeholder.remove();
        
        container.prepend(reportDiv);
    }

    function populateHistoryPane() {
        if (!historyContent || !historyCount) return;

        historyContent.innerHTML = ''; // Clear previous entries
        
        if (historyError) {
            historyCount.textContent = 'Ошибка';
            historyContent.innerHTML = `<p style="text-align:center; padding: 20px; color: var(--status-danger-text);">${historyError}</p>`;
            return;
        }

        historyCount.textContent = alertHistory.length;

        if (alertHistory.length === 0) {
            historyContent.innerHTML = '<p style="text-align:center; padding: 20px; color: var(--text-placeholder);">Нет записей в истории.</p>';
            return;
        }

        for (const alertData of alertHistory) {
            const status = alertData.type === 'disease_alert' ? 'ОПАСНОСТЬ' : 'ВНИМАНИЕ';
            const statusClass = alertData.type === 'disease_alert' ? 'status-danger' : 'status-warning';
            
            addReportToDOM(historyContent, {
                timestamp: new Date(alertData.timestamp * 1000).toLocaleString('ru-RU'),
                details: alertData.message,
                status: status,
                statusClass: statusClass,
                isDemo: alertData.is_demo || false
            });
        }
    }


    // --- DEV & DEMO MODE LOGIC ---
    function updateDebugInfo(data) {
        if (!data || !debugInfoContent) return;
    
        const avgMovement = data.flock_avg_movement?.toFixed(2) ?? 'N/A';
        let content = `Flock Avg Movement: ${avgMovement}\n\n`;
        content += "--- Tracked Chickens ---\n";
    
        if (data.tracked_objects && Object.keys(data.tracked_objects).length > 0) {
            for (const [id, info] of Object.entries(data.tracked_objects)) {
                const movement = info.movement?.toFixed(2) ?? 'N/A';
                const history = JSON.stringify(info.dz_history) ?? '[]';
                const clfCount = info.clf_count ?? 'N/A'; // Handle missing clf_count
                content += `ID #${id}: Move=${movement}, History=${history}, CLF=${clfCount}\n`;
            }
        } else {
            content += "No chickens currently tracked.\n";
        }
    
        debugInfoContent.textContent = content;
    }

    function updateFeedToggleButton() {
        if (toggleFeedBtn) {
            toggleFeedBtn.textContent = isLiveFeedActive ? 'К статическому видео' : 'К живой камере';
        }
    }

    if (toggleFeedBtn) {
        toggleFeedBtn.addEventListener('click', () => {
            isLiveFeedActive = !isLiveFeedActive;
            isMulticamCombinedActive = false;
            currentViewMode = 'normal';
            updateFeedToggleButton();
            updateShowAllCamerasButton();
            if (activeTabId === 'camera-pane') {
                subscribeToCameraFeeds();
            }
        });
    }

    if (showAllCamerasBtn) {
        showAllCamerasBtn.addEventListener('click', () => {
            isMulticamCombinedActive = !isMulticamCombinedActive;
            currentViewMode = isMulticamCombinedActive ? 'combined' : 'normal';
            updateShowAllCamerasButton();
            updateFeedToggleButton();
            if (activeTabId === 'camera-pane') {
                subscribeToCameraFeeds();
            }
        });
    }

    function updateShowAllCamerasButton() {
        if (showAllCamerasBtn) {
            showAllCamerasBtn.classList.toggle('active', isMulticamCombinedActive);
            showAllCamerasBtn.textContent = isMulticamCombinedActive ? 'Одна камера' : 'Все камеры';
        }
    }

    if (demoModeToggle) {
        demoModeToggle.addEventListener('change', () => {
            isDemoModeActive = demoModeToggle.checked;
            isMulticamCombinedActive = false;
            currentViewMode = 'normal';
            updateShowAllCamerasButton();
            updateFeedToggleButton();
            if (devInfoPanel) devInfoPanel.style.display = isDemoModeActive ? '' : 'none';
            if (switchToVideoBtn) switchToVideoBtn.style.display = isDemoModeActive ? '' : 'none';
            
            if (activeTabId === 'camera-pane') {
                subscribeToCameraFeeds();
            }
        });
    }

    if (switchToVideoBtn) {
        switchToVideoBtn.addEventListener('click', () => {
            // This button is only for demo mode to switch the source
            sendWebSocketMessage({ action: "set_source", source: "static_video" });
        });
    }

    if (alertTriggerControls) {
        alertTriggerControls.addEventListener('click', (event) => {
            const button = event.target.closest('.dev-button');
            if (!button) return;

            const alertType = button.dataset.alertType === 'disease' ? 'disease_alert' : 'behavior_alert';
            const message = button.dataset.message;
            const timestamp = Date.now() / 1000;

            sendWebSocketMessage({
                action: "store_alert",
                alert: { type: alertType, message: message, timestamp: timestamp }
            });
            showOverlay(reportsListPane);
        });
    }

    if (lethargyDemoBtn) {
        lethargyDemoBtn.addEventListener('click', () => {
            sendWebSocketMessage({ action: "toggle_lethargy_demo" });
            lethargyDemoBtn.classList.toggle('active');
            lethargyDemoBtn.textContent = lethargyDemoBtn.classList.contains('active') ? 'Deactivate' : 'Activate';
        });
    }

    // --- INITIALIZATION ---
    setupOverlayClosers();
    updateFeedToggleButton();
    connectWebSocket();
    animateTabContent(activeTabId);

    // Dev Panel Activation
    const infoNavButton = document.querySelector('.nav-button[data-tab="info-pane"]');
    let devClickCount = 0;
    let devClickTimer = null;
    if (infoNavButton) {
        infoNavButton.addEventListener('click', () => {
            if (devModeOverlay && devModeOverlay.classList.contains('active')) return;
            devClickCount++;
            if (devClickTimer) clearTimeout(devClickTimer);
            if (devClickCount >= 3) {
                showOverlay(devModeOverlay);
                devClickCount = 0;
            } else {
                devClickTimer = setTimeout(() => { devClickCount = 0; }, 1000);
            }
        });
    }
});
