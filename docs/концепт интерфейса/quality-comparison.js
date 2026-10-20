"use strict";

(() => {
  const state = {
    catalog: [],
    comparison: null,
    intervalIndex: -1,
    blind: false,
    revealed: false,
    blindOrder: ["before", "after"],
    activeSide: null,
    syncing: false,
  };

  const elements = {
    title: document.querySelector("#page-title"),
    description: document.querySelector("#page-description"),
    select: document.querySelector("#comparison-select"),
    status: document.querySelector("#status"),
    content: document.querySelector("#comparison-content"),
    empty: document.querySelector("#empty-state"),
    summary: document.querySelector("#summary"),
    blindToggle: document.querySelector("#blind-toggle"),
    reveal: document.querySelector("#reveal-button"),
    activeState: document.querySelector("#active-state"),
    intervalCount: document.querySelector("#interval-count"),
    intervalList: document.querySelector("#interval-list"),
    intervalTime: document.querySelector("#interval-time"),
    intervalTitle: document.querySelector("#interval-title"),
    intervalReason: document.querySelector("#interval-reason"),
    confidence: document.querySelector("#confidence"),
    metrics: document.querySelector("#metrics"),
    leftCard: document.querySelector('[data-player-card="left"]'),
    rightCard: document.querySelector('[data-player-card="right"]'),
    leftLabel: document.querySelector("#left-label"),
    rightLabel: document.querySelector("#right-label"),
    leftBadge: document.querySelector("#left-badge"),
    rightBadge: document.querySelector("#right-badge"),
    leftAudio: document.querySelector("#left-audio"),
    rightAudio: document.querySelector("#right-audio"),
    leftMissing: document.querySelector("#left-missing"),
    rightMissing: document.querySelector("#right-missing"),
    playLeft: document.querySelector("#play-left"),
    playRight: document.querySelector("#play-right"),
    pauseAll: document.querySelector("#pause-all"),
  };

  const metricLabels = {
    input_dbfs: "До обработки",
    before_dbfs: "До исправления",
    after_dbfs: "После исправления",
    loss_db: "Потеря",
    restored_db: "Восстановлено",
    english_margin_db: "Запас EN",
    speech_band_loss_db: "Речевые частоты",
    duration_sec: "Длительность",
  };

  function setStatus(message, isError = false) {
    elements.status.textContent = message || "";
    elements.status.classList.toggle("error", isError);
  }

  function formatTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = Math.floor(value % 60);
    const millis = Math.round((value - Math.floor(value)) * 100);
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(millis).padStart(2, "0")}`;
  }

  function formatMetric(key, value) {
    if (typeof value === "number") {
      const digits = Math.abs(value) >= 100 ? 0 : 2;
      const suffix = key.endsWith("_db") || key.endsWith("_dbfs") ? " дБ" : key.endsWith("_sec") ? " с" : "";
      return `${value.toLocaleString("ru-RU", { maximumFractionDigits: digits })}${suffix}`;
    }
    if (typeof value === "boolean") return value ? "да" : "нет";
    return String(value ?? "");
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function appendDefinition(container, className, name, value) {
    const wrapper = document.createElement("div");
    wrapper.className = className;
    const term = document.createElement("dt");
    term.textContent = name;
    const detail = document.createElement("dd");
    detail.textContent = value;
    wrapper.append(term, detail);
    container.append(wrapper);
  }

  function stopPlayers(reset = false) {
    for (const audio of [elements.leftAudio, elements.rightAudio]) {
      audio.pause();
      if (reset) {
        try { audio.currentTime = 0; } catch (_error) { /* metadata is not loaded yet */ }
      }
    }
    state.activeSide = null;
    elements.leftCard.classList.remove("active");
    elements.rightCard.classList.remove("active");
    elements.activeState.textContent = "Воспроизведение остановлено";
  }

  function currentInterval() {
    return state.comparison?.intervals?.[state.intervalIndex] || null;
  }

  function sourceForKind(interval, kind) {
    return kind === "before" ? interval.before_audio_url : interval.after_audio_url;
  }

  function labelsForKind(kind) {
    return kind === "before" ? "До исправления" : "После исправления";
  }

  function chooseBlindOrder() {
    const random = new Uint32Array(1);
    if (window.crypto?.getRandomValues) window.crypto.getRandomValues(random);
    else random[0] = Math.floor(Math.random() * 0xffffffff);
    state.blindOrder = random[0] % 2 ? ["before", "after"] : ["after", "before"];
  }

  function displayedOrder() {
    return state.blind ? state.blindOrder : ["before", "after"];
  }

  function updatePlayerLabels() {
    const order = displayedOrder();
    const hidden = state.blind && !state.revealed;
    elements.leftLabel.textContent = hidden ? "Вариант A" : labelsForKind(order[0]);
    elements.rightLabel.textContent = hidden ? "Вариант B" : labelsForKind(order[1]);
    elements.leftBadge.textContent = "A";
    elements.rightBadge.textContent = "B";
    elements.playLeft.textContent = `▶ ${hidden ? "Вариант A" : labelsForKind(order[0])}`;
    elements.playRight.textContent = `▶ ${hidden ? "Вариант B" : labelsForKind(order[1])}`;
    elements.reveal.hidden = !state.blind;
    elements.reveal.textContent = state.revealed ? "Скрыть варианты" : "Показать варианты";
  }

  function setAudioSource(audio, missing, source) {
    audio.pause();
    audio.removeAttribute("src");
    if (source) audio.src = source;
    audio.hidden = !source;
    missing.hidden = Boolean(source);
    audio.load();
  }

  function renderPlayers(interval) {
    stopPlayers(true);
    const order = displayedOrder();
    setAudioSource(elements.leftAudio, elements.leftMissing, sourceForKind(interval, order[0]));
    setAudioSource(elements.rightAudio, elements.rightMissing, sourceForKind(interval, order[1]));
    elements.playLeft.disabled = !elements.leftAudio.src;
    elements.playRight.disabled = !elements.rightAudio.src;
    elements.pauseAll.disabled = !elements.leftAudio.src && !elements.rightAudio.src;
    updatePlayerLabels();
  }

  function renderMetrics(interval) {
    clearNode(elements.metrics);
    Object.entries(interval.metrics || {}).forEach(([key, value]) => {
      const label = metricLabels[key] || key.replaceAll("_", " ");
      appendDefinition(elements.metrics, "metric", label, formatMetric(key, value));
    });
  }

  function renderInterval(index) {
    const intervals = state.comparison?.intervals || [];
    if (!intervals[index]) return;
    state.intervalIndex = index;
    state.revealed = false;
    chooseBlindOrder();
    const interval = intervals[index];

    elements.intervalTime.textContent = `${formatTime(interval.start_sec)} — ${formatTime(interval.end_sec)}`;
    elements.intervalTitle.textContent = interval.decision || `Интервал ${index + 1}`;
    elements.intervalReason.textContent = interval.reason || "Причина не указана.";
    const confidence = interval.confidence;
    elements.confidence.hidden = confidence === null || confidence === undefined || confidence === "";
    elements.confidence.textContent = typeof confidence === "number"
      ? `Уверенность ${Math.round(confidence <= 1 ? confidence * 100 : confidence)}%`
      : String(confidence || "");
    renderMetrics(interval);
    renderPlayers(interval);

    elements.intervalList.querySelectorAll(".interval-button").forEach((button, buttonIndex) => {
      button.classList.toggle("active", buttonIndex === index);
      button.setAttribute("aria-current", buttonIndex === index ? "true" : "false");
    });
  }

  function renderIntervalList() {
    clearNode(elements.intervalList);
    const intervals = state.comparison?.intervals || [];
    elements.intervalCount.textContent = String(intervals.length);
    intervals.forEach((interval, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "interval-button";
      const time = document.createElement("span");
      time.className = "interval-time";
      time.textContent = `${formatTime(interval.start_sec)} — ${formatTime(interval.end_sec)}`;
      const summary = document.createElement("span");
      summary.className = "interval-summary";
      summary.textContent = interval.reason || interval.decision || `Интервал ${index + 1}`;
      button.append(time, summary);
      if (!interval.audio_ready) {
        const unavailable = document.createElement("span");
        unavailable.className = "unavailable";
        unavailable.textContent = "Аудио не готово";
        button.append(unavailable);
      }
      button.addEventListener("click", () => renderInterval(index));
      elements.intervalList.append(button);
    });
  }

  function renderSummary() {
    clearNode(elements.summary);
    appendDefinition(elements.summary, "summary-item", "Интервалов", String(state.comparison.intervals.length));
    Object.entries(state.comparison.summary || {}).forEach(([key, value]) => {
      appendDefinition(elements.summary, "summary-item", metricLabels[key] || key.replaceAll("_", " "), formatMetric(key, value));
    });
  }

  function renderComparison() {
    const comparison = state.comparison;
    elements.title.textContent = comparison.title || "Сравнение результата";
    elements.description.textContent = comparison.description || "Выберите интервал и сравните звук до и после исправления.";
    elements.empty.hidden = true;
    elements.content.hidden = false;
    renderSummary();
    renderIntervalList();
    if (comparison.intervals.length) renderInterval(0);
    else {
      elements.content.hidden = true;
      elements.empty.hidden = false;
    }
    setStatus(comparison.generated_at ? `Подготовлено: ${comparison.generated_at}` : "");
  }

  function renderEmpty() {
    stopPlayers(true);
    elements.content.hidden = true;
    elements.empty.hidden = false;
    elements.title.textContent = "Сравнение результата";
    elements.description.textContent = "Здесь появятся фрагменты до и после исправления.";
    setStatus("");
  }

  async function loadComparison(comparisonId) {
    setStatus("Загрузка сравнения…");
    try {
      const response = await fetch(`/api/quality-comparisons/${encodeURIComponent(comparisonId)}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Не удалось загрузить сравнение (${response.status}).`);
      state.comparison = await response.json();
      const url = new URL(window.location.href);
      url.searchParams.set("id", comparisonId);
      window.history.replaceState(null, "", url);
      renderComparison();
    } catch (error) {
      state.comparison = null;
      renderEmpty();
      setStatus(error.message || "Не удалось загрузить сравнение.", true);
    }
  }

  async function loadCatalog() {
    try {
      const response = await fetch("/api/quality-comparisons", { cache: "no-store" });
      if (!response.ok) throw new Error(`Не удалось получить список сравнений (${response.status}).`);
      const payload = await response.json();
      state.catalog = Array.isArray(payload.comparisons) ? payload.comparisons : [];
      clearNode(elements.select);
      if (!state.catalog.length) {
        const option = document.createElement("option");
        option.textContent = "Нет готовых сравнений";
        elements.select.append(option);
        elements.select.disabled = true;
        renderEmpty();
        return;
      }
      state.catalog.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.title} (${item.audio_ready_count}/${item.interval_count})`;
        elements.select.append(option);
      });
      elements.select.disabled = false;
      const requested = new URLSearchParams(window.location.search).get("id");
      const selected = state.catalog.find((item) => item.id === requested) || state.catalog[0];
      elements.select.value = selected.id;
      await loadComparison(selected.id);
    } catch (error) {
      elements.select.disabled = true;
      renderEmpty();
      setStatus(error.message || "Не удалось загрузить данные.", true);
    }
  }

  function switchPlayback(side) {
    const target = side === "left" ? elements.leftAudio : elements.rightAudio;
    const other = side === "left" ? elements.rightAudio : elements.leftAudio;
    if (!target.src) return;
    const sourceTime = state.activeSide ? other.currentTime : Math.max(target.currentTime, other.currentTime);
    other.pause();
    if (Number.isFinite(sourceTime) && Math.abs(target.currentTime - sourceTime) > 0.06) {
      try { target.currentTime = Math.min(sourceTime, Number.isFinite(target.duration) ? target.duration : sourceTime); } catch (_error) { /* wait for metadata */ }
    }
    state.activeSide = side;
    elements.leftCard.classList.toggle("active", side === "left");
    elements.rightCard.classList.toggle("active", side === "right");
    elements.activeState.textContent = `${side === "left" ? elements.leftLabel.textContent : elements.rightLabel.textContent} · ${formatTime(target.currentTime)}`;
    target.play().catch(() => {
      elements.activeState.textContent = "Браузер не разрешил воспроизведение.";
    });
  }

  function mirrorPosition(source, target) {
    if (state.syncing || !Number.isFinite(source.currentTime) || !target.src) return;
    if (Math.abs(target.currentTime - source.currentTime) < 0.18) return;
    state.syncing = true;
    try { target.currentTime = source.currentTime; } catch (_error) { /* metadata is not loaded */ }
    state.syncing = false;
  }

  elements.select.addEventListener("change", () => loadComparison(elements.select.value));
  elements.blindToggle.addEventListener("change", () => {
    state.blind = elements.blindToggle.checked;
    state.revealed = false;
    chooseBlindOrder();
    const interval = currentInterval();
    if (interval) renderPlayers(interval);
  });
  elements.reveal.addEventListener("click", () => {
    state.revealed = !state.revealed;
    updatePlayerLabels();
  });
  elements.playLeft.addEventListener("click", () => switchPlayback("left"));
  elements.playRight.addEventListener("click", () => switchPlayback("right"));
  elements.pauseAll.addEventListener("click", () => stopPlayers(false));

  [["left", elements.leftAudio, elements.rightAudio], ["right", elements.rightAudio, elements.leftAudio]].forEach(([side, audio, other]) => {
    audio.addEventListener("play", () => {
      if (state.activeSide !== side) switchPlayback(side);
    });
    audio.addEventListener("seeking", () => mirrorPosition(audio, other));
    audio.addEventListener("timeupdate", () => {
      if (state.activeSide === side) {
        mirrorPosition(audio, other);
        elements.activeState.textContent = `${side === "left" ? elements.leftLabel.textContent : elements.rightLabel.textContent} · ${formatTime(audio.currentTime)}`;
      }
    });
    audio.addEventListener("ended", () => stopPlayers(true));
  });

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.key.toLowerCase() === "a") switchPlayback("left");
    if (event.key.toLowerCase() === "b") switchPlayback("right");
    if (event.key === " ") {
      event.preventDefault();
      stopPlayers(false);
    }
  });

  loadCatalog();
})();
