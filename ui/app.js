(() => {
  "use strict";

  const config = window.PATHSNAP_CONFIG || {};
  const STORAGE_KEY = "pathsnap.apiBaseUrl";
  const TIMEOUT_MS = config.requestTimeoutMs || 60000;
  const GPS_THRESHOLD = config.gpsPointThreshold || 3000;
  const MAX_LOG_ENTRIES = 8;
  const $ = (id) => document.getElementById(id);

  const els = {
    apiAddress: $("api-address"), connectionDot: $("connection-dot"), connectionLabel: $("connection-label"),
    tripCount: $("trip-count"), tripState: $("trip-state"), tripList: $("trip-list"), selectedSummary: $("selected-summary"),
    runButton: $("run-button"), runState: $("run-state"), mapEmpty: $("map-empty"), mapNotice: $("map-notice"),
    mapLegend: $("map-legend"), resultTrip: $("result-trip"), resultsPlaceholder: $("results-placeholder"),
    resultsContent: $("results-content"), settingsDialog: $("settings-dialog"), settingsForm: $("settings-form"),
    apiUrl: $("api-url"), healthResult: $("health-result"), requestLog: $("request-log-list"), logCount: $("log-count"),
  };

  let apiBaseUrl = readSavedBaseUrl();
  let trips = [];
  let selectedTripId = null;
  let activeController = null;
  let activeRequestId = 0;
  let map = null;
  let tileLayer = null;
  let gpsLayer = null;
  let classifierLayer = null;
  let hmmLayer = null;
  let layerControl = null;
  let resultCount = 0;
  const requestLog = [];

  function readSavedBaseUrl() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return normalizeBaseUrl(saved || config.defaultApiBaseUrl || "http://localhost:8000");
    } catch (_) {
      return normalizeBaseUrl(config.defaultApiBaseUrl || "http://localhost:8000");
    }
  }

  function normalizeBaseUrl(value) {
    return String(value || "").trim().replace(/\/+$/, "");
  }

  function setConnection(status, label) {
    els.connectionDot.className = `connection-dot ${status}`;
    els.connectionLabel.textContent = label;
  }

  function updateApiAddress() {
    els.apiAddress.textContent = apiBaseUrl;
    els.apiAddress.title = apiBaseUrl;
  }

  function setTripState(message, isError = false, retry = false) {
    els.tripState.replaceChildren(document.createTextNode(message));
    els.tripState.classList.toggle("error", isError);
    if (retry) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.retry = "true";
      button.className = "button button-secondary";
      button.textContent = "Retry";
      button.style.marginTop = "8px";
      button.style.padding = "6px 9px";
      button.style.fontSize = "10px";
      els.tripState.append(document.createElement("br"), button);
    }
  }

  function setRunState(message = "", isError = false, loading = false) {
    els.runState.classList.toggle("error", isError);
    els.runState.replaceChildren();
    if (loading) {
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      spinner.setAttribute("aria-hidden", "true");
      els.runState.append(spinner);
    }
    if (message) els.runState.append(document.createTextNode(message));
  }

  function endpoint(path) {
    return `${apiBaseUrl}${path}`;
  }

  function fetchJson(url, options = {}, timeoutMs = TIMEOUT_MS) {
    const controller = options.controller || new AbortController();
    const timer = window.setTimeout(() => controller.abort("timeout"), timeoutMs);
    const fetchOptions = { ...options, signal: controller.signal };
    delete fetchOptions.controller;
    return fetch(url, fetchOptions)
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          const error = new Error(data.detail || `API returned HTTP ${response.status}`);
          error.status = response.status;
          error.code = data.code;
          throw error;
        }
        return data;
      })
      .finally(() => window.clearTimeout(timer));
  }

  async function loadTrips() {
    setTripState("Loading trips…");
    setConnection("", "Connecting…");
    try {
      const data = await fetchJson(endpoint("/trips"), {}, 10000);
      if (!Array.isArray(data)) throw new Error("The API returned an invalid trip list.");
      trips = data.filter((trip) => trip && typeof trip.trip_id === "string");
      renderTrips();
      if (trips.length === 0) {
        setTripState("The API returned no trips.", true);
        setConnection("online", "API online");
        return;
      }
      setTripState(`${trips.length} trip${trips.length === 1 ? "" : "s"} available`);
      setConnection("online", "API online");
    } catch (error) {
      trips = [];
      renderTrips();
      setConnection("offline", "API unavailable");
      setTripState(`Cannot reach API at ${apiBaseUrl}. ${readableError(error)} Check the address in Settings.`, true, true);
    }
  }

  function renderTrips() {
    els.tripList.replaceChildren();
    els.tripCount.textContent = trips.length ? String(trips.length) : "—";
    for (const trip of trips) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "trip-option";
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(trip.trip_id === selectedTripId));
      button.dataset.tripId = trip.trip_id;
      const id = document.createElement("span");
      id.className = "trip-id";
      id.textContent = trip.trip_id;
      const count = document.createElement("span");
      count.className = "trip-points";
      const points = Number.isFinite(Number(trip.n_points)) ? Number(trip.n_points) : null;
      count.innerHTML = `<b>${points === null ? "—" : points.toLocaleString()}</b> pts`;
      button.append(id, count);
      button.addEventListener("click", () => selectTrip(trip.trip_id));
      els.tripList.append(button);
    }
    if (selectedTripId && !trips.some((trip) => trip.trip_id === selectedTripId)) {
      selectedTripId = null;
      updateSelection();
    }
    els.runButton.disabled = !selectedTripId || Boolean(activeController);
  }

  function selectTrip(tripId) {
    if (activeController) {
      activeController.abort("trip-changed");
      activeController = null;
      activeRequestId += 1;
      setRunState("Prediction cancelled because the selected trip changed.");
    }
    selectedTripId = tripId;
    clearMapLayers();
    clearResult();
    setMapNotice("");
    updateSelection();
    els.mapEmpty.hidden = false;
  }

  function updateSelection() {
    const selected = trips.find((trip) => trip.trip_id === selectedTripId);
    for (const button of els.tripList.querySelectorAll(".trip-option")) {
      button.setAttribute("aria-selected", String(button.dataset.tripId === selectedTripId));
    }
    if (selected) {
      const count = Number.isFinite(Number(selected.n_points)) ? Number(selected.n_points).toLocaleString() : "unknown";
      els.selectedSummary.textContent = `Trip ${selected.trip_id} · ${count} GPS points`;
    } else {
      els.selectedSummary.textContent = "Select a trip to continue";
    }
    els.runButton.disabled = !selectedTripId || Boolean(activeController);
  }

  function initMap() {
    if (!window.L) {
      setMapNotice("The map library could not load. Check this laptop's internet connection and reload the page.");
      return;
    }
    map = L.map("map", { preferCanvas: true, zoomControl: false, scrollWheelZoom: true }).setView([0, 0], 2);
    L.control.zoom({ position: "topright" }).addTo(map);
    tileLayer = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap</a> contributors',
    }).addTo(map);
    gpsLayer = L.layerGroup();
    classifierLayer = L.layerGroup();
    hmmLayer = L.layerGroup();
    requestAnimationFrame(() => map.invalidateSize());
  }

  function clearMapLayers() {
    if (!map) return;
    for (const layer of [gpsLayer, classifierLayer, hmmLayer]) layer.clearLayers();
    if (layerControl) {
      map.removeControl(layerControl);
      layerControl = null;
    }
    if (map.hasLayer(gpsLayer)) map.removeLayer(gpsLayer);
    if (map.hasLayer(classifierLayer)) map.removeLayer(classifierLayer);
    if (map.hasLayer(hmmLayer)) map.removeLayer(hmmLayer);
    els.mapLegend.hidden = true;
  }

  function setMapNotice(message) {
    els.mapNotice.textContent = message;
    els.mapNotice.hidden = !message;
  }

  function clearResult() {
    els.resultsPlaceholder.hidden = false;
    els.resultsContent.hidden = true;
    els.resultTrip.textContent = "Waiting for a prediction";
    els.mockIndicator.hidden = true;
  }

  function startPrediction() {
    if (!selectedTripId || activeController) return;
    const tripId = selectedTripId;
    const requestId = ++activeRequestId;
    const controller = new AbortController();
    activeController = controller;
    els.runButton.disabled = true;
    els.mapEmpty.hidden = true;
    setMapNotice("");
    setRunState("Running prediction · 0s", false, true);
    const started = performance.now();
    const timer = window.setInterval(() => {
      if (requestId === activeRequestId) setRunState(`Running prediction · ${Math.floor((performance.now() - started) / 1000)}s`, false, true);
    }, 350);

    fetchJson(endpoint(`/predict/${encodeURIComponent(tripId)}`), {
      method: "POST",
      controller,
      headers: { Accept: "application/json" },
    }).then((data) => {
      if (requestId !== activeRequestId) return;
      const elapsed = (performance.now() - started) / 1000;
      renderPrediction(data);
      appendLog(tripId, "Success", elapsed, false);
      setRunState(`Complete · ${elapsed.toFixed(1)}s`);
      setConnection("online", "API online");
    }).catch((error) => {
      if (requestId !== activeRequestId || error.name === "AbortError" && controller.signal.reason === "trip-changed") return;
      const elapsed = (performance.now() - started) / 1000;
      const message = controller.signal.reason === "timeout"
        ? "The prediction timed out after 60 seconds. Try again or check the API laptop."
        : error.status === 404 || error.code === "UNKNOWN_TRIP"
          ? `Trip ${tripId} is not known to the API. Refresh the trip list and try again.`
          : readableError(error);
      setRunState(message, true);
      setMapNotice(message);
      appendLog(tripId, "Failed", elapsed, true);
      if (!error.status) setConnection("offline", "API unavailable");
    }).finally(() => {
      window.clearInterval(timer);
      if (requestId === activeRequestId) {
        activeController = null;
        els.runButton.disabled = !selectedTripId;
      }
    });
  }

  function renderPrediction(data) {
    if (!data || typeof data !== "object") throw new Error("The API returned an invalid prediction response.");
    const gps = Array.isArray(data.gps) ? data.gps : [];
    const metrics = data.metrics || {};
    renderMap(data);
    els.resultsPlaceholder.hidden = true;
    els.resultsContent.hidden = false;
    els.resultTrip.textContent = `TRIP ${data.trip_id || selectedTripId || "—"}`;
    $("metric-top1").textContent = formatPercent(metrics.top1_accuracy);
    $("metric-continuity").textContent = formatPercent(metrics.route_continuity);
    $("metric-f1").textContent = formatPercent(metrics.route_arc_f1);
    $("metric-points").textContent = formatInteger(metrics.n_points ?? data.n_points ?? gps.length);
    $("metric-scored").textContent = `${formatInteger(metrics.n_scored_points)} scored by API`;
    renderComparison(metrics.classifier_only || {}, metrics);
    $("detail-model").textContent = [data.model?.name, data.model?.version].filter(Boolean).join(" · ") || "—";
    $("detail-time").textContent = Number.isFinite(Number(data.timing?.seconds)) ? `${Number(data.timing.seconds).toFixed(2)} s (API)` : "—";
    $("detail-miss").textContent = formatPercent(metrics.candidate_miss_rate);
    renderWarnings(data.warnings);
    $("mock-indicator").hidden = data.mock !== true;
  }

  function renderMap(data) {
    if (!map) {
      els.mapEmpty.hidden = true;
      return;
    }
    clearMapLayers();
    const gps = Array.isArray(data.gps) ? data.gps : [];
    const pointArcs = data.per_point?.hmm_arc || [];
    const bounds = [];
    const step = gps.length > GPS_THRESHOLD ? Math.ceil(gps.length / GPS_THRESHOLD) : 1;
    for (let index = 0; index < gps.length; index += step) {
      const point = gps[index];
      if (!Array.isArray(point) || !Number.isFinite(Number(point[0])) || !Number.isFinite(Number(point[1]))) continue;
      const latLng = [Number(point[0]), Number(point[1])];
      bounds.push(latLng);
      const arcId = pointArcs[index];
      const marker = L.circleMarker(latLng, {
        renderer: L.canvas(), radius: gps.length > 1000 ? 2.5 : 3.5,
        color: "#fff", weight: 1, fillColor: "#5886a2", fillOpacity: 0.84,
      });
      marker.bindPopup(pointPopup(index, point, arcId));
      marker.bindTooltip(`Point ${index + 1}`, { direction: "top", opacity: 0.92 });
      marker.addTo(gpsLayer);
    }

    addRouteSegments(data.classifier_route?.segments, classifierLayer, {
      color: "#e88b3d", weight: 2.6, opacity: 0.86, lineCap: "round", lineJoin: "round",
    }, bounds);
    addRouteSegments(data.hmm_route?.segments, hmmLayer, {
      color: "#16a36a", weight: 5, opacity: 0.92, lineCap: "round", lineJoin: "round",
    }, bounds);

    const gpsToggle = `GPS observations${step > 1 ? ` (1 in ${step})` : ""}`;
    const overlays = { [gpsToggle]: gpsLayer, "Classifier route": classifierLayer, "Classifier + HMM": hmmLayer };
    gpsLayer.addTo(map);
    if (classifierLayer.getLayers().length) classifierLayer.addTo(map);
    if (hmmLayer.getLayers().length) hmmLayer.addTo(map);
    layerControl = L.control.layers({ "OpenStreetMap": tileLayer }, overlays, { collapsed: false, position: "topright" }).addTo(map);
    els.mapLegend.hidden = false;
    els.mapEmpty.hidden = true;
    if (step > 1) setMapNotice(`Showing every ${step}th GPS observation to keep the map responsive (${gps.length.toLocaleString()} total).`);
    else if (!gps.length) setMapNotice("This response contains no GPS observations.");
    else if (!classifierLayer.getLayers().length && !hmmLayer.getLayers().length) setMapNotice("This response has an empty route geometry; there are no route lines to draw.");
    else setMapNotice("");
    if (bounds.length) map.fitBounds(L.latLngBounds(bounds), { padding: [38, 38], maxZoom: 16 });
    requestAnimationFrame(() => map.invalidateSize());
  }

  function addRouteSegments(segments, targetLayer, style, bounds) {
    if (!Array.isArray(segments)) return;
    for (const segment of segments) {
      if (!Array.isArray(segment)) continue;
      const points = segment.filter((point) => Array.isArray(point) && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])))
        .map((point) => [Number(point[0]), Number(point[1])]);
      if (points.length < 2) {
        if (points.length === 1) bounds.push(points[0]);
        continue;
      }
      bounds.push(...points);
      L.polyline(points, style).addTo(targetLayer);
    }
  }

  function pointPopup(index, point, arcId) {
    const time = Number.isFinite(Number(point[2])) ? `${Number(point[2]).toLocaleString()} s` : "Not provided";
    const predicted = arcId === undefined || arcId === null ? "Not provided" : Number(arcId) === -1 ? "No candidate arc" : String(arcId);
    return `<div class="point-popup"><strong>Observation ${index + 1}</strong><span>Time: ${escapeHtml(time)}</span><span>Predicted HMM arc: ${escapeHtml(predicted)}</span></div>`;
  }

  function renderComparison(classifier, hmm) {
    const rows = [
      ["Top-1 accuracy", classifier.top1_accuracy, hmm.top1_accuracy],
      ["Route continuity", classifier.route_continuity, hmm.route_continuity],
      ["Route arc F1", classifier.route_arc_f1, hmm.route_arc_f1],
    ];
    const container = $("comparison-list");
    container.replaceChildren();
    for (const [name, before, after] of rows) {
      const row = document.createElement("div");
      row.className = "comparison-row";
      const label = document.createElement("span");
      label.className = "comparison-name";
      label.textContent = name;
      const oldValue = document.createElement("span");
      oldValue.className = "comparison-before";
      oldValue.textContent = formatPercent(before);
      const arrow = document.createElement("span");
      arrow.className = "comparison-arrow";
      arrow.textContent = "→";
      const newValue = document.createElement("span");
      newValue.className = "comparison-after";
      newValue.textContent = formatPercent(after);
      const delta = document.createElement("span");
      delta.className = "comparison-delta";
      const difference = Number(after) - Number(before);
      if (Number.isFinite(difference)) {
        delta.classList.add(difference >= 0 ? "positive" : "negative");
        delta.textContent = `${difference > 0 ? "+" : ""}${(difference * 100).toFixed(1)} pp`;
      } else delta.textContent = "—";
      row.append(label, oldValue, arrow, newValue, delta);
      container.append(row);
    }
  }

  function renderWarnings(warnings) {
    const section = $("warnings-section");
    const list = $("warnings-list");
    list.replaceChildren();
    const messages = Array.isArray(warnings) ? warnings : [];
    for (const warning of messages) {
      const item = document.createElement("li");
      item.textContent = String(warning);
      list.append(item);
    }
    section.hidden = messages.length === 0;
  }

  function appendLog(tripId, status, seconds, isError) {
    const now = new Date();
    const entry = { time: now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }), tripId, status, seconds, isError };
    requestLog.unshift(entry);
    requestLog.splice(MAX_LOG_ENTRIES);
    els.logCount.textContent = String(requestLog.length);
    els.requestLog.replaceChildren();
    if (!requestLog.length) {
      const empty = document.createElement("li");
      empty.className = "log-empty";
      empty.textContent = "No requests yet.";
      els.requestLog.append(empty);
      return;
    }
    for (const item of requestLog) {
      const row = document.createElement("li");
      const time = document.createElement("span");
      time.textContent = item.time;
      const trip = document.createElement("span");
      trip.textContent = item.tripId;
      const state = document.createElement("span");
      state.className = `log-status ${item.isError ? "error" : "ok"}`;
      state.textContent = item.status;
      const duration = document.createElement("span");
      duration.textContent = `${item.seconds.toFixed(1)} s`;
      row.append(time, trip, state, duration);
      els.requestLog.append(row);
    }
  }

  function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
  }

  function formatInteger(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.round(number).toLocaleString() : "—";
  }

  function readableError(error) {
    if (error?.name === "AbortError") return "The request was cancelled.";
    return error?.message || "The request failed. Check the API address and try again.";
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  }

  function openSettings() {
    els.apiUrl.value = apiBaseUrl;
    els.healthResult.textContent = "";
    els.healthResult.className = "health-result";
    els.settingsDialog.showModal();
    els.apiUrl.focus();
  }

  async function testConnection() {
    const candidate = normalizeBaseUrl(els.apiUrl.value);
    if (!candidate) {
      els.healthResult.textContent = "Enter an API base URL first.";
      els.healthResult.className = "health-result error";
      return;
    }
    els.healthResult.textContent = "Checking /health…";
    els.healthResult.className = "health-result";
    try {
      const health = await fetchJson(`${candidate}/health`, {}, 10000);
      const detail = [health.status || "available", health.model, health.version ? `v${health.version}` : null].filter(Boolean).join(" · ");
      els.healthResult.textContent = `Connected: ${detail}`;
      els.healthResult.className = "health-result success";
    } catch (error) {
      els.healthResult.textContent = `Cannot reach API at ${candidate}. ${readableError(error)}`;
      els.healthResult.className = "health-result error";
    }
  }

  function saveSettings(event) {
    event.preventDefault();
    const candidate = normalizeBaseUrl(els.apiUrl.value);
    if (!candidate) return;
    try { localStorage.setItem(STORAGE_KEY, candidate); } catch (_) { /* Keep the current-session value if storage is blocked. */ }
    apiBaseUrl = candidate;
    updateApiAddress();
    selectedTripId = null;
    activeController?.abort("trip-changed");
    activeController = null;
    updateSelection();
    clearMapLayers();
    clearResult();
    setMapNotice("");
    els.mapEmpty.hidden = false;
    setRunState("");
    els.settingsDialog.close();
    loadTrips();
  }

  function bindEvents() {
    $("settings-open").addEventListener("click", openSettings);
    $("settings-close").addEventListener("click", () => els.settingsDialog.close());
    $("test-connection").addEventListener("click", testConnection);
    els.settingsForm.addEventListener("submit", saveSettings);
    els.runButton.addEventListener("click", startPrediction);
    els.tripState.addEventListener("click", (event) => {
      if (event.target.closest("[data-retry]")) loadTrips();
    });
  }

  initMap();
  bindEvents();
  updateApiAddress();
  loadTrips();
})();
