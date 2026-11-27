    "use strict";
    const savedGraphPositions = parseJson(localStorage.getItem("cortex-graph-positions") || "{}", {});
    const app = {
      data: null, view: "overview", selectedId: null, selectedTrace: null, explorerPage: 1, pageSize: 36, firstLoad: true, kindGuide: null,
      deepLinkMemory: new URLSearchParams(window.location.search).get("memory"),
      deepLinkTrace: new URLSearchParams(window.location.search).get("trace"),
      healthReview: { kind: null, index: 0, pending: null, busy: false }, sleepRunId: null, sleepFilter: "all", metacognitionFilter: "all",
      review: { filter: "creation", index: 0, busy: false, preview: null, previewLoading: false, connectionPattern: "", lastConnection: null, creationReceipt: null, creationDrafts: {}, autoMap: localStorage.getItem("cortex-review-auto-map") !== "false", copilotByProposal: {} },
      recallSets: { data: null, preview: null, loaded: false, loading: false, busy: false },
      policy: { busy: false, pending: null },
      explorerRole: "readable", graphShowReference: false,
      failureCause: "all", failureTaskType: "all",
      graph: {
        scale: 1, panX: 0, panY: 0, showLabels: localStorage.getItem("cortex-graph-labels") !== "false",
        rendererMode: localStorage.getItem("cortex-graph-renderer") || "physics", renderToken: 0,
        clusterMode: localStorage.getItem("cortex-graph-cluster") || "neighborhood", searchMatchId: null,
        focusId: null, positions: new Map(Object.entries(savedGraphPositions)), customIds: new Set(Object.keys(savedGraphPositions)),
        visibleNodes: [], visibleEdges: [], world: null, nodeEls: new Map(), edgeEls: [], drag: null, pan: null,
        suppressClickUntil: 0, instances: {}, scripts: {}, highlightEdge: null, matrix: { scale:1, panX:0, panY:0 }
      }
    };
    const svgNS = "http://www.w3.org/2000/svg";
    const themeNames = {
      field: "Field", ink: "Ink", dusk: "Dusk", tide: "Tide", moss: "Moss", mono: "Mono",
      midnight: "Midnight", ember: "Ember", orchid: "Orchid", glacier: "Glacier", sepia: "Sepia", terminal: "Terminal"
    };
    const memoryKindGuide = {
      semantic: ["Semantic memory", "Facts and durable knowledge about the world. Cortex retrieves these when a task needs stable background, definitions, or known relationships."],
      episode: ["Episodic memory", "What happened in a particular conversation or task. Cortex uses episodes for continuity, recent context, and offline replay during Sleep."],
      procedure: ["Procedural memory", "Repeatable instructions and ways of working. Cortex favors these for how-to requests and can reinforce them with successful tool outcomes."],
      preference: ["Preference memory", "How the user likes things done, written, organized, or presented. Cortex uses it to personalize future work without asking again."],
      identity: ["Identity memory", "Stable facts about people, projects, systems, and roles. Cortex uses identity to keep references consistent across conversations."],
      decision: ["Decision memory", "A choice that was made and the context behind it. Cortex recalls decisions to avoid reopening settled questions or contradicting prior direction."],
      prospective: ["Prospective memory", "Something intended for the future: a plan, reminder, commitment, or next step. Cortex keeps it active until it is resolved or no longer relevant."],
      operational: ["Operational memory", "Current technical state such as paths, ports, services, environments, and deployment details. Cortex weighs currentness heavily because these facts can change."],
      schema: ["Schema memory", "A reviewed abstraction supported by at least three independently used source memories. The examples remain available, corrections mark the schema dirty, and the abstraction can be reversed."],
    };
    const authState = { enabled: true, username: "", currentPassword: "" };
    let sleepPollTimer = null;
    let benchmarkPollTimer = null;
    let evaluationPollTimer = null;

    const settingsThemeSlot = document.getElementById("settings-theme-slot");
    const themePicker = document.querySelector(".topbar .theme-picker");
    if (settingsThemeSlot && themePicker) {
      settingsThemeSlot.append(themePicker);
      document.getElementById("theme-toggle").hidden = true;
      document.getElementById("theme-menu").hidden = false;
    }
    const settingsAccountSlot = document.getElementById("settings-account-slot");
    const accountButton = document.getElementById("account-open");
    if (settingsAccountSlot && accountButton) {
      accountButton.textContent = "Change password or sign out";
      settingsAccountSlot.append(accountButton);
    }

    async function authRequest(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        cache: "no-store",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-Cortex-Request": "1" },
        body: JSON.stringify(payload || {})
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      return data;
    }

    async function authGet(path) {
      const response=await fetch(path,{cache:"no-store",credentials:"same-origin",headers:{"X-Cortex-Request":"1"}});
      const data=await response.json().catch(()=>({}));
      if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
      return data;
    }

    async function bootAuth() {
      try {
        const response = await fetch("/api/auth/status", { cache: "no-store", credentials: "same-origin" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const status = await response.json();
        authState.enabled = Boolean(status.auth_enabled);
        authState.username = status.username || "";
        document.getElementById("auth-username").value = authState.username;
        document.getElementById("account-user").textContent = authState.username ? `Signed in as ${authState.username}` : "";
        document.getElementById("settings-account-status").textContent = authState.username ? `Signed in as ${authState.username}` : "Dashboard account";
        document.getElementById("account-open").hidden = !authState.enabled;
        if (!authState.enabled || (status.authenticated && !status.must_change_password)) {
          enterDashboard();
          return;
        }
        if (status.authenticated && status.must_change_password) showForcedPasswordChange();
        else showLogin();
      } catch (error) {
        showLogin(`Could not check authentication: ${error.message}`);
      }
    }

    function enterDashboard() {
      authState.currentPassword = "";
      document.getElementById("auth-gate").hidden = true;
      document.getElementById("account-gate").hidden = true;
      document.body.style.overflow = "";
      load(false);
    }

    function showLogin(message = "") {
      document.getElementById("auth-gate").hidden = false;
      document.getElementById("auth-login-pane").hidden = false;
      document.getElementById("auth-change-pane").hidden = true;
      document.getElementById("auth-login-error").textContent = message;
      document.getElementById("loading-screen").classList.add("hidden");
      document.body.style.overflow = "hidden";
      requestAnimationFrame(() => document.getElementById("auth-username").focus());
    }

    function showForcedPasswordChange(currentPassword = "") {
      const form = document.getElementById("auth-forced-change-form");
      document.getElementById("auth-gate").hidden = false;
      document.getElementById("auth-login-pane").hidden = true;
      document.getElementById("auth-change-pane").hidden = false;
      form.current_password.value = currentPassword || authState.currentPassword;
      form.querySelector(".auth-error").textContent = "";
      document.body.style.overflow = "hidden";
      requestAnimationFrame(() => form.new_password.focus());
    }

    async function submitPasswordChange(form, forced) {
      const error = form.querySelector(".auth-error"), button = form.querySelector("button[type=submit]");
      error.textContent = "";
      if (form.new_password.value !== form.confirm_password.value) {
        error.textContent = "The new passwords do not match.";
        return;
      }
      button.disabled = true;
      try {
        const result = await authRequest("/api/auth/change-password", {
          current_password: form.current_password.value,
          new_password: form.new_password.value
        });
        authState.username = result.username || authState.username;
        document.getElementById("settings-account-status").textContent = authState.username ? `Signed in as ${authState.username}` : "Dashboard account";
        form.reset();
        if (forced) enterDashboard();
        else {
          document.getElementById("account-gate").hidden = true;
          document.body.style.overflow = "";
          showToast("Password updated. Other sessions were signed out.");
        }
      } catch (requestError) {
        error.textContent = requestError.message;
      } finally {
        button.disabled = false;
      }
    }

    document.getElementById("auth-login-form").addEventListener("submit", async event => {
      event.preventDefault();
      const form = event.currentTarget, error = document.getElementById("auth-login-error"), button = form.querySelector("button[type=submit]");
      error.textContent = ""; button.disabled = true;
      try {
        const password = form.password.value;
        const result = await authRequest("/api/auth/login", { username: form.username.value, password });
        authState.username = result.username || form.username.value;
        authState.currentPassword = password;
        document.getElementById("account-user").textContent = `Signed in as ${authState.username}`;
        document.getElementById("settings-account-status").textContent = `Signed in as ${authState.username}`;
        if (result.must_change_password) showForcedPasswordChange(password);
        else { form.reset(); enterDashboard(); }
      } catch (requestError) {
        error.textContent = requestError.message;
      } finally {
        button.disabled = false;
      }
    });
    document.getElementById("auth-forced-change-form").addEventListener("submit", event => { event.preventDefault(); submitPasswordChange(event.currentTarget, true); });
    document.getElementById("account-change-form").addEventListener("submit", event => { event.preventDefault(); submitPasswordChange(event.currentTarget, false); });
    document.getElementById("account-open").addEventListener("click", () => { document.getElementById("account-gate").hidden = false; document.body.style.overflow = "hidden"; requestAnimationFrame(() => document.querySelector("#account-change-form input").focus()); });
    document.getElementById("account-close").addEventListener("click", () => { document.getElementById("account-gate").hidden = true; document.getElementById("account-change-form").reset(); document.body.style.overflow = ""; });
    document.getElementById("account-logout").addEventListener("click", async () => {
      try { await authRequest("/api/auth/logout", {}); } catch (_) { /* The local session is cleared either way. */ }
      document.getElementById("account-gate").hidden = true;
      showLogin("Signed out.");
    });

    document.querySelectorAll(".nav-btn").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
    document.querySelectorAll("[data-go]").forEach(button => button.addEventListener("click", () => setView(button.dataset.go)));
    document.getElementById("refresh").addEventListener("click", () => load(false, { preserveReviewDraft: reviewDecisionInProgress() }));
    document.getElementById("theme-toggle").addEventListener("click", event => { event.stopPropagation(); toggleThemeMenu(); });
    document.getElementById("theme-menu").addEventListener("click", event => event.stopPropagation());
    document.querySelectorAll("[data-theme-choice]").forEach(button => button.addEventListener("click", () => setTheme(button.dataset.themeChoice)));
    document.addEventListener("click", closeThemeMenu);
    document.getElementById("drawer-close").addEventListener("click", closeDrawer);
    document.getElementById("drawer-backdrop").addEventListener("click", closeDrawer);
    document.getElementById("global-search").addEventListener("keydown", event => { if (event.key === "Enter") openGlobalSearch(); });
    document.getElementById("global-search").addEventListener("input", event => { if (!event.target.value && app.view === "explorer") document.getElementById("memory-search").value = ""; });
    document.getElementById("graph-renderer").addEventListener("change", switchRenderer);
    document.querySelectorAll("[data-renderer-mode]").forEach(button=>button.addEventListener("click",()=>setRendererMode(button.dataset.rendererMode)));
    document.getElementById("graph-cluster").addEventListener("change",event=>{app.graph.clusterMode=event.target.value;localStorage.setItem("cortex-graph-cluster",app.graph.clusterMode);renderActiveGraph();});
    ["graph-kind","graph-state","graph-relation"].forEach(id => document.getElementById(id).addEventListener("input", () => { renderActiveGraph(); requestAnimationFrame(fitActiveRenderer); }));
    let graphSearchTimer;document.getElementById("graph-search").addEventListener("input",()=>{clearTimeout(graphSearchTimer);graphSearchTimer=setTimeout(handleGraphSearch,180);});
    ["memory-search","memory-kind","memory-state","memory-source","memory-sort"].forEach(id => document.getElementById(id).addEventListener("input", () => { app.explorerPage = 1; renderExplorer(); }));
    document.querySelectorAll("[data-role-view]").forEach(button => button.addEventListener("click", () => { app.explorerRole = button.dataset.roleView; app.explorerPage = 1; renderExplorer(); }));
    document.getElementById("health-review-close").addEventListener("click", closeHealthReview);
    document.getElementById("training-next-action").addEventListener("click", handleTrainingNextAction);
    document.getElementById("policy-compile").addEventListener("click", () => runPolicyAction("compile"));
    document.getElementById("decision-log-filter").addEventListener("change", renderTrainingGovernance);
    document.getElementById("review-filters").addEventListener("click", event => {
      const button=event.target.closest("[data-review-filter]");if(!button)return;
      app.review.filter=button.dataset.reviewFilter||"all";app.review.index=0;app.review.connectionPattern="";
      document.querySelectorAll("[data-review-filter]").forEach(item=>item.classList.toggle("active",item===button));
      renderReviewInbox();
    });
    document.getElementById("connection-auto-map").checked=app.review.autoMap;
    document.getElementById("connection-auto-map").addEventListener("change",event=>{app.review.autoMap=event.target.checked;localStorage.setItem("cortex-review-auto-map",String(app.review.autoMap));});
    document.getElementById("connection-map-receipt-open").addEventListener("click",()=>{if(app.review.lastConnection)showConnectionOnMap(app.review.lastConnection);});
    document.getElementById("graph-return-review").addEventListener("click",()=>goToTrainingReviews("connection"));
    document.getElementById("timeline-range").addEventListener("change", renderTimeline);
    document.getElementById("trends-range").addEventListener("change", renderTrendChart);
    document.getElementById("capacity-impact-range").addEventListener("change", renderCapacityImpactChart);
    document.getElementById("benchmark-start").addEventListener("click", startBenchmark);
    document.getElementById("evaluation-start").addEventListener("click", startPrivateEvaluation);
    document.getElementById("metacognition-filter").addEventListener("change", event => { app.metacognitionFilter=event.target.value||"all"; renderMetacognitivePredictions(); });
    document.getElementById("sleep-start").addEventListener("click", startSleepSession);
    document.getElementById("experiment-toggle").addEventListener("click", toggleRecallExperiment);
    document.getElementById("sleep-trial-start").addEventListener("click", startSleepTrial);
    document.getElementById("summary-generate").addEventListener("click", generateSummaryCandidates);
    document.getElementById("prospective-form").addEventListener("submit", createProspectiveMemory);
    document.getElementById("failure-type").addEventListener("change", event=>{app.failureCause=event.target.value||"all";renderFailureExplorer();});
    document.getElementById("failure-task-type").addEventListener("change", event=>{app.failureTaskType=event.target.value||"all";renderFailureExplorer();});
    document.getElementById("sleep-ledger-filters").addEventListener("click", event => {
      const button=event.target.closest("[data-sleep-filter]");if(!button)return;
      app.sleepFilter=button.dataset.sleepFilter||"all";
      document.querySelectorAll("[data-sleep-filter]").forEach(item=>item.classList.toggle("active",item===button));
      if(app.data)renderSleepLedger(app.data.sleep_proposals||[],app.data.sleep_edge_changes||[],app.data.sleep_state_changes||[]);
    });
    document.getElementById("graph-plus").addEventListener("click", () => zoomActiveRenderer(1.22));
    document.getElementById("graph-minus").addEventListener("click", () => zoomActiveRenderer(1 / 1.22));
    document.getElementById("graph-reset").addEventListener("click", fitActiveRenderer);
    document.getElementById("graph-fit").addEventListener("click", fitActiveRenderer);
    document.getElementById("graph-reference").addEventListener("click", () => { app.graphShowReference = !app.graphShowReference; document.getElementById("graph-reference").setAttribute("aria-pressed", String(app.graphShowReference)); renderActiveGraph(); });
    document.getElementById("graph-arrange").addEventListener("click", resetActiveLayout);
    document.getElementById("graph-labels").addEventListener("click", toggleGraphLabels);
    document.getElementById("graph-focus").addEventListener("click", toggleGraphFocus);
    document.getElementById("graph-fullscreen").addEventListener("click", toggleGraphFullscreen);
    document.getElementById("graph-details").addEventListener("click", () => { if (app.selectedId) selectMemory(app.selectedId); });
    document.getElementById("graph-clear-focus").addEventListener("click", clearGraphFocus);
    document.addEventListener("fullscreenchange", () => { document.getElementById("graph-fullscreen").textContent = document.fullscreenElement ? "Exit full screen" : "Full screen"; setTimeout(() => { resizeActiveRenderer(); fitActiveRenderer(); }, 100); });
    document.getElementById("page-prev").addEventListener("click", () => { app.explorerPage = Math.max(1, app.explorerPage - 1); renderExplorer(); scrollExplorer(); });
    document.getElementById("page-next").addEventListener("click", () => { app.explorerPage += 1; renderExplorer(); scrollExplorer(); });
    let graphResizeTimer;
    window.addEventListener("resize",()=>{clearTimeout(graphResizeTimer);graphResizeTimer=setTimeout(()=>{if(app.view==="graph")resizeActiveRenderer();},120);});
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") { closeDrawer(); closeThemeMenu(); }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); document.getElementById("global-search").focus(); }
      if (event.key === "/" && !["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName)) { event.preventDefault(); document.getElementById("global-search").focus(); }
      if (app.view === "graph" && !["INPUT","SELECT","TEXTAREA"].includes(document.activeElement.tagName)) {
        if (event.key.toLowerCase() === "f") { event.preventDefault(); fitActiveRenderer(); }
        if (event.key.toLowerCase() === "l") { event.preventDefault(); toggleGraphLabels(); }
      }
    });

    const savedTheme = localStorage.getItem("cortex-theme");
    setTheme(themeNames[savedTheme] ? savedTheme : "field", false);
    if (![...document.getElementById("graph-renderer").options].some(option => option.value === app.graph.rendererMode)) app.graph.rendererMode = "physics";
    document.getElementById("graph-renderer").value = app.graph.rendererMode;
    if(![...document.getElementById("graph-cluster").options].some(option=>option.value===app.graph.clusterMode))app.graph.clusterMode="kind";
    document.getElementById("graph-cluster").value=app.graph.clusterMode;
    updateRendererModeControls();
    document.getElementById("graph-labels").setAttribute("aria-pressed", String(app.graph.showLabels));

    function reviewDecisionInProgress() {
      return app.view === "review" && Boolean(document.getElementById("review-decision-panel")?.dataset.action);
    }

    async function load(silent = false, { preserveReviewDraft = silent && reviewDecisionInProgress() } = {}) {
      const refresh = document.getElementById("refresh");
      refresh.classList.add("refreshing");
      try {
        const response = await fetch("/api/snapshot", { cache: "no-store" });
        if (response.status === 401) { showLogin("Your session expired. Sign in again."); return; }
        if (response.status === 403) { showForcedPasswordChange(); return; }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        app.data = await response.json();
        populateFilters();
        renderAll({ preserveReviewDraft });
        const generated = new Date(app.data.generated_at);
        document.getElementById("freshness").textContent = `Updated ${timeAgo(generated)}`;
        document.getElementById("sidebar-status").textContent = `${formatNum(app.data.stats.memories)} memories · ${formatNum(app.data.stats.edges)} connections`;
        if (app.deepLinkTrace) {
          const taskId = app.deepLinkTrace;
          app.deepLinkTrace = null;
          await selectTrace(taskId, { updateUrl: false });
        } else if (app.deepLinkMemory) {
          const memoryId = app.deepLinkMemory;
          app.deepLinkMemory = null;
          await selectMemory(memoryId, { updateUrl: false });
        }
        if (!silent && !app.firstLoad) showToast("Cortex data refreshed");
      } catch (error) {
        showToast(`Could not load Cortex: ${error.message}`, true);
      } finally {
        refresh.classList.remove("refreshing");
        if (app.firstLoad) { app.firstLoad = false; setTimeout(() => document.getElementById("loading-screen").classList.add("hidden"), 220); }
      }
    }

    function renderAll({ preserveReviewDraft = false } = {}) {
      renderOverview(); renderTimeline(); if (!preserveReviewDraft) renderReviewInbox(); renderTrainingGovernance(); renderInsights(); renderLearning(); renderCognition(); renderMetacognition(); renderTools(); renderHealth(); renderExplorer(); renderAutoJudge();
      if (app.view === "graph") renderActiveGraph();
    }

    function setView(name) {
      app.view = name;
      document.querySelectorAll(".nav-btn").forEach(button => button.setAttribute("aria-selected", String(button.dataset.view === name)));
      document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
      window.scrollTo({ top: 0, behavior: "smooth" });
      if (name === "graph") requestAnimationFrame(renderActiveGraph);
      if (name === "timeline") requestAnimationFrame(renderTimeline);
      if (name === "review") requestAnimationFrame(renderReviewInbox);
      if (name === "insights") requestAnimationFrame(renderInsights);
      if (name === "trust") requestAnimationFrame(renderMetacognition);
    }

    function toggleThemeMenu() {
      const menu = document.getElementById("theme-menu"), open = menu.hidden;
      menu.hidden = !open; document.getElementById("theme-toggle").setAttribute("aria-expanded", String(open));
    }

    function closeThemeMenu() {
      const menu = document.getElementById("theme-menu");
      if (menu.closest(".settings-theme-slot")) return;
      menu.hidden = true;
      document.getElementById("theme-toggle").setAttribute("aria-expanded", "false");
    }

    function setTheme(theme, persist = true) {
      if (!themeNames[theme]) theme = "field";
      document.documentElement.dataset.theme = theme;
      document.getElementById("theme-current").textContent = themeNames[theme];
      document.querySelectorAll("[data-theme-choice]").forEach(button => {
        const active = button.dataset.themeChoice === theme;
        button.setAttribute("aria-checked", String(active));
        button.querySelector(".theme-check").textContent = active ? "✓" : "";
      });
      if (persist) localStorage.setItem("cortex-theme", theme);
      requestAnimationFrame(() => {
        const bg = cssVar("--bg");
        document.querySelector('meta[name="theme-color"]').setAttribute("content", bg);
        if (app.data) { renderActivityChart(); renderKindDonut(); renderHealth(); renderTimeline(); renderInsights(); renderMetacognition(); if (app.view === "graph") renderActiveGraph(); }
      });
    }

    function openGlobalSearch() {
      const value = document.getElementById("global-search").value.trim();
      document.getElementById("memory-search").value = value; app.explorerPage = 1; setView("explorer"); renderExplorer();
    }

    function populateFilters() {
      const memories = app.data.memories || [];
      setOptions("graph-kind", unique(memories.map(row => row.kind)));
      setOptions("graph-state", unique(memories.map(row => row.state)));
      setOptions("memory-kind", unique(memories.map(row => row.kind)));
      setOptions("memory-state", unique(memories.map(row => row.state)));
      setOptions("memory-source", unique(memories.map(row => row.source_category)));
      setOptions("graph-relation", unique([...(app.data.edges || []).map(row => row.relation), ...(app.data.dependencies || []).map(() => "dependency")]));
    }

    function setOptions(id, values) {
      const select = document.getElementById(id), current = select.value;
      while (select.options.length > 1) select.remove(1);
      values.filter(Boolean).sort().forEach(value => select.add(new Option(humanize(value), value)));
      if ([...select.options].some(option => option.value === current)) select.value = current;
    }

    function renderOverview() {
      const d = app.data, memories = d.memories || [];
      const active = Number(d.stats?.states?.active ?? memories.filter(row => row.state === "active").length);
      const retrieved = sum(memories,"retrieved_count"), used = sum(memories,"used_count");
      const structural=technicalIssueCount(d.audit||{}),status=structural?"Repair":"Ready",statusValue=structural?30:100;
      document.getElementById("hero-health").textContent = status;
      const heroCount = document.getElementById("hero-memory-count");
      if (heroCount) {
        const stored = Number(d.stats?.memories);
        heroCount.textContent = Number.isFinite(stored) ? `${formatNum(stored)} memories` : "no data yet";
      }
      document.getElementById("health-orb").style.setProperty("--health-angle", `${statusValue}%`);
      const aj = d.auto_judge || {};
      const ajDecisions = aj.summary?.total || 0;
      metricCards("overview-metrics", [
        ["Active memories", active, `${formatNum(d.stats.memories)} stored`, "●"],
        ["Indexed notes", d.stats.documents, `${formatNum(d.stats.document_chunks)} sections`, "▤"],
        ["Connections", d.stats.edges, `${formatNum(d.contradiction_count)} contradictions`, "⌁"],
        ["Recall events", retrieved, `${formatNum(used)} confirmed used`, "↗"],
        ["Auto-Judge decisions", ajDecisions, `${formatNum(aj.summary?.remembered||0)} remembered · ${formatNum(aj.summary?.rejected||0)} rejected`, "⚖"],
        ["Tool observations", d.stats.tool_executions, `${formatNum(d.stats.tool_strategies)} strategies`, "◇"]
      ]);
      renderActivityChart();
      renderKindDonut();
      renderRoleStrip();
      renderRecentActivity();
      const highSignal = [...memories].filter(row => row.state === "active").sort((a,b) => signalScore(b)-signalScore(a)).slice(0,6);
      renderSignalList(highSignal);
      renderSources();
      document.getElementById("activity-total").textContent = `${formatNum((d.access_by_day || []).reduce((n,row)=>n+Number(row.count||0),0))} events`;
    }

    function renderActivityChart() {
      if (!app.data) return;
      const svg = document.getElementById("activity-chart"); svg.replaceChildren();
      const byDay = {};
      (app.data.access_by_day || []).forEach(row => { byDay[row.day] = (byDay[row.day] || 0) + Number(row.count || 0); });
      let entries = Object.entries(byDay).sort((a,b)=>a[0].localeCompare(b[0])).slice(-21);
      if (!entries.length) { svg.append(svgText(380,120,"Activity will appear after Cortex recalls memories","chart-label","middle")); return; }
      if(entries.length<4){const events={};(app.data.access_by_day||[]).forEach(row=>{events[row.event]=(events[row.event]||0)+Number(row.count||0);});const rows=Object.entries(events).sort((a,b)=>b[1]-a[1]),max=Math.max(1,...rows.map(row=>row[1])),W=760,L=108,R=46,T=28,rowH=44;rows.forEach(([name,count],index)=>{const y=T+index*rowH,width=(W-L-R)*count/max;svg.append(svgText(L-10,y+16,humanize(name),"chart-label","end"),svgEl("rect",{x:L,y,width:Math.max(2,width),height:18,fill:cssVar("--mint"),opacity:.82}),svgText(Math.min(W-R,L+width+8),y+15,formatNum(count),"chart-label"));});svg.append(svgText(L,230,"Current activity mix · a time trend will appear after several active days","chart-label"));return;}
      const W=760,H=245,L=38,R=12,T=18,B=34,max=Math.max(1,...entries.map(e=>e[1]));
      const defs=svgEl("defs"), gradient=svgEl("linearGradient",{id:"activity-gradient",x1:"0",y1:"0",x2:"0",y2:"1"});
      const activityColor=cssVar("--mint");
      gradient.append(svgEl("stop",{offset:"0%","stop-color":activityColor,"stop-opacity":".22"}),svgEl("stop",{offset:"100%","stop-color":activityColor,"stop-opacity":"0"})); defs.append(gradient); svg.append(defs);
      for(let i=0;i<4;i++){ const y=T+i*(H-T-B)/3; svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,class:"chart-grid"})); svg.append(svgText(L-8,y+4,String(Math.round(max*(3-i)/3)),"chart-label","end")); }
      const points=entries.map((entry,i)=>({x:L+i*(W-L-R)/Math.max(1,entries.length-1),y:T+(max-entry[1])*(H-T-B)/max,value:entry[1],day:entry[0]}));
      const line=points.map((p,i)=>`${i?"L":"M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      const area=`${line} L${points[points.length-1].x},${H-B} L${points[0].x},${H-B} Z`;
      svg.append(svgEl("path",{d:area,class:"activity-area"}),svgEl("path",{d:line,class:"activity-line"}));
      points.forEach((p,i)=>{ if(entries.length<=10 || i%Math.ceil(entries.length/6)===0 || i===points.length-1) svg.append(svgText(p.x,H-10,shortDay(p.day),"chart-label","middle")); const c=svgEl("circle",{cx:p.x,cy:p.y,r:4,class:"chart-point"}); const title=svgEl("title"); title.textContent=`${p.day}: ${p.value} memory events`; c.append(title); svg.append(c); });
    }

    function renderRoleStrip() {
      const host=document.getElementById("role-strip");if(!host)return;host.replaceChildren();
      const counts=app.data?.refinery?.role_counts||{};
      const order=["canonical","reference","event","claim"];
      order.filter(role=>counts[role]).forEach(role=>{const chip=el("span",`role-chip role-${role}`,roleLabel(role));chip.append(el("strong","",formatNum(counts[role])));host.append(chip);});
      if(!host.childNodes.length)host.append(el("span","small-meta","Record roles appear after the first classification pass."));
    }
    function renderKindDonut() {
      const aggregateKinds=app.data?.stats?.kinds, counts=aggregateKinds&&typeof aggregateKinds==="object"?Object.fromEntries(Object.entries(aggregateKinds).map(([kind,count])=>[kind,Number(count)||0])):countBy(app.data.memories||[],"kind"), entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]), total=Math.max(1,entries.reduce((n,e)=>n+e[1],0));
      let cursor=0, stops=[];
      entries.forEach(([kind,count])=>{ const start=cursor, end=cursor+count/total*100; stops.push(`${kindColor(kind)} ${start}% ${end}%`); cursor=end; });
      document.getElementById("kind-donut").style.background=`conic-gradient(${stops.join(",")})`;
      document.getElementById("donut-total").textContent=formatNum(entries.reduce((n,e)=>n+e[1],0));
      const legend=document.getElementById("kind-legend"); legend.replaceChildren();
      entries.slice(0,6).forEach(([kind,count])=>{ const row=el("div","legend-row"); const sw=el("i","legend-swatch"); sw.style.background=kindColor(kind); row.append(sw,el("span","",humanize(kind)),el("strong","",formatNum(count))); legend.append(row); });
      renderKindGuide(counts);
    }

    function renderKindGuide(counts) {
      const host=document.getElementById("kind-guide");host.replaceChildren();
      Object.entries(memoryKindGuide).forEach(([kind,[title]])=>{const button=el("button","kind-guide-button");button.type="button";button.setAttribute("aria-expanded",String(app.kindGuide===kind));const swatch=el("i","legend-swatch");swatch.style.background=kindColor(kind);button.append(swatch,el("strong","",humanize(kind)),el("small","",formatNum(counts[kind]||0)));button.addEventListener("click",()=>{app.kindGuide=app.kindGuide===kind?null:kind;renderKindGuide(counts);});host.append(button);});
      if(app.kindGuide&&memoryKindGuide[app.kindGuide]){const[title,copy]=memoryKindGuide[app.kindGuide],detail=el("div","kind-guide-detail");detail.append(el("strong","",title),el("p","",copy));host.append(detail);}
    }

    function renderRecentActivity() {
      const host=document.getElementById("recent-activity"); host.replaceChildren(); host.removeAttribute("aria-busy");
      const rows=(app.data.recent_access||[]).filter(row=>row.memory_id&&row.content).slice(0,7);
      if(!rows.length){ host.append(emptyState("No recall activity yet","The feed will wake up when Hermes uses Cortex.")); return; }
      rows.forEach(row=>{ const item=el("div","activity-item"); item.tabIndex=0; item.append(el("div","activity-glyph",eventGlyph(row.event))); const copy=el("div",""); copy.append(el("div","activity-content",truncate(cleanMemory(row.content),88)),el("div","activity-meta",`${humanize(row.event)} · ${humanize(row.kind)}`)); item.append(copy,el("div","activity-time",timeAgo(new Date(row.created_at)))); item.addEventListener("click",()=>selectMemory(row.memory_id)); item.addEventListener("keydown",e=>{if(e.key==="Enter")selectMemory(row.memory_id)}); host.append(item); });
    }

    function renderSignalList(rows) {
      const host=document.getElementById("signal-list"); host.replaceChildren(); host.removeAttribute("aria-busy");
      rows.forEach(row=>{ const item=el("div","signal-item"); item.tabIndex=0; const copy=el("div",""); copy.append(el("div","signal-title",truncate(cleanMemory(row.content),100)),el("div","small-meta",`${humanize(row.kind)} · ${sourceName(row)}`)); item.append(copy,el("div","signal-score",Math.round(signalScore(row)*100))); item.addEventListener("click",()=>selectMemory(row.id)); item.addEventListener("keydown",e=>{if(e.key==="Enter")selectMemory(row.id)}); host.append(item); });
    }

    function renderSources() {
      const sources=app.data.sources||[], counts={};
      sources.filter(row=>row.status==="active").forEach(row=>{ const part=(row.source_path||"root").split("/")[0]; const key=(row.source_path||"").includes("/")?part:"root"; counts[key]=(counts[key]||0)+1; });
      const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]), max=Math.max(1,...entries.map(e=>e[1]));
      document.getElementById("source-total").textContent=`${formatNum(sources.filter(r=>r.status==="active").length)} sources`;
      const host=document.getElementById("source-list"); host.replaceChildren();
      entries.forEach(([name,count])=>{ const row=el("div","source-item"); const track=el("div","bar-track"),fill=el("div","bar-fill"); fill.style.width=`${count/max*100}%`; track.append(fill); row.append(el("div","",humanize(name)),track,el("div","source-count",count)); host.append(row); });
    }

    function buildGraphDataset() {
      const graph = app.graph;
      const query = document.getElementById("graph-search").value.trim().toLowerCase();
      const kind = document.getElementById("graph-kind").value, state = document.getElementById("graph-state").value;
      const allEdges = graphAllEdges();
      const activeRecallSet = String(app.data?.recall_set?.active_set || "legacy").toLowerCase();
      let candidates = (app.data.memories || []).filter(row =>
        (activeRecallSet !== "trained" || row.recall_eligibility === "primary" ||
          (app.graphShowReference && row.recall_eligibility === "evidence_only")) &&
        (app.graphShowReference || memoryRole(row) === "canonical") &&
        (!kind || row.kind === kind) && (!state || row.state === state)
      );
      if (graph.focusId) {
        const neighborhood = new Set([graph.focusId]);
        allEdges.forEach(edge => {
          if (edge.src_id === graph.focusId) neighborhood.add(edge.dst_id);
          if (edge.dst_id === graph.focusId) neighborhood.add(edge.src_id);
        });
        candidates = candidates.filter(row => neighborhood.has(row.id));
      }
      candidates.sort((a,b) => (connectionCount(b.id) + signalScore(b) * 2) - (connectionCount(a.id) + signalScore(a) * 2));
      const nodes = candidates.slice(0, 650), nodeMap = new Map(nodes.map(row => [row.id, row]));
      graph.searchMatchId=query?(nodes.find(row=>searchText(row).includes(query))?.id||null):null;
      const edges = allEdges.filter(edge => nodeMap.has(edge.src_id) && nodeMap.has(edge.dst_id));
      graph.visibleNodes = nodes; graph.visibleEdges = edges; ensureGraphLayout(nodes);
      return { nodes, edges };
    }

    const rendererDefinitions = {
      manual: { label:"Manual SVG", layer:"graph-svg", help:"Stable SVG layout · drag nodes and space · saved manual positions" },
      atlas: { label:"Atlas WebGL", layer:"atlas-renderer", help:"Sigma WebGL atlas · fast camera navigation · drag nodes to reshape the atlas" },
      physics: { label:"Physics Canvas", layer:"physics-renderer", help:"Live force simulation · nodes react as you drag · wheel to zoom and drag empty space to pan" },
      analysis: { label:"Cytoscape Analysis", layer:"analysis-renderer", help:"Graph-analysis view · structured layout · select and drag nodes · inspect neighborhoods" },
      constellation: { label:"3D Constellation", layer:"constellation-renderer", help:"Rotate, orbit, zoom, and drag through the memory field in three dimensions" },
      matrix: { label:"Adjacency Matrix", layer:"matrix-renderer", help:"Every row and column is a memory · marks reveal clusters and cross-links without tangled edges" }
    };
    const rendererAssets = {
      atlas: [
        ["/vendor/graphology.umd.min.js", () => globalThis.graphology],
        ["/vendor/sigma.min.js", () => globalThis.Sigma]
      ],
      physics: [["/vendor/force-graph.min.js", () => globalThis.ForceGraph]],
      analysis: [["/vendor/cytoscape.min.js", () => globalThis.cytoscape]],
      constellation: [["/vendor/3d-force-graph.min.js", () => globalThis.ForceGraph3D]]
    };

    function switchRenderer() {
      const mode=document.getElementById("graph-renderer").value;
      setRendererMode(mode);
    }

    function setRendererMode(mode) {
      if(!rendererDefinitions[mode]||mode===app.graph.rendererMode){updateRendererModeControls();return;}
      teardownRenderer(app.graph.rendererMode);app.graph.rendererMode=mode;localStorage.setItem("cortex-graph-renderer",mode);document.getElementById("graph-renderer").value=mode;updateRendererModeControls();renderActiveGraph();
    }

    function updateRendererModeControls() {
      document.querySelectorAll("[data-renderer-mode]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.rendererMode===app.graph.rendererMode)));
      const cluster=document.getElementById("graph-cluster");if(cluster)cluster.disabled=!['physics','constellation'].includes(app.graph.rendererMode);
    }

    function handleGraphSearch() {
      renderActiveGraph();const query=document.getElementById("graph-search").value.trim();if(query&&!app.graph.searchMatchId)showToast("No memory matches that search",true);
    }

    function activateRendererLayer(mode) {
      document.querySelectorAll(".renderer-layer").forEach(layer=>layer.classList.toggle("active",layer.id===rendererDefinitions[mode].layer));
      document.getElementById("graph-help").textContent=rendererDefinitions[mode].help;
      document.getElementById("graph-arrange").textContent=mode==="manual"?"Reset layout":mode==="analysis"?"Run layout":mode==="matrix"?"Sort matrix":"Reheat";
      const labels=document.getElementById("graph-labels");labels.disabled=mode==="matrix";labels.setAttribute("aria-pressed",String(app.graph.showLabels));
      updateRendererModeControls();
    }

    async function renderActiveGraph() {
      if(!app.data)return;
      const mode=app.graph.rendererMode,definition=rendererDefinitions[mode]||rendererDefinitions.manual,dataset=buildGraphDataset(),token=++app.graph.renderToken;
      if(mode!=="manual"&&app.graph.instances[mode])teardownRenderer(mode);
      activateRendererLayer(mode);updateGraphSelection();updateGraphSummary();
      const loading=document.getElementById("renderer-loading");loading.classList.add("active");loading.textContent=`Loading ${definition.label}…`;
      if(mode==="manual"){loading.classList.remove("active");renderGraph(dataset);revealGraphSearchMatch(mode);return;}
      try{
        await ensureRendererAssets(mode);
        if(token!==app.graph.renderToken||mode!==app.graph.rendererMode)return;
        if(mode==="atlas")renderAtlas(dataset);
        else if(mode==="physics")renderPhysics(dataset);
        else if(mode==="analysis")renderAnalysis(dataset);
        else if(mode==="constellation")renderConstellation(dataset);
        else if(mode==="matrix")renderMatrix(dataset);
      }catch(error){
        if(token!==app.graph.renderToken)return;
        loading.replaceChildren();const box=el("div","renderer-error");box.append(el("strong","",`${definition.label} could not load`),el("div","small-meta",error.message));loading.append(box);return;
      }
      if(token===app.graph.renderToken){loading.classList.remove("active");revealGraphSearchMatch(mode);}
    }

    async function ensureRendererAssets(mode) {
      const assets=rendererAssets[mode]||[];
      for(const[url,check]of assets)await loadScriptOnce(url,check);
    }

    function loadScriptOnce(url,check) {
      if(check())return Promise.resolve();
      if(app.graph.scripts[url])return app.graph.scripts[url];
      app.graph.scripts[url]=new Promise((resolve,reject)=>{const script=document.createElement("script");script.src=url;script.async=true;script.onload=()=>check()?resolve():reject(new Error("Renderer loaded without its browser API"));script.onerror=()=>reject(new Error("The renderer library could not be downloaded"));document.head.append(script);}).catch(error=>{delete app.graph.scripts[url];throw error;});
      return app.graph.scripts[url];
    }

    function teardownRenderer(mode) {
      const instance=app.graph.instances[mode];if(!instance)return;
      try{
        if(mode==="atlas")instance.renderer.kill();
        else if(mode==="analysis")instance.destroy();
        else if(mode==="physics"||mode==="constellation"){instance.pauseAnimation?.();instance._destructor?.();}
      }catch{}
      const layer=document.getElementById(rendererDefinitions[mode]?.layer||"");if(layer&&mode!=="manual"&&mode!=="matrix")layer.replaceChildren();
      delete app.graph.instances[mode];
    }

    function renderGraph(dataset = buildGraphDataset()) {
      if (!app.data) return;
      const graph = app.graph, svg = document.getElementById("graph-svg"), {nodes,edges} = dataset;
      graph.nodeEls = new Map(); graph.edgeEls = [];
      svg.replaceChildren();
      const title = svgEl("title"); title.textContent = "Cortex memory map";
      const description = svgEl("desc"); description.textContent = `${nodes.length} memories and ${edges.length} visible connections. Drag nodes to arrange them, drag the background to pan, and use the mouse wheel to zoom.`;
      svg.append(title, description);
      if (!nodes.length) {
        svg.append(svgText(500,300,"No memories match these filters","axis-label","middle"));
        updateGraphSummary(); return;
      }
      const world = svgEl("g", { class: "graph-world" }); graph.world = world;
      edges.forEach(edge => {
        const a = graph.positions.get(edge.src_id), b = graph.positions.get(edge.dst_id);
        if (!a || !b) return;
        const highlighted=edgeMatchesHighlight(edge),muted=!highlighted&&relationFocus()&&edge.relation!==relationFocus();
        const line = svgEl("line", { x1:a.x, y1:a.y, x2:b.x, y2:b.y, class:`graph-edge ${edge.relation || "related"}${muted?" is-muted":""}${highlighted?" is-highlighted":""}` });
        const edgeTitle = svgEl("title"); edgeTitle.textContent = `${humanize(edge.relation)} · ${edge.explanation||"legacy link; evidence was not preserved"}\nEvidence: ${humanize(edge.evidence_type||"legacy_unattributed")} · weight ${Number(edge.weight || 0).toFixed(2)}`; line.append(edgeTitle);
        graph.edgeEls.push({ line, src:edge.src_id, dst:edge.dst_id }); world.append(line);
      });
      renderGraphClusterLabels(world, nodes);
      nodes.forEach(node => {
        const position = graph.positions.get(node.id), radius = 4.8 + Math.min(8, Number(node.importance || 0) * 5 + Math.log2(1 + Number(node.used_count || 0)) * 1.6);
        const group = svgEl("g", { class:"graph-node-group", transform:`translate(${position.x} ${position.y})`, "data-memory-id":node.id });
        if (node.id === app.selectedId) group.append(svgEl("circle", { cx:0, cy:0, r:radius + 7, class:"graph-halo" }));
        const circle = svgEl("circle", { cx:0, cy:0, r:radius, class:"graph-node", fill:kindColor(node.kind), opacity:node.state === "archived" ? .42 : 1 });
        const nodeTitle = svgEl("title"); nodeTitle.textContent = `${humanize(node.kind)} · ${humanize(node.state)}\n${cleanMemory(node.content)}`; circle.append(nodeTitle); group.append(circle);
        if (graph.showLabels && (nodes.length < 90 || Number(node.used_count) > 0 || Number(node.importance) > .72 || node.id === app.selectedId)) {
          group.append(svgText(radius + 5, 3, truncate(cleanMemory(node.content), 34), "graph-label"));
        }
        group.addEventListener("pointerdown", event => startGraphNodeDrag(event, node.id, group));
        group.addEventListener("click", event => { event.stopPropagation(); if (Date.now() > graph.suppressClickUntil) selectGraphNode(node.id); });
        group.addEventListener("dblclick", event => { event.preventDefault(); event.stopPropagation(); selectGraphNode(node.id); selectMemory(node.id); });
        graph.nodeEls.set(node.id, group); world.append(group);
      });
      svg.append(world); applyGraphTransform(); bindGraphPointerEvents(svg); updateGraphSelection(); updateGraphSummary();
      if (!graph.didInitialFit) { graph.didInitialFit = true; requestAnimationFrame(fitGraph); }
    }

    function renderAtlas({nodes,edges}) {
      const container=document.getElementById("atlas-renderer");container.replaceChildren();
      const GraphClass=globalThis.graphology?.Graph||globalThis.graphology?.MultiGraph||globalThis.graphology;
      const SigmaClass=globalThis.Sigma?.Sigma||globalThis.Sigma?.default||globalThis.Sigma;
      if(typeof GraphClass!=="function"||typeof SigmaClass!=="function")throw new Error("Sigma or Graphology did not expose a compatible browser build");
      const graph=new GraphClass({multi:true,type:"directed",allowSelfLoops:true});
      nodes.forEach(node=>{const p=app.graph.positions.get(node.id);graph.addNode(node.id,{label:truncate(cleanMemory(node.content),72),x:p.x,y:p.y,size:nodeRadius(node),color:kindColor(node.kind),kind:node.kind,state:node.state});});
      edges.forEach((edge,index)=>{if(graph.hasNode(edge.src_id)&&graph.hasNode(edge.dst_id)){const highlighted=edgeMatchesHighlight(edge);graph.addEdgeWithKey(`e${index}`,edge.src_id,edge.dst_id,{size:highlighted?5:relationFocus()&&edge.relation!==relationFocus()?.18:Math.max(.4,Number(edge.weight||.25)*1.8),color:highlighted?cssVar("--violet"):edgeDisplayColor(edge.relation),relation:edge.relation,highlighted});}});
      const renderer=new SigmaClass(graph,container,{renderEdgeLabels:false,labelDensity:app.graph.showLabels?1:.08,labelGridCellSize:90,labelRenderedSizeThreshold:app.graph.showLabels?6:14,zIndex:true,nodeReducer:(id,data)=>id===app.selectedId?{...data,size:data.size*1.65,color:cssVar("--violet"),zIndex:2}:data,edgeReducer:(id,data)=>data.highlighted?{...data,size:5,color:cssVar("--violet"),zIndex:3}:data});
      let draggedNode=null,moved=false;
      renderer.on("clickNode",event=>{if(!moved)selectGraphNode(event.node);moved=false;});
      try{renderer.on("doubleClickNode",event=>{selectGraphNode(event.node);selectMemory(event.node);});}catch{}
      renderer.on("downNode",event=>{draggedNode=event.node;moved=false;graph.setNodeAttribute(draggedNode,"highlighted",true);if(!renderer.getCustomBBox?.())renderer.setCustomBBox?.(renderer.getBBox());});
      const captor=renderer.getMouseCaptor?.();
      if(captor){
        captor.on("mousemovebody",event=>{if(!draggedNode)return;const position=renderer.viewportToGraph(event);graph.setNodeAttribute(draggedNode,"x",position.x);graph.setNodeAttribute(draggedNode,"y",position.y);moved=true;event.preventSigmaDefault?.();event.original?.preventDefault?.();event.original?.stopPropagation?.();});
        captor.on("mouseup",()=>{if(!draggedNode)return;const attrs=graph.getNodeAttributes(draggedNode);app.graph.positions.set(draggedNode,{x:attrs.x,y:attrs.y});app.graph.customIds.add(draggedNode);persistGraphPositions();graph.removeNodeAttribute(draggedNode,"highlighted");draggedNode=null;});
      }
      app.graph.instances.atlas={renderer,graph};requestAnimationFrame(()=>renderer.getCamera().animatedReset?.({duration:350}));
    }

    function renderPhysics({nodes,edges}) {
      const container=document.getElementById("physics-renderer");container.replaceChildren();const size=rendererSize(container);
      const targets=graphClusterTargets(nodes,2);
      const data={nodes:nodes.map(node=>{const p=app.graph.positions.get(node.id),target=targets.get(node.id)||{x:0,y:0};return{id:node.id,label:truncate(cleanMemory(node.content),80),kind:node.kind,state:node.state,importance:Number(node.importance||0),used:Number(node.used_count||0),x:p.x,y:p.y,clusterX:target.x,clusterY:target.y};}),links:edges.map(edge=>({source:edge.src_id,target:edge.dst_id,relation:edge.relation,weight:Number(edge.weight||.25),highlighted:edgeMatchesHighlight(edge)}))};
      const instance=globalThis.ForceGraph()(container).width(size.width).height(size.height).backgroundColor(cssVar("--surface")).graphData(data).nodeId("id").nodeVal(node=>1+node.importance*2+Math.log2(1+node.used)).nodeColor(node=>node.id===app.selectedId?cssVar("--violet"):kindColor(node.kind)).nodeLabel(node=>escapeHtml(node.label)).linkColor(link=>link.highlighted?cssVar("--violet"):edgeDisplayColor(link.relation)).linkWidth(link=>link.highlighted?5:relationFocus()&&link.relation!==relationFocus()?.16:Math.max(.35,link.weight*1.6)).linkDirectionalParticles(link=>link.highlighted?3:!relationFocus()||link.relation===relationFocus()?((link.relation==="supersedes"||link.relation==="contradicts")?1:0):0).linkDirectionalParticleWidth(link=>link.highlighted?3:1.5).cooldownTime(9000).warmupTicks(18).d3VelocityDecay(.24).enableNodeDrag(true).enablePanInteraction(true).enableZoomInteraction(true).onNodeHover(node=>{container.style.cursor=node?"pointer":"grab";}).onNodeClick((node,event)=>{selectGraphNode(node.id);if(event?.detail>=2)selectMemory(node.id);}).onNodeDragEnd(node=>{node.fx=node.x;node.fy=node.y;app.graph.positions.set(node.id,{x:node.x,y:node.y});app.graph.customIds.add(node.id);persistGraphPositions();});
      instance.nodeCanvasObject((node,context,scale)=>paintForceNode(node,context,scale)).nodeCanvasObjectMode(()=>"replace");
      try{instance.d3Force("charge").strength(-48);instance.d3Force("link").distance(link=>46+(1-Number(link.weight||.25))*42);instance.d3Force("cluster",app.graph.clusterMode==="none"?null:makeClusterForce(2,.2));}catch{}
      let fitted=Boolean(app.graph.searchMatchId);instance.onEngineStop(()=>{if(!fitted){fitted=true;instance.zoomToFit(500,55);}});app.graph.instances.physics=instance;
    }

    function paintForceNode(node,context,scale) {
      const radius=3.2+node.importance*3+Math.min(3,Math.log2(1+node.used));context.beginPath();context.arc(node.x,node.y,radius,0,Math.PI*2);context.fillStyle=node.id===app.selectedId?cssVar("--violet"):kindColor(node.kind);context.globalAlpha=node.state==="archived"?.42:1;context.fill();context.globalAlpha=1;
      if(app.graph.showLabels&&(app.graph.visibleNodes.length<100||node.used>0||node.importance>.72||node.id===app.selectedId)){const fontSize=10/Math.max(.2,scale);context.font=`${fontSize}px Avenir Next, sans-serif`;context.fillStyle=cssVar("--text");context.textBaseline="middle";context.fillText(truncate(node.label,34),node.x+radius+3/Math.max(.2,scale),node.y);}
    }

    function renderAnalysis({nodes,edges}) {
      const container=document.getElementById("analysis-renderer");container.replaceChildren();
      const elements=[...nodes.map(node=>{const p=app.graph.positions.get(node.id);return{data:{id:node.id,label:truncate(cleanMemory(node.content),62),kind:node.kind,state:node.state,size:nodeRadius(node)},position:{x:p.x,y:p.y},selected:node.id===app.selectedId};}),...edges.map((edge,index)=>({data:{id:`e${index}`,source:edge.src_id,target:edge.dst_id,relation:edge.relation,weight:Number(edge.weight||.25),highlighted:edgeMatchesHighlight(edge)?"yes":"no"}}))];
      const cy=globalThis.cytoscape({container,elements,layout:{name:"preset",fit:true,padding:45},minZoom:.04,maxZoom:8,wheelSensitivity:.18,boxSelectionEnabled:true,style:[
        {selector:"node",style:{"background-color":element=>kindColor(element.data("kind")),"width":element=>element.data("size")*1.7,"height":element=>element.data("size")*1.7,"label":app.graph.showLabels?"data(label)":"","font-size":9,"font-family":"Avenir Next, sans-serif","color":cssVar("--text"),"text-background-color":cssVar("--surface"),"text-background-opacity":.82,"text-background-padding":"2px","text-valign":"center","text-halign":"right","text-margin-x":5,"border-width":1,"border-color":cssVar("--surface"),"opacity":element=>element.data("state")==="archived"?.42:1}},
        {selector:"node:selected",style:{"background-color":cssVar("--violet"),"border-width":4,"border-color":cssVar("--violet"),"overlay-opacity":.08,"overlay-color":cssVar("--violet")}},
        {selector:"edge",style:{"line-color":element=>edgeDisplayColor(element.data("relation")),"width":element=>relationFocus()&&element.data("relation")!==relationFocus()?.35:Math.max(.5,element.data("weight")*2),"curve-style":"bezier","opacity":element=>relationFocus()&&element.data("relation")!==relationFocus()?.1:.58,"target-arrow-shape":element=>["supersedes","dependency"].includes(element.data("relation"))?"triangle":"none","target-arrow-color":element=>edgeDisplayColor(element.data("relation")),"arrow-scale":.55}},
        {selector:"edge[highlighted = 'yes']",style:{"line-color":cssVar("--violet"),"target-arrow-color":cssVar("--violet"),"width":5,"opacity":1,"z-index":8}},
        {selector:"edge[relation = 'contradicts']",style:{"line-style":"dashed"}}
      ]});
      let lastTap={id:null,time:0};cy.on("tap","node",event=>{const id=event.target.id(),now=Date.now();selectGraphNode(id);if(lastTap.id===id&&now-lastTap.time<350)selectMemory(id);lastTap={id,time:now};});
      cy.on("dragfree","node",event=>{const id=event.target.id(),p=event.target.position();app.graph.positions.set(id,{x:p.x,y:p.y});app.graph.customIds.add(id);persistGraphPositions();});
      app.graph.instances.analysis=cy;requestAnimationFrame(()=>cy.fit(undefined,45));
    }

    function renderConstellation({nodes,edges}) {
      const container=document.getElementById("constellation-renderer");container.replaceChildren();const size=rendererSize(container);
      const targets=graphClusterTargets(nodes,3);
      const data={nodes:nodes.map(node=>{const p=app.graph.positions.get(node.id),target=targets.get(node.id)||{x:0,y:0,z:0};return{id:node.id,label:truncate(cleanMemory(node.content),90),kind:node.kind,state:node.state,importance:Number(node.importance||0),used:Number(node.used_count||0),x:p.x,y:p.y,z:(hashNumber(`${node.id}-z`)-.5)*700,clusterX:target.x,clusterY:target.y,clusterZ:target.z};}),links:edges.map(edge=>({source:edge.src_id,target:edge.dst_id,relation:edge.relation,weight:Number(edge.weight||.25),highlighted:edgeMatchesHighlight(edge)}))};
      const instance=new globalThis.ForceGraph3D(container,{controlType:"orbit"});instance.width(size.width).height(size.height).backgroundColor(cssVar("--surface")).showNavInfo(false).graphData(data).nodeId("id").nodeVal(node=>1.4+node.importance*3+Math.log2(1+node.used)).nodeColor(node=>node.id===app.selectedId?cssVar("--violet"):kindColor(node.kind)).nodeOpacity(.88).nodeLabel(node=>app.graph.showLabels?`<div style="padding:6px 8px;background:${cssVar("--surface")};color:${cssVar("--text")};border:1px solid ${cssVar("--border")};max-width:280px">${escapeHtml(node.label)}</div>`:"").linkColor(link=>link.highlighted?cssVar("--violet"):edgeDisplayColor(link.relation,true)).linkOpacity(.52).linkWidth(link=>link.highlighted?4:relationFocus()&&link.relation!==relationFocus()?.08:Math.max(.25,link.weight*1.3)).linkDirectionalParticles(link=>link.highlighted?3:(!relationFocus()||link.relation===relationFocus())&&link.relation==="supersedes"?1:0).linkDirectionalParticleWidth(link=>link.highlighted?3:1.5).cooldownTime(11000).warmupTicks(22).d3VelocityDecay(.26).enableNodeDrag(true).onNodeClick((node,event)=>{selectGraphNode(node.id);if(event?.detail>=2)selectMemory(node.id);}).onNodeDragEnd(node=>{node.fx=node.x;node.fy=node.y;node.fz=node.z;app.graph.positions.set(node.id,{x:node.x,y:node.y});app.graph.customIds.add(node.id);persistGraphPositions();});
      try{instance.d3Force("charge").strength(-62);instance.d3Force("link").distance(link=>54+(1-Number(link.weight||.25))*64);instance.d3Force("cluster",app.graph.clusterMode==="none"?null:makeClusterForce(3,.17));}catch{}
      let fitted=Boolean(app.graph.searchMatchId);instance.onEngineStop(()=>{if(!fitted){fitted=true;instance.zoomToFit(650,70);}});app.graph.instances.constellation=instance;
    }

    function renderMatrix({nodes,edges}) {
      const canvas=document.getElementById("matrix-renderer"),container=canvas.parentElement,size=rendererSize(container),ratio=Math.max(1,window.devicePixelRatio||1);canvas.width=Math.round(size.width*ratio);canvas.height=Math.round(size.height*ratio);canvas.style.width=`${size.width}px`;canvas.style.height=`${size.height}px`;
      const ordered=[...nodes].sort((a,b)=>a.kind.localeCompare(b.kind)||connectionCount(b.id)-connectionCount(a.id)||signalScore(b)-signalScore(a));
      const index=new Map(ordered.map((node,i)=>[node.id,i]));app.graph.matrix={...app.graph.matrix,canvas,context:canvas.getContext("2d"),ratio,width:size.width,height:size.height,nodes:ordered,edges,index};drawMatrix();
      canvas.onmousemove=event=>{const hit=matrixHit(event);if(!hit)return;const source=ordered[hit.row],target=ordered[hit.col];document.getElementById("graph-help").textContent=`Row: ${truncate(cleanMemory(source.content),48)} · Column: ${truncate(cleanMemory(target.content),48)}${matrixHasEdge(source.id,target.id,edges)?" · connected":""}`;};
      canvas.onmouseleave=()=>document.getElementById("graph-help").textContent=rendererDefinitions.matrix.help;
      canvas.onclick=event=>{const hit=matrixHit(event);if(hit)selectGraphNode(ordered[hit.row].id);};
      canvas.ondblclick=event=>{const hit=matrixHit(event);if(hit){const id=ordered[hit.row].id;selectGraphNode(id);selectMemory(id);}};
      canvas.onwheel=event=>{event.preventDefault();const matrix=app.graph.matrix,old=matrix.scale||1,next=Math.max(1,Math.min(18,old*(event.deltaY<0?1.18:1/1.18))),rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;matrix.panX=x-(x-matrix.panX)*next/old;matrix.panY=y-(y-matrix.panY)*next/old;matrix.scale=next;drawMatrix();updateGraphSummary();};
      canvas.onpointerdown=startMatrixPan;app.graph.instances.matrix=app.graph.matrix;
    }

    function drawMatrix() {
      const matrix=app.graph.matrix;if(!matrix.context||!matrix.nodes)return;const{context:ctx,ratio,width,height,nodes,edges,index}=matrix;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);ctx.fillStyle=cssVar("--surface");ctx.fillRect(0,0,width,height);
      const margin=width<620?25:92,baseSize=Math.max(40,Math.min(width-margin-14,height-margin-14)),scale=matrix.scale||1,cell=baseSize/Math.max(1,nodes.length)*scale,x0=margin+(matrix.panX||0),y0=margin+(matrix.panY||0);matrix.plot={x:x0,y:y0,size:baseSize*scale,cell};ctx.save();ctx.beginPath();ctx.rect(margin,margin,baseSize,baseSize);ctx.clip();
      ctx.fillStyle=cssVar("--surface-2");ctx.fillRect(x0,y0,baseSize*scale,baseSize*scale);ctx.globalAlpha=.82;nodes.forEach((node,i)=>{ctx.fillStyle=kindColor(node.kind);ctx.fillRect(x0+i*cell,y0+i*cell,Math.max(1,cell),Math.max(1,cell));});edges.forEach(edge=>{const row=index.get(edge.src_id),col=index.get(edge.dst_id);if(row===undefined||col===undefined)return;const highlighted=edgeMatchesHighlight(edge);ctx.fillStyle=highlighted?cssVar("--violet"):edgeColor(edge.relation);ctx.fillRect(x0+col*cell,y0+row*cell,Math.max(highlighted?4:1.2,cell),Math.max(highlighted?4:1.2,cell));});
      if(app.selectedId&&index.has(app.selectedId)){const selected=index.get(app.selectedId);ctx.fillStyle=cssVar("--violet");ctx.globalAlpha=.18;ctx.fillRect(x0,y0+selected*cell,baseSize*scale,Math.max(2,cell));ctx.fillRect(x0+selected*cell,y0,Math.max(2,cell),baseSize*scale);}ctx.restore();ctx.globalAlpha=1;
      if(width>=620){ctx.font="10px ui-monospace, monospace";ctx.fillStyle=cssVar("--muted");const groups=groupRanges(nodes);groups.forEach(group=>{const mid=x0+(group.start+group.count/2)*cell;ctx.save();ctx.translate(mid,margin-8);ctx.rotate(-Math.PI/3);ctx.fillText(humanize(group.kind),0,0);ctx.restore();ctx.fillText(humanize(group.kind),4,y0+(group.start+group.count/2)*cell);});}
    }

    function matrixHit(event) {const matrix=app.graph.matrix,rect=matrix.canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,plot=matrix.plot;if(!plot||x<plot.x||y<plot.y||x>plot.x+plot.size||y>plot.y+plot.size)return null;const col=Math.floor((x-plot.x)/plot.cell),row=Math.floor((y-plot.y)/plot.cell);return row>=0&&col>=0&&row<matrix.nodes.length&&col<matrix.nodes.length?{row,col}:null;}
    function matrixHasEdge(source,target,edges){return edges.some(edge=>edge.src_id===source&&edge.dst_id===target);}
    function groupRanges(nodes){const groups=[];nodes.forEach((node,index)=>{const last=groups[groups.length-1];if(last&&last.kind===node.kind)last.count++;else groups.push({kind:node.kind,start:index,count:1});});return groups;}
    function startMatrixPan(event){if(event.button!==0)return;const matrix=app.graph.matrix,start={x:event.clientX,y:event.clientY,panX:matrix.panX||0,panY:matrix.panY||0};matrix.canvas.setPointerCapture(event.pointerId);matrix.canvas.onpointermove=move=>{matrix.panX=start.panX+move.clientX-start.x;matrix.panY=start.panY+move.clientY-start.y;drawMatrix();};const finish=()=>{matrix.canvas.onpointermove=null;matrix.canvas.onpointerup=null;matrix.canvas.onpointercancel=null;};matrix.canvas.onpointerup=finish;matrix.canvas.onpointercancel=finish;}

    function graphAllEdges() {
      const rawEdges=(app.data.edges||[]).filter(edge=>edge.explainable!==false).map(edge=>({...edge})),pairKey=edge=>[edge.src_id,edge.dst_id].sort().join("::"),typedPairs=new Set(rawEdges.filter(edge=>edge.relation!=="sleep_replay").map(pairKey));
      const edges=rawEdges.filter(edge=>edge.relation!=="sleep_replay"||!typedPairs.has(pairKey(edge)));
      (app.data.dependencies || []).forEach(edge => edges.push({ src_id:edge.memory_id, dst_id:edge.evidence_id, relation:"dependency", weight:edge.weight, explanation:"This memory was derived from the linked evidence memory.", evidence_type:"recorded_dependency", evidence_records:1, explainable:true }));
      return edges;
    }

    function edgeMatchesHighlight(edge){const highlighted=app.graph.highlightEdge;if(!highlighted)return false;return edge.src_id===highlighted.src_id&&edge.dst_id===highlighted.dst_id&&edge.relation===highlighted.relation;}
    function showConnectionOnMap(edge){if(!edge?.src_id||!edge?.dst_id)return;app.review.lastConnection=edge;app.graph.highlightEdge={src_id:edge.src_id,dst_id:edge.dst_id,relation:edge.relation,explanation:edge.explanation||""};app.graphShowReference=true;document.getElementById("graph-reference").setAttribute("aria-pressed","true");document.getElementById("graph-search").value="";document.getElementById("graph-kind").value="";document.getElementById("graph-state").value="";document.getElementById("graph-relation").value="";app.graph.focusId=edge.src_id;app.selectedId=edge.src_id;setView("graph");setTimeout(fitActiveRenderer,950);}

    function ensureGraphLayout(nodes) {
      const graph = app.graph, groups = unique(nodes.map(row => row.kind));
      groups.forEach((group, groupIndex) => {
        const angle = -Math.PI / 2 + groupIndex * Math.PI * 2 / Math.max(1, groups.length);
        const centerRadius = groups.length === 1 ? 0 : 325;
        const cx = 500 + Math.cos(angle) * centerRadius, cy = 310 + Math.sin(angle) * centerRadius * .72;
        const peers = nodes.filter(node => node.kind === group);
        peers.forEach((node, index) => {
          const saved = graph.positions.get(node.id);
          if (saved && Number.isFinite(Number(saved.x)) && Number.isFinite(Number(saved.y))) {
            graph.positions.set(node.id, { x:Number(saved.x), y:Number(saved.y) }); return;
          }
          const spiral = index * 2.399963229728653 + hashNumber(node.id) * .65;
          const radius = 18 + Math.sqrt(index + 1) * 13.5;
          graph.positions.set(node.id, { x:cx + Math.cos(spiral) * radius, y:cy + Math.sin(spiral) * radius });
        });
      });
    }

    function renderGraphClusterLabels(world, nodes) {
      unique(nodes.map(node => node.kind)).forEach(kind => {
        const positions = nodes.filter(node => node.kind === kind).map(node => app.graph.positions.get(node.id)).filter(Boolean);
        if (!positions.length) return;
        const x = positions.reduce((sum,p) => sum + p.x, 0) / positions.length;
        const y = Math.min(...positions.map(p => p.y)) - 28;
        world.append(svgText(x, y, humanize(kind), "graph-cluster", "middle"));
      });
    }

    function bindGraphPointerEvents(svg) {
      svg.onpointerdown = event => {
        if (event.button !== 0 || event.target.closest(".graph-node-group")) return;
        event.preventDefault();
        app.graph.pan = { pointerId:event.pointerId, startX:event.clientX, startY:event.clientY, panX:app.graph.panX, panY:app.graph.panY };
        svg.setPointerCapture(event.pointerId); document.querySelector(".graph-canvas").classList.add("is-panning");
      };
      svg.onpointermove = handleGraphPointerMove;
      svg.onpointerup = finishGraphPointer;
      svg.onpointercancel = finishGraphPointer;
      svg.onwheel = event => { event.preventDefault(); zoomGraph(event.deltaY < 0 ? 1.12 : 1 / 1.12, graphLocalPoint(event)); };
      svg.ondblclick = event => event.preventDefault();
    }

    function startGraphNodeDrag(event, id, group) {
      if (event.button !== 0) return;
      event.preventDefault(); event.stopPropagation();
      const point = graphWorldPoint(event), position = app.graph.positions.get(id);
      app.graph.drag = { id, group, pointerId:event.pointerId, startX:event.clientX, startY:event.clientY, offsetX:point.x-position.x, offsetY:point.y-position.y, moved:false };
      group.setPointerCapture(event.pointerId); group.classList.add("is-dragging");
    }

    function handleGraphPointerMove(event) {
      const graph = app.graph;
      if (graph.drag && graph.drag.pointerId === event.pointerId) {
        const point = graphWorldPoint(event), moved = Math.hypot(event.clientX-graph.drag.startX,event.clientY-graph.drag.startY) > 3;
        graph.drag.moved = graph.drag.moved || moved;
        graph.positions.set(graph.drag.id, { x:point.x-graph.drag.offsetX, y:point.y-graph.drag.offsetY });
        updateGraphNodePosition(graph.drag.id); return;
      }
      if (graph.pan && graph.pan.pointerId === event.pointerId) {
        const svg = document.getElementById("graph-svg"), rect = svg.getBoundingClientRect();
        graph.panX = graph.pan.panX + (event.clientX-graph.pan.startX) * 1000 / Math.max(1,rect.width);
        graph.panY = graph.pan.panY + (event.clientY-graph.pan.startY) * 620 / Math.max(1,rect.height);
        applyGraphTransform();
      }
    }

    function finishGraphPointer(event) {
      const graph = app.graph;
      if (graph.drag && graph.drag.pointerId === event.pointerId) {
        graph.drag.group.classList.remove("is-dragging");
        if (graph.drag.moved) { graph.suppressClickUntil = Date.now() + 220; graph.customIds.add(graph.drag.id); persistGraphPositions(); }
        try { graph.drag.group.releasePointerCapture(event.pointerId); } catch {}
        graph.drag = null;
      }
      if (graph.pan && graph.pan.pointerId === event.pointerId) {
        try { document.getElementById("graph-svg").releasePointerCapture(event.pointerId); } catch {}
        graph.pan = null; document.querySelector(".graph-canvas").classList.remove("is-panning");
      }
    }

    function updateGraphNodePosition(id) {
      const graph = app.graph, position = graph.positions.get(id), group = graph.nodeEls.get(id);
      if (group && position) group.setAttribute("transform", `translate(${position.x} ${position.y})`);
      graph.edgeEls.forEach(edge => {
        if (edge.src !== id && edge.dst !== id) return;
        const a = graph.positions.get(edge.src), b = graph.positions.get(edge.dst);
        edge.line.setAttribute("x1", a.x); edge.line.setAttribute("y1", a.y); edge.line.setAttribute("x2", b.x); edge.line.setAttribute("y2", b.y);
      });
    }

    function graphLocalPoint(event) {
      const svg = document.getElementById("graph-svg"), point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      const matrix = svg.getScreenCTM(); return matrix ? point.matrixTransform(matrix.inverse()) : {x:500,y:310};
    }

    function graphWorldPoint(event) {
      const point = graphLocalPoint(event), graph = app.graph;
      return { x:(point.x-graph.panX)/graph.scale, y:(point.y-graph.panY)/graph.scale };
    }

    function applyGraphTransform() {
      const graph = app.graph;
      if (graph.world) graph.world.setAttribute("transform", `translate(${graph.panX} ${graph.panY}) scale(${graph.scale})`);
      updateGraphSummary();
    }

    function zoomGraph(factor, anchor={x:500,y:310}) {
      const graph = app.graph, oldScale = graph.scale, nextScale = Math.max(.15,Math.min(5,oldScale*factor));
      const worldX = (anchor.x-graph.panX)/oldScale, worldY = (anchor.y-graph.panY)/oldScale;
      graph.scale = nextScale; graph.panX = anchor.x-worldX*nextScale; graph.panY = anchor.y-worldY*nextScale; applyGraphTransform();
    }

    function fitGraph() {
      const graph = app.graph, points = graph.visibleNodes.map(node => graph.positions.get(node.id)).filter(Boolean);
      if (!points.length) return;
      const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
      const width=Math.max(80,maxX-minX),height=Math.max(80,maxY-minY),scale=Math.max(.15,Math.min(3,Math.min(860/(width+100),500/(height+100))));
      graph.scale=scale;graph.panX=500-(minX+maxX)/2*scale;graph.panY=310-(minY+maxY)/2*scale;applyGraphTransform();
    }

    function fitActiveRenderer() {
      const mode=app.graph.rendererMode,instance=app.graph.instances[mode];
      try{
        if(mode==="manual")fitGraph();
        else if(mode==="atlas")instance?.renderer.getCamera().animatedReset({duration:400});
        else if(mode==="physics")instance?.zoomToFit(450,55);
        else if(mode==="analysis")instance?.fit(undefined,45);
        else if(mode==="constellation")instance?.zoomToFit(600,70);
        else if(mode==="matrix"){app.graph.matrix.scale=1;app.graph.matrix.panX=0;app.graph.matrix.panY=0;drawMatrix();updateGraphSummary();}
      }catch{}
    }

    function zoomActiveRenderer(factor) {
      const mode=app.graph.rendererMode,instance=app.graph.instances[mode];
      try{
        if(mode==="manual")zoomGraph(factor);
        else if(mode==="atlas"){const camera=instance?.renderer.getCamera(),state=camera?.getState();if(camera&&state)camera.animate({...state,ratio:Math.max(.01,state.ratio/factor)},{duration:220});}
        else if(mode==="physics")instance?.zoom(instance.zoom()*factor,220);
        else if(mode==="analysis"&&instance){const extent=instance.extent(),center={x:(extent.x1+extent.x2)/2,y:(extent.y1+extent.y2)/2};instance.zoom({level:Math.max(instance.minZoom(),Math.min(instance.maxZoom(),instance.zoom()*factor)),position:center});}
        else if(mode==="constellation"&&instance){const camera=instance.camera?.();if(camera){camera.position.multiplyScalar(1/factor);camera.updateProjectionMatrix();}}
        else if(mode==="matrix"){app.graph.matrix.scale=Math.max(1,Math.min(18,(app.graph.matrix.scale||1)*factor));drawMatrix();updateGraphSummary();}
      }catch{}
    }

    function resizeActiveRenderer() {
      const mode=app.graph.rendererMode,instance=app.graph.instances[mode],layer=document.getElementById(rendererDefinitions[mode]?.layer||""),size=rendererSize(layer);
      try{
        if(mode==="atlas")instance?.renderer.refresh();
        else if(mode==="physics"||mode==="constellation"){instance?.width(size.width);instance?.height(size.height);}
        else if(mode==="analysis"){instance?.resize();}
        else if(mode==="matrix")renderMatrix({nodes:app.graph.visibleNodes,edges:app.graph.visibleEdges});
      }catch{}
    }

    function resetGraphLayout() {
      app.graph.positions.clear(); app.graph.customIds.clear(); localStorage.removeItem("cortex-graph-positions"); app.graph.focusId=null;
      document.getElementById("graph-focus").setAttribute("aria-pressed","false"); renderGraph(); requestAnimationFrame(fitGraph); showToast("Map layout reset");
    }

    function resetActiveLayout() {
      const mode=app.graph.rendererMode,instance=app.graph.instances[mode];
      if(mode==="manual"||mode==="atlas"){
        app.graph.positions.clear();app.graph.customIds.clear();localStorage.removeItem("cortex-graph-positions");renderActiveGraph();requestAnimationFrame(fitActiveRenderer);showToast("Layout rebuilt");return;
      }
      if(mode==="physics"){instance?.d3ReheatSimulation?.();showToast("Physics simulation reheated");return;}
      if(mode==="analysis"){instance?.layout({name:"cose",animate:true,animationDuration:650,fit:true,padding:45,nodeRepulsion:6500,idealEdgeLength:55})?.run();return;}
      if(mode==="constellation"){instance?.d3ReheatSimulation?.();showToast("3D simulation reheated");return;}
      if(mode==="matrix"){app.graph.matrix.scale=1;app.graph.matrix.panX=0;app.graph.matrix.panY=0;drawMatrix();showToast("Matrix reset");}
    }

    function persistGraphPositions() {
      const saved={};app.graph.customIds.forEach(id=>{const p=app.graph.positions.get(id);if(p)saved[id]={x:Number(p.x.toFixed(2)),y:Number(p.y.toFixed(2))};});
      localStorage.setItem("cortex-graph-positions",JSON.stringify(saved));
    }

    function toggleGraphLabels() {
      app.graph.showLabels=!app.graph.showLabels;localStorage.setItem("cortex-graph-labels",String(app.graph.showLabels));
      document.getElementById("graph-labels").setAttribute("aria-pressed",String(app.graph.showLabels));renderActiveGraph();
    }

    function selectGraphNode(id) {
      app.selectedId=id;
      app.graph.nodeEls.forEach((group,nodeId)=>{
        const existing=group.querySelector(".graph-halo");if(existing)existing.remove();
        if(nodeId===id){const circle=group.querySelector(".graph-node"),radius=Number(circle?.getAttribute("r")||6);group.insertBefore(svgEl("circle",{cx:0,cy:0,r:radius+7,class:"graph-halo"}),group.firstChild);}
      });
      refreshRendererSelection();updateGraphSelection();
    }

    function refreshRendererSelection() {
      const mode=app.graph.rendererMode,instance=app.graph.instances[mode];
      try{
        if(mode==="atlas")instance?.renderer.refresh();
        else if(mode==="physics"||mode==="constellation")instance?.refresh?.();
        else if(mode==="analysis"&&instance){instance.$(":selected").unselect();const node=instance.getElementById(app.selectedId);if(node?.length)node.select();}
        else if(mode==="matrix")drawMatrix();
      }catch{}
    }

    function updateGraphSelection() {
      const memory=(app.data.memories||[]).find(row=>row.id===app.selectedId),host=document.getElementById("graph-selection"),focus=document.getElementById("graph-focus");
      const linksHost=document.getElementById("graph-selection-links"),returnButton=document.getElementById("graph-return-review");returnButton.hidden=!app.graph.highlightEdge;
      if(!memory){host.hidden=true;focus.disabled=true;linksHost.replaceChildren();return;}
      host.hidden=false;focus.disabled=false;document.getElementById("graph-selection-title").textContent=cleanMemory(memory.content);
      const interaction=app.graph.rendererMode==="matrix"?"select its row or column":app.graph.rendererMode==="constellation"?"orbit or drag to explore":"drag to reposition",neighborhood=memory.primary_neighborhood||memory.neighborhoods?.[0]?.label||"Ungrouped";
      const justCreated=app.graph.highlightEdge?` · showing your new ${connectionRelationLabel(app.graph.highlightEdge.relation)} connection`:"";document.getElementById("graph-selection-meta").textContent=`${humanize(memory.kind)} · ${neighborhood} · ${humanize(memory.state)} · ${connectionCount(memory.id)} connections${justCreated} · ${interaction} · double-click for detail`;
      linksHost.replaceChildren();
      const memoryById=new Map((app.data.memories||[]).map(row=>[row.id,row]));
      graphAllEdges().filter(edge=>edge.src_id===memory.id||edge.dst_id===memory.id).sort((a,b)=>Number(edgeMatchesHighlight(b))-Number(edgeMatchesHighlight(a))||Number(b.evidence_records||b.evidence_count||0)-Number(a.evidence_records||a.evidence_count||0)||Number(b.weight||0)-Number(a.weight||0)).slice(0,4).forEach(edge=>{
        const peerId=edge.src_id===memory.id?edge.dst_id:edge.src_id,peer=memoryById.get(peerId),button=el("button","graph-link-reason");button.type="button";
        const evidenceCount=Number(edge.evidence_records||edge.evidence_count||0),evidenceLabel=edge.explainable===false||edge.evidence_type==="legacy_unattributed"?"legacy link; evidence was not preserved":`${humanize(edge.evidence_type)} · ${formatNum(evidenceCount)} evidence record${evidenceCount===1?"":"s"}`;
        button.append(el("strong","",`${edgeMatchesHighlight(edge)?"Just created · ":""}${connectionRelationLabel(edge.relation)} → ${truncate(cleanMemory(peer?.content||peerId),88)}`),el("span","",edge.explanation||"This legacy link predates preserved evidence."),el("span","",evidenceLabel));
        button.addEventListener("click",()=>selectGraphNode(peerId));linksHost.append(button);
      });
      focus.setAttribute("aria-pressed",String(app.graph.focusId===memory.id));focus.textContent=app.graph.focusId===memory.id?"Show all":"Neighbors";
    }

    function toggleGraphFocus() {
      if(!app.selectedId)return;app.graph.focusId=app.graph.focusId===app.selectedId?null:app.selectedId;renderActiveGraph();requestAnimationFrame(fitActiveRenderer);
    }

    function clearGraphFocus() {
      app.graph.focusId=null;app.graph.highlightEdge=null;document.getElementById("graph-search").value="";document.getElementById("graph-kind").value="";document.getElementById("graph-state").value="";document.getElementById("graph-relation").value="";renderActiveGraph();requestAnimationFrame(fitActiveRenderer);
    }

    async function toggleGraphFullscreen() {
      const panel=document.querySelector(".graph-panel");
      try { if(document.fullscreenElement) await document.exitFullscreen(); else await panel.requestFullscreen(); } catch { showToast("Full screen is not available in this browser",true); }
    }

    function updateGraphSummary() {
      const graph=app.graph,summary=document.getElementById("graph-summary");if(!summary)return;
      const focus=graph.focusId?" · neighborhood view":"",zoom=graph.rendererMode==="manual"?` · ${Math.round(graph.scale*100)}%`:graph.rendererMode==="matrix"?` · ${Math.round((graph.matrix.scale||1)*100)}%`:"",relation=graph.highlightEdge?` · showing new ${connectionRelationLabel(graph.highlightEdge.relation)} edge`:relationFocus()?` · highlighting ${humanize(relationFocus())}`:"",cluster=["physics","constellation"].includes(graph.rendererMode)?` · ${graph.clusterMode==="none"?"natural layout":`clustered by ${graph.clusterMode}`}`:"";
      const referenceTotal=Number(app.data?.refinery?.view_counts?.reference||0);
      const roleNote=app.graphShowReference?" · readable + reference":referenceTotal?` · readable only (${formatNum(referenceTotal)} reference hidden)`:"";
      summary.textContent=`${rendererDefinitions[graph.rendererMode]?.label||"Memory map"} · ${formatNum(graph.visibleNodes.length)} records · ${formatNum(graph.visibleEdges.length)} links${roleNote}${cluster}${relation}${zoom}${focus}`;
    }

    function renderTimeline() {
      if(!app.data)return;
      const range=document.getElementById("timeline-range").value,now=Date.now(),cutoff=range==="all"?0:now-Number(range)*86400000,cutoffDay=range==="all"?"":dateKey(new Date(now-(Number(range)-1)*86400000)),detailMemories=app.data.memories||[],fallbackTimeline=[...detailMemories.reduce((map,memory)=>{const day=dateKey(memory.observed_at),key=`${day}:${memory.kind}`,row=map.get(key)||{day,kind:memory.kind,count:0};row.count++;map.set(key,row);return map;},new Map()).values()],timelineSource=Array.isArray(app.data.memory_timeline_by_day_kind)?app.data.memory_timeline_by_day_kind:fallbackTimeline,timelineRows=timelineSource.filter(row=>row.day&&(!cutoffDay||row.day>=cutoffDay)),memories=detailMemories.filter(row=>{const t=Date.parse(row.observed_at);return Number.isFinite(t)&&t>=cutoff;}).sort((a,b)=>Date.parse(a.observed_at)-Date.parse(b.observed_at));
      const access=(app.data.recent_access||[]).filter(row=>Date.parse(row.created_at)>=cutoff),accessEvents=(app.data.access_by_day||[]).filter(row=>Date.parse(`${row.day}T23:59:59Z`)>=cutoff).reduce((sum,row)=>sum+Number(row.count||0),0);
      const captured=timelineRows.reduce((sum,row)=>sum+Number(row.count||0),0),activeDays=unique(timelineRows.map(row=>row.day)).length,kindCounts=timelineRows.reduce((counts,row)=>{counts[row.kind]=(counts[row.kind]||0)+Number(row.count||0);return counts;},{}),topKind=Object.entries(kindCounts).sort((a,b)=>b[1]-a[1])[0];
      metricCards("timeline-metrics",[["Captured",captured,range==="all"?"Across all observed history":`Within the last ${range} calendar days`,"→"],["Days with learning",activeDays,"Days when at least one memory was added","▦"],["Recall events",accessEvents,"Retrieval, selection, injection, and use events in this period","↗"],["Most captured",topKind?humanize(topKind[0]):"—",topKind?`${formatNum(topKind[1])} memories`:"No memories in this period","●"]],true);
      renderTimelineBars(timelineRows);
      renderTimelineTrail(memories,access);
      renderTimelineKinds(kindCounts,captured);
    }

    function renderTimelineBars(timelineRows) {
      const svg=document.getElementById("timeline-svg"),legend=document.getElementById("timeline-legend");svg.replaceChildren();legend.replaceChildren();
      if(!timelineRows.length){svg.append(svgText(410,180,"No memories were captured in this time range","axis-label","middle"));return;}
      const dates=timelineRows.map(row=>new Date(`${row.day}T00:00:00Z`)).filter(date=>!Number.isNaN(date.getTime())),minDate=new Date(Math.min(...dates)),maxDate=new Date(Math.max(...dates)),spanDays=Math.max(1,Math.ceil((maxDate-minDate)/86400000)+1),weekly=spanDays>70,buckets=new Map();
      timelineRows.forEach(item=>{const date=new Date(`${item.day}T00:00:00Z`),key=weekly?weekKey(date):item.day,bucket=buckets.get(key)||{key,date:weekly?startOfWeek(date):date,total:0,kinds:{}},count=Number(item.count||0);bucket.total+=count;bucket.kinds[item.kind]=(bucket.kinds[item.kind]||0)+count;buckets.set(key,bucket);});
      const rows=[...buckets.values()].sort((a,b)=>a.date-b.date),kinds=unique(timelineRows.map(row=>row.kind)).sort(),max=Math.max(1,...rows.map(row=>row.total)),W=820,H=360,L=42,R=14,T=18,B=46,plotW=W-L-R,plotH=H-T-B,slot=plotW/rows.length,barW=Math.max(5,Math.min(44,slot*.68));
      for(let i=0;i<=4;i++){const value=Math.round(max*(4-i)/4),y=T+i*plotH/4;svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,class:"axis-line"}),svgText(L-7,y+4,String(value),"axis-label","end"));}
      rows.forEach((row,index)=>{const x=L+index*slot+(slot-barW)/2;let y=H-B; kinds.forEach(kind=>{const count=row.kinds[kind]||0;if(!count)return;const height=count/max*plotH;y-=height;const rect=svgEl("rect",{x,y,width:barW,height:Math.max(1,height),fill:kindColor(kind),class:"timeline-bar"});const title=svgEl("title");title.textContent=`${weekly?"Week of ":""}${shortDate(row.date)} · ${humanize(kind)}: ${count}`;rect.append(title);svg.append(rect);});if(index%Math.max(1,Math.ceil(rows.length/7))===0||index===rows.length-1)svg.append(svgText(x+barW/2,H-15,row.date.toLocaleDateString(undefined,{month:"short",day:"numeric"}),"axis-label","middle"));});
      kinds.forEach(kind=>{const item=el("span","");const swatch=el("i","legend-swatch");swatch.style.background=kindColor(kind);item.append(swatch,document.createTextNode(humanize(kind)));legend.append(item);});
    }

    function renderTimelineTrail(memories,access) {
      const host=document.getElementById("timeline-trail");host.replaceChildren();
      const events=[...memories.map(memory=>({time:memory.observed_at,type:"captured",memory_id:memory.id,kind:memory.kind,content:memory.content})),...access.map(row=>({time:row.created_at,type:row.event,memory_id:row.memory_id,kind:row.kind,content:row.content}))].sort((a,b)=>Date.parse(b.time)-Date.parse(a.time)).slice(0,12);
      if(!events.length){host.append(emptyState("No activity in this period","Choose a longer range to see earlier captures."));return;}
      events.forEach(event=>{const row=el("div","trail-item");row.tabIndex=0;row.append(el("div","trail-time",compactTime(event.time)),el("div","trail-mark",eventGlyph(event.type)));const copy=el("div","");copy.append(el("div","trail-title",truncate(cleanMemory(event.content),92)),el("div","trail-meta",`${humanize(event.type)} · ${humanize(event.kind)}`));row.append(copy);if(event.memory_id){row.addEventListener("click",()=>selectMemory(event.memory_id));row.addEventListener("keydown",e=>{if(e.key==="Enter")selectMemory(event.memory_id);});}host.append(row);});
    }

    function renderTimelineKinds(counts,total) {
      const host=document.getElementById("timeline-kind-list");host.replaceChildren();const max=Math.max(1,...Object.values(counts));
      Object.entries(counts).sort((a,b)=>b[1]-a[1]).forEach(([kind,count])=>{const row=el("div","timeline-kind-row"),track=el("div","bar-track"),fill=el("div","bar-fill");fill.style.width=`${count/max*100}%`;fill.style.background=kindColor(kind);track.append(fill);row.append(el("span","",humanize(kind)),track,el("strong","",`${formatNum(count)} · ${Math.round(count/Math.max(1,total)*100)}%`));host.append(row);});
    }

    function isConnectionReview(item){return ["association","association_reinforcement"].includes(item?.proposal_kind)&&Boolean(item?.connection_review);}
    function isCreationReview(item){return item?.item_type==="creation"||["memory_creation","creation_candidate"].includes(item?.proposal_kind);}
    function reviewCategory(item){return isCreationReview(item)?"creation":item?.category||"";}
    function reviewItems(){const items=app.data?.review_inbox?.items||[],filter=app.review.filter||"creation";let filtered=filter==="all"?items:items.filter(item=>reviewCategory(item)===filter);if(filter==="connections"&&app.review.connectionPattern)filtered=filtered.filter(item=>item.connection_review?.pattern_key===app.review.connectionPattern);return filtered;}
    function creationReviewData(item){const creation=item.creation||{},details=item.details||{},candidate=creation.candidate||item.candidate||{},assessment=creation.assessment||{},proposedContent=creation.proposed_content||creation.content||item.proposed_content||candidate.content||item.content||item.memories?.[0]?.content||"No proposed text was preserved.",duplicateId=assessment.duplicate_memory_id||"",contradictionIds=Array.isArray(assessment.contradiction_ids)?assessment.contradiction_ids:[],flags=creation.assessment_flags||assessment.quality_flags||assessment.flags||details.flags||item.flags||[];return{
      proposedContent:cleanMemory(proposedContent),
      whatHappened:creation.what_happened||details.what_happened||candidate.source_context||creation.source_context||item.source_context||item.question||"Cortex noticed something that might be useful later.",
      whyNoticed:creation.why_noticed||details.why_noticed||creation.assessment_reason||assessment.reason||details.assessment_reason||(Array.isArray(flags)&&flags.length?flags.map(humanize).join(" · "):"No specific durable-memory signal was preserved."),
      behavior:creation.behavior_if_approved||details.behavior_if_approved||"If approved, this becomes recallable context that the agent may use when a future request appears related.",
      source:creation.source_label||item.source_label||[creation.source_category||item.source_category,creation.source_type||item.source_type,creation.source_ref||item.source_ref].filter(Boolean).map(humanize).join(" · ")||"Source not recorded",
      repeatCount:Number(creation.recurrence_count||creation.repeat_count||item.repeat_count||1),
      positiveFeedback:Number(creation.positive_feedback_count||0),strongFeedback:Number(creation.strong_feedback_count||0),
      duplicates:Array.isArray(creation.duplicates)?creation.duplicates:Array.isArray(details.duplicates)?details.duplicates:duplicateId?[{id:duplicateId,summary:`Existing memory ${shortId(duplicateId)}`}]:[],
      duplicateCount:Number(creation.duplicate_count||details.duplicate_count||item.exact_duplicate_count||(duplicateId?1:0)),
      conflicts:Array.isArray(creation.conflicts)?creation.conflicts:Array.isArray(details.conflicts)?details.conflicts:contradictionIds.map(id=>({id,summary:`Possible conflict ${shortId(id)}`})),
      conflictCount:Number(creation.conflict_count||details.conflict_count||contradictionIds.length),
      redacted:Boolean(creation.redacted),quarantine:creation.quarantine_reason||"",
      neighborhoods:Array.isArray(creation.proposed_neighborhoods)?creation.proposed_neighborhoods:[]
    };}
    function renderCreationReviewGuide(){const guide=document.getElementById("creation-review-guide"),visible=app.review.filter==="creation";guide.hidden=!visible;if(!visible)return;renderRecallSetPanel();const receipt=app.review.creationReceipt,host=document.getElementById("creation-review-receipt");host.hidden=!receipt;if(!receipt)return;document.getElementById("creation-review-receipt-title").textContent=receipt.title||"Decision saved";document.getElementById("creation-review-receipt-copy").textContent=receipt.copy||"The audited decision is now part of Cortex training evidence.";const undo=document.getElementById("creation-review-receipt-undo");undo.hidden=!receipt.reviewId;undo.disabled=app.review.busy;undo.onclick=receipt.reviewId?()=>undoCreationReviewDecision(receipt.reviewId):null;}
    function recallSetPayload(){return app.recallSets.data?.recall_sets||app.recallSets.data||{};}
    function recallSetActiveName(data=recallSetPayload()){return String(data.active_set||data.active?.name||data.current?.name||data.current_set||"legacy").toLowerCase();}
    function recallSetDetails(data,name){const sets=data.sets||{},direct=data[name]||sets[name]||{};return typeof direct==="object"&&direct?direct:{};}
    function firstNumber(...values){const value=values.find(item=>item!==undefined&&item!==null&&item!==""&&!Number.isNaN(Number(item)));return value===undefined?null:Number(value);}
    function displayExactCount(value){return value===null||value===undefined?"—":formatNum(value);}
    function renderRecallSetPanel(){const state=app.recallSets,data=recallSetPayload(),active=recallSetActiveName(data),details=recallSetDetails(data,active),title=document.getElementById("recall-set-title"),copy=document.getElementById("recall-set-copy"),status=document.getElementById("recall-set-status"),preview=document.getElementById("recall-set-preview"),back=document.getElementById("recall-set-switch-back");if(!state.loaded&&!state.loading)loadRecallSets();title.textContent=state.loading?"Checking the active recall set…":state.data?`${humanize(active)} recall set is active`:"Recall-set status is unavailable";copy.textContent=active==="trained"?"The agent’s normal recall is limited to the trained set. Legacy records still exist and remain available if you switch back. No memory content or Hermes profile file was deleted.":"Legacy keeps the current recall behavior. Previewing a clean start shows exactly what the Trained set would include before anything changes.";status.replaceChildren();const eligible=firstNumber(details.eligible_count,details.recallable_count,data.eligible_count,data.recallable_count),proposalCount=firstNumber(data.pending_creation_count,data.creation_proposals?.pending);status.append(el("span","",`Active: ${humanize(active)}`),el("span","",`Recallable: ${displayExactCount(eligible)}`));if(proposalCount!==null)status.append(el("span","",`New-memory decisions: ${displayExactCount(proposalCount)}`));preview.disabled=state.busy||state.loading;preview.textContent=state.busy?"Working…":"Preview a clean start";preview.onclick=previewRecallSet;back.hidden=active!=="trained";back.disabled=state.busy;back.onclick=switchToLegacyRecallSet;}
    async function loadRecallSets(force=false){const state=app.recallSets;if(state.loading||(!force&&state.loaded))return;state.loading=true;renderRecallSetPanel();try{state.data=await authGet("/api/recall-sets");state.loaded=true;}catch(error){state.data=null;state.loaded=true;showToast(`Recall-set status unavailable: ${error.message}`,true);}finally{state.loading=false;renderRecallSetPanel();}}
    function recallSetPreviewCounts(preview){const legacy=preview.legacy||{},trained=preview.trained||{},counts=preview.counts||{};return[["Legacy recallable",firstNumber(preview.legacy_eligible_count,legacy.eligible_count,counts.legacy_eligible)],["Trained on activation",firstNumber(preview.trained_eligible_count,trained.eligible_count,counts.trained_eligible)],["Legacy records held back",firstNumber(preview.excluded_count,preview.held_back_count,counts.excluded)],["Waiting for your review",firstNumber(preview.pending_creation_count,preview.queued_count,counts.pending_creation)]];}
    function showRecallSetPreview(preview){const dialog=document.getElementById("recall-set-dialog"),counts=document.getElementById("recall-set-counts"),copy=document.getElementById("recall-set-dialog-copy");counts.replaceChildren();recallSetPreviewCounts(preview).forEach(([label,value])=>{const card=el("div","recall-set-count");card.append(el("span","",label),el("strong","",displayExactCount(value)));counts.append(card);});copy.textContent=preview.summary||"These are the exact records normal recall would use immediately after activation. Held-back records remain stored in Legacy; new approved memories can join Trained.";document.getElementById("recall-set-dialog-cancel").onclick=()=>dialog.close();const activate=document.getElementById("recall-set-activate");activate.disabled=false;activate.textContent="Activate Trained set";activate.onclick=activateTrainedRecallSet;if(typeof dialog.showModal==="function")dialog.showModal();else dialog.setAttribute("open","");}
    async function previewRecallSet(){const state=app.recallSets;if(state.busy)return;state.busy=true;renderRecallSetPanel();try{const response=await authRequest("/api/recall-sets/preview",{});state.preview=response.preview||response.result||response;showRecallSetPreview(state.preview);}catch(error){showToast(`Could not preview a clean start: ${error.message}`,true);}finally{state.busy=false;renderRecallSetPanel();}}
    async function activateTrainedRecallSet(){const state=app.recallSets;if(state.busy||!state.preview)return;state.busy=true;const button=document.getElementById("recall-set-activate");button.disabled=true;button.textContent="Activating…";try{const previewId=state.preview.preview_id||state.preview.id||"";await authRequest("/api/recall-sets/activate",previewId?{preview_id:previewId}:{});document.getElementById("recall-set-dialog").close();showToast("Trained recall set activated. Legacy remains intact and available to switch back.");state.loaded=false;state.preview=null;await loadRecallSets(true);await load(true);}catch(error){showToast(`Could not activate the Trained set: ${error.message}`,true);button.disabled=false;button.textContent="Activate Trained set";}finally{state.busy=false;renderRecallSetPanel();}}
    async function switchToLegacyRecallSet(){const state=app.recallSets;if(state.busy)return;state.busy=true;renderRecallSetPanel();try{await authRequest("/api/recall-sets/switch",{target:"legacy"});showToast("Switched back to Legacy recall. No memory content was changed.");state.loaded=false;await loadRecallSets(true);await load(true);}catch(error){showToast(`Could not switch recall sets: ${error.message}`,true);}finally{state.busy=false;renderRecallSetPanel();}}
    function connectionMemoryTitle(memory){return cleanMemory(memory?.display_title||memory?.content||"Untitled memory");}
    function connectionMemorySummary(memory){return cleanMemory(memory?.display_summary||memory?.content||"No readable summary is available yet.");}
    function connectionRelationLabel(relation){return({same_subject:"same durable subject",supports:"supports",same_context:"same situation",useful_together:"useful together",operator_link:"operator-approved link",sleep_replay:"replay-supported"})[relation]||humanize(relation||"connection");}
    function renderConnectionReviewGuide(inbox){const guide=document.getElementById("connection-review-guide"),visible=app.review.filter==="connections";guide.hidden=!visible;if(!visible)return;const groups=inbox.connection_groups||[],total=groups.reduce((sum,group)=>sum+Number(group.pending||0),0),patterns=document.getElementById("connection-patterns"),receipt=document.getElementById("connection-map-receipt");document.getElementById("connection-review-count").textContent=formatNum(total);patterns.replaceChildren();const all=el("button",`connection-pattern ${app.review.connectionPattern?"":"active"}`);all.type="button";all.append(el("strong","","All connection patterns"),el("span","",`${formatNum(total)} pairs · review one at a time`));all.addEventListener("click",()=>{app.review.connectionPattern="";app.review.index=0;renderReviewInbox();});patterns.append(all);groups.forEach(group=>{const training=group.training||{},button=el("button",`connection-pattern ${app.review.connectionPattern===group.pattern_key?"active":""}`);button.type="button";button.append(el("strong","",group.pattern_label||"Matching connection pattern"),el("span","",`${formatNum(group.pending||0)} waiting · ${formatNum(training.denied||0)} denied · ${formatNum(training.approved||0)} approved`));button.addEventListener("click",()=>{app.review.connectionPattern=app.review.connectionPattern===group.pattern_key?"":group.pattern_key;app.review.index=0;renderReviewInbox();});patterns.append(button);});if(app.review.lastConnection){const edge=app.review.lastConnection;receipt.hidden=false;document.getElementById("connection-map-receipt-title").textContent=`Created: ${connectionRelationLabel(edge.relation)}`;document.getElementById("connection-map-receipt-copy").textContent=edge.explanation||"Your explained edge is now part of the memory map.";}else receipt.hidden=true;}
    function reviewReasonSuggestions(item,action){const kind=item.proposal_kind||"";
      if(item.item_type==="clarity"){
        if(action==="keep_canonical")return[["reads_clearly","It reads clearly as is"],["personally_confirmed","I know this is right"],["flag_wrong","The readability flag is wrong"]];
        if(action==="keep_reference")return[["raw_material","It is raw technical material"],["lookup_only","Useful only for exact lookup"],["not_a_statement","It is not a memory statement"]];
        if(action==="rewrite")return[["clearer_wording","Clearer wording"],["missing_context","It needed missing context"],["remove_noise","Removed surrounding noise"]];
        if(action==="split")return[["mixed_claims","It mixed unrelated claims"],["one_idea_each","One idea per memory"],["partial_value","Only parts are worth keeping"]];
        if(action==="archive")return[["not_needed_now","Not needed day to day"],["stale","It is outdated"],["history_only","Keep it only for history"]];
        return[["duplicate","It is a duplicate"],["wrong","It is wrong"],["transient","It is temporary noise"],["not_durable","It is not worth remembering"]];
      }
      if(item.item_type==="outcome")return action==="helpful"?[["memory_helped","Memory helped"],["answer_useful","The answer was useful"]]:action==="validated"?[["answer_verified","I checked it"],["source_confirmed","The source was right"]]:action==="harmful"?[["memory_misled","Memory misled the agent"],["wrong_context","Wrong context"]]:[["answer_corrected","I corrected it"],["outdated_memory","The memory was outdated"]];
      if(["association","association_reinforcement"].includes(kind))return action.startsWith("approve_")?[]:[["unrelated","No meaningful relationship"],["co_occurrence_only","Only appeared near each other"],["too_broad","Relationship would be too broad"],["confusing_link","Recalling these together would confuse the agent"]];
      if(item.item_type==="conflict"||kind==="interference_review")return action==="both_valid"?[["different_contexts","Different situations"],["different_times","Different times"],["both_can_be_true","Both can be true"]]:action==="trash_both"?[["both_wrong","Both are wrong"],["both_stale","Both are outdated"],["not_durable","Neither is worth keeping"]]:[["newer_is_current","This one is newer"],["correct_statement","This one is correct"],["other_context_only","The other applies elsewhere"]];
      if(action==="confirm"||action==="keep")return[["personally_confirmed","I know this is right"],["unique_value","This is useful"],["still_current","This is still current"],["worth_remembering","Worth remembering"]];
      if(action==="archive"||action==="quarantine")return[["not_needed_now","Not needed day to day"],["stale","It is outdated"],["superseded","Something newer replaced it"],["history_only","Keep it only for history"]];
      if(action==="trash")return[["duplicate","It is a duplicate"],["wrong","It is wrong"],["transient","It is temporary noise"],["sensitive","It is too private"],["not_durable","It is not worth remembering"]];
      if(action==="approve")return[["expected_change","This change makes sense"],["low_value_link","The current influence is too high"]];
      if(action==="deny")return[["keep_as_is","Keep it as it is"],["not_enough_evidence","There is not enough evidence"]];
      return[];
    }
    function reviewActionDefinitions(item){const kind=item.proposal_kind||"";
      if(item.item_type==="clarity")return[["keep_canonical","Keep as readable memory","Confirm it reads as one clear statement the agent can use normally."],["keep_reference","Keep only as reference","File it as searchable raw evidence instead of a normal memory."],["rewrite","Rewrite clearly","Write the clear version yourself. The raw record stays as linked evidence."],["split","Split into separate memories","Break it into standalone statements, each linked to the raw record."],["archive","Archive it","Stop normal recall but keep its history."],["trash","Move to trash","Stop it from being recalled. You can still restore it later."]];
      if(item.item_type==="outcome")return[["helpful","Helpful","The answer used relevant memory and the result was useful."],["validated","Validated","You independently verified the memory-backed result."],["harmful","Harmful","Recalled memory made the answer or task worse."],["corrected","Corrected","You had to correct the memory-backed answer."]];
      if(item.item_type==="inference")return[["confirm","Keep as true","Keep it available for recall and protect it from automatic cleanup."],["archive","Archive it","Stop normal recall but keep its history."],["trash","Move to trash","Stop it from being recalled. You can still restore it later."]];
      if(item.item_type==="conflict"||kind==="interference_review")return[["keep_first","Memory A is current","Keep A available and archive B as superseded."],["keep_second","Memory B is current","Keep B available and archive A as superseded."],["both_valid","Both are valid","Keep both and record that the difference is contextual."],["trash_both","Trash both","Tombstone both statements while preserving their histories."]];
      if(kind==="association")return[["approve_same_subject","Same durable subject","These are two useful facts about the same specific subject."],["approve_a_supports_b","Memory A supports B","A is useful evidence, context, or instruction for understanding B."],["approve_b_supports_a","Memory B supports A","B is useful evidence, context, or instruction for understanding A."],["approve_useful_together","Useful together","Neither supports the other, but a future task should recall both together."],["deny","Not meaningfully related","Do not add a map connection. This rejection can train this proposal pattern."]];
      if(kind==="association_reinforcement"){
        const relation=item.connection_review?.existing_edge?.relation||"",alreadyTyped=!['','sleep_replay','related','operator_link'].includes(relation);
        if(alreadyTyped)return[["approve_reinforcement",`Strengthen: ${connectionRelationLabel(relation)}`,"The displayed typed relationship is useful and the new evidence should make it more influential."],["deny","Do not strengthen it","Keep the current connection unchanged and train the agent that this evidence is insufficient."]];
        return[["approve_same_subject","Same durable subject","Give this replay-supported pair a meaningful type: two useful facts about the same specific subject."],["approve_a_supports_b","Memory A supports B","Give this pair a direction: A helps explain or support B."],["approve_b_supports_a","Memory B supports A","Give this pair a direction: B helps explain or support A."],["approve_useful_together","Useful together","Neither supports the other, but future tasks should recall both together."],["deny","Not meaningfully related","Do not strengthen the replay-supported link. This rejection can train the proposal pattern."]];
      }
      if(kind==="edge_downscale")return[["approve","Allow lower weight","Reduce this connection’s influence to the displayed proposed weight."],["deny","Keep current weight","Leave the connection unchanged and record why."]];
      if(kind==="dependency_repair")return[["keep","Keep it","Keep it available for recall and protect it from automatic cleanup."],["quarantine","Pause recall","Hide it from the agent until its evidence is repaired."],["trash","Move to trash","Stop it from being recalled. You can still restore it later."]];
      return[["keep","Keep it","Keep it available for recall and protect it from automatic cleanup."],["archive","Archive it","Stop normal recall but keep its history."],["trash","Move to trash","Stop it from being recalled. You can still restore it later."]];
    }
    function reviewSource(memory){return`${humanize(memory.source_category||"unknown")} · ${humanize(memory.source_type||"unknown")}${memory.source_ref?` · ${memory.source_ref}`:""}`;}
    function reviewMap(value){if(!value)return"None recorded";if(typeof value==="string")value=parseJson(value,{});if(Array.isArray(value))return value.join(", ")||"None recorded";return Object.entries(value).map(([key,item])=>`${humanize(key)}=${item}`).join(" · ")||"None recorded";}
    function renderReviewMemory(memory,index,total,readable=false){const card=el("article","review-memory-card"),head=el("div","review-memory-head"),label=total>1?`Memory ${String.fromCharCode(65+index)}`:"Memory under review",open=el("button","","Open full record ↗");open.type="button";open.addEventListener("click",()=>selectMemory(memory.id));head.append(el("span","",`${label} · ${humanize(memory.kind||"memory")} · ${humanize(memory.state||"active")}`),open);card.append(head);if(readable){const presentation=el("div","review-memory-readable");presentation.append(el("strong","",connectionMemoryTitle(memory)),el("p","",connectionMemorySummary(memory)));card.append(presentation);const raw=document.createElement("details");raw.className="review-memory-raw";const rawSummary=document.createElement("summary");rawSummary.textContent="Raw stored text and provenance";raw.append(rawSummary,el("div","review-memory-full",cleanMemory(memory.content||"No content available")));card.append(raw);}else card.append(el("div","review-memory-full",cleanMemory(memory.content||"No content available")));const facts=el("div","review-memory-facts"),positive=Number(memory.helpful_count||0)+Number(memory.validated_count||0),negative=Number(memory.harmful_count||0)+Number(memory.correction_count||0)+Number(memory.false_positive_count||0);[
      ["Source",reviewSource(memory)],["Observed",formatDate(memory.observed_at||memory.created_at)],["Validity",memory.valid_from||memory.valid_to?`${memory.valid_from||"open"} → ${memory.valid_to||"open"}`:"No validity window recorded"],["Scope",reviewMap(memory.scope||memory.scope_json)],["Context",memory.source_context||"No source context recorded"],["Use history",`${formatNum(memory.retrieved_count||0)} recalled · ${formatNum(memory.used_count||0)} used · ${formatNum(positive)} positive · ${formatNum(negative)} negative`],["Signals",`confidence ${Math.round(Number(memory.confidence||0)*100)}% · trust ${Math.round(Number(memory.trust||0)*100)}% · currentness ${Math.round(Number(memory.currentness_confidence||0)*100)}%`],["Memory ID",memory.id||"Unknown"]
    ].forEach(([name,value])=>{const fact=el("div","");fact.append(el("strong","",`${name}: `),document.createTextNode(String(value)));facts.append(fact);});card.append(facts);return card;}
    function renderConnectionBrief(item){const review=item.connection_review||{},memories=item.memories||[],box=el("section","connection-brief");box.append(el("h3","","What relationship would help the agent later?"),el("p","",review.decision_rule||"Approve only when you can name a stable relationship."));const pair=el("div","connection-brief-pair");memories.slice(0,2).forEach((memory,index)=>{if(index)pair.append(el("div","connection-brief-arrow","↔"));const card=el("div","connection-brief-memory");card.append(el("span","",`Memory ${index?"B":"A"}`),el("strong","",connectionMemoryTitle(memory)),el("p","",connectionMemorySummary(memory)));pair.append(card);});box.append(pair);const evidence=review.evidence_summary||{},existing=review.existing_edge,signals=el("div","connection-signal-grid");[["Independent replay",`${formatNum(evidence.distinct_witnesses||0)} witnesses · ${formatNum(evidence.required_witnesses||2)} minimum`],["Helpful co-use",evidence.helpful_co_use?`${formatNum(evidence.helpful_co_use)} observed successful uses`:"None recorded"],["Current connection",existing?`${connectionRelationLabel(existing.relation)} · weight ${Number(existing.weight||0).toFixed(2)}`:"No explained edge yet"]].forEach(([label,value])=>{const signal=el("div","connection-signal");signal.append(el("strong","",label),el("span","",value));signals.append(signal);});box.append(signals);(review.shared_signals||[]).forEach(signal=>{const row=el("div","connection-witness");row.append(el("strong","",signal.label),el("span","",signal.value));box.append(row);});box.append(el("div","connection-caution","Why this is only a proposal: replay or shared wording shows that the memories appeared together. It does not prove they should be recalled together. Your relationship choice supplies that missing meaning."));return box;}
    function creationActionDefinitions(){return[
      ["remember","Remember it","Create this memory as written and allow the agent to recall it."],
      ["remember_edited","Edit before remembering","Change the wording, then create only your edited version."],
      ["reference","Keep as lookup evidence","Keep the source available for exact lookup, outside normal memory recall."],
      ["reject","Don’t make memory","Dismiss this candidate. Nothing becomes recallable."],
      ["needs_context","Need more context","Leave it outside memory and ask Cortex to gather better context first."]
    ];}
    function creationActionLabel(action){return creationActionDefinitions().find(([value])=>value===action)?.[1]||humanize(action);}
    function creationActionEffect(action,scope){const teaching=scope==="policy_evidence"?" Your choice is also saved as one example for a future tested creation standard.":" Cortex will not infer a general rule from this decision.";const effects={remember:"This exact statement becomes a recallable memory.",remember_edited:"Only the edited statement becomes a recallable memory.",reference:"The source stays searchable as lookup evidence, but normal recall will not use it.",reject:"No memory is created from this candidate.",needs_context:"No memory is created yet; the candidate is held for better context."};return`${effects[action]||"Choose an outcome."}${teaching}`;}
    function creationEvidenceText(values,empty){if(!values?.length)return empty;return values.slice(0,3).map(value=>cleanMemory(typeof value==="string"?value:value.summary||value.content||value.id||JSON.stringify(value))).join(" · ");}
    function updateCreationConfirmation(item){const panel=document.getElementById("creation-decision-panel"),summary=document.getElementById("creation-final-summary"),button=document.getElementById("creation-apply-decision"),edit=document.getElementById("creation-edit-content");if(!panel||!summary||!button)return;const action=panel.dataset.action||"",scope=document.querySelector('input[name="creation-decision-scope"]:checked')?.value||"item_only";if(!action){summary.textContent="Choose the outcome that matches what you want.";button.disabled=true;button.textContent="Choose an outcome";return;}if(action==="remember_edited"&&String(edit?.value||"").trim().length<8){summary.textContent="Write the clear memory you want the agent to keep.";button.disabled=true;button.textContent="Finish the edited memory";return;}summary.textContent=creationActionEffect(action,scope);button.disabled=app.review.busy||app.data?.review_writes_enabled!==true;button.textContent=`Confirm: ${creationActionLabel(action)}`;}
    function chooseCreationAction(item,action){const panel=document.getElementById("creation-decision-panel"),followup=document.getElementById("creation-decision-followup"),editWrap=document.getElementById("creation-edit-wrap");if(!panel||!followup)return;panel.dataset.action=action;panel.querySelectorAll(".review-action-option").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.action===action)));followup.hidden=false;if(editWrap)editWrap.hidden=action!=="remember_edited";updateCreationConfirmation(item);requestAnimationFrame(()=>followup.scrollIntoView({behavior:"smooth",block:"nearest"}));}
    function renderCreationReviewFocus(item,index,total){const host=document.getElementById("review-focus"),data=creationReviewData(item),head=el("div","review-focus-head"),copy=el("div",""),meta=el("div","review-sequence",`${index+1} of ${formatNum(total)}`);copy.append(el("div","eyebrow","New memory candidate"),el("h2","",item.title||"Should Cortex remember this?"),el("p","","This text is paused outside memory. Review what happened and what approval would change."));head.append(copy,meta);host.append(head);
      const status=el("div","review-evidence-grid");[["Current stage","Not recallable"],["Source",data.source],["Would live in",data.neighborhoods.join(" · ")||"Cortex will group it after approval"],["Seen",data.repeatCount===1?"Once":`${formatNum(data.repeatCount)} times`],["Positive feedback",data.positiveFeedback?`${formatNum(data.positiveFeedback)} signal${data.positiveFeedback===1?"":"s"}${data.strongFeedback?` · ${formatNum(data.strongFeedback)} strong`:""}`:"None yet"],["Waiting since",formatDate(item.created_at)]].forEach(([label,value])=>{const stat=el("div","review-evidence-stat");stat.append(el("span","",label),el("strong","",value));status.append(stat);});host.append(status);
      const proposal=el("section","creation-proposal");proposal.append(el("span","creation-proposal-label","What Cortex proposes remembering"),el("div","creation-proposal-content",data.proposedContent));host.append(proposal);
      const safetyCopy=data.redacted||data.quarantine?`Cortex removed unsafe or secret content: ${data.quarantine||"sensitive value redacted"}. Edit this into a safe reference such as where the credential is stored; never restore the secret value.`:"No memory row is active, no normal recall can use this text, and no connection can be created from it until you approve it.",evidence=el("section","creation-explanation-grid");[["What happened","The observation behind the candidate",data.whatHappened],["Why Cortex noticed","The signal that made this look durable",data.whyNoticed],["If you remember it","How behavior may change later",data.behavior],["Safety boundary",data.redacted||data.quarantine?"Sensitive value blocked":"What has not happened yet",safetyCopy]].forEach(([label,title,body])=>{const card=el("article","creation-explanation");card.append(el("span","",label),el("strong","",title),el("p","",body));evidence.append(card);});host.append(evidence);
      const checks=el("div","creation-checks"),duplicate=el("div","creation-check"),conflict=el("div","creation-check");duplicate.append(el("span","","Duplicate check"),el("strong","",data.duplicateCount||data.duplicates.length?`${formatNum(Math.max(data.duplicateCount,data.duplicates.length))} possible match${Math.max(data.duplicateCount,data.duplicates.length)===1?"":"es"}: ${creationEvidenceText(data.duplicates,"Open the source before approving.")}`:"No matching memory was reported."));conflict.append(el("span","","Conflict check"),el("strong","",data.conflictCount||data.conflicts.length?`${formatNum(Math.max(data.conflictCount,data.conflicts.length))} possible conflict${Math.max(data.conflictCount,data.conflicts.length)===1?"":"s"}: ${creationEvidenceText(data.conflicts,"Review the source before approving.")}`:"No contradiction was reported."));checks.append(duplicate,conflict);host.append(checks);
      const decision=el("section","review-decision creation-decision");decision.id="creation-decision-panel";decision.dataset.action="";decision.append(el("h3","","What should the agent do with this candidate?"),el("p","","Choose the result in plain language. You never have to pick a reason."));const actions=el("div","review-action-options");creationActionDefinitions().forEach(([action,label,effect])=>{const unsafeDirect=data.redacted&&["remember","reference"].includes(action),button=el("button",`review-action-option ${action==="reject"?"danger":""}`);button.type="button";button.dataset.action=action;button.disabled=unsafeDirect;button.setAttribute("aria-pressed","false");button.append(el("strong","",label),el("span","",unsafeDirect?"Blocked because a secret value was removed. Use Edit before remembering to save only a safe secret-manager reference.":effect));button.addEventListener("click",()=>chooseCreationAction(item,action));actions.append(button);});decision.append(actions);const skip=el("button","ghost-btn","Skip for now");skip.type="button";skip.addEventListener("click",()=>{app.review.index=(app.review.index+1)%Math.max(1,reviewItems().length);renderReviewInbox();});decision.append(skip);
      const followup=el("div","");followup.id="creation-decision-followup";followup.hidden=true;const editWrap=el("div","review-step");editWrap.id="creation-edit-wrap";editWrap.hidden=true;const editLabel=el("div","review-step-head");editLabel.append(el("span","review-step-number","2"),el("strong","","Write the memory the agent should keep"),el("span","","Only this version will be created."));const edit=el("textarea","creation-edit");edit.id="creation-edit-content";edit.setAttribute("aria-label","Edited memory content");edit.value=app.review.creationDrafts[item.proposal_id]??data.proposedContent;edit.addEventListener("input",()=>{app.review.creationDrafts[item.proposal_id]=edit.value;updateCreationConfirmation(item);});editWrap.append(editLabel,edit);followup.append(editWrap);
      const scope=el("fieldset","review-scope review-step"),legend=el("legend","","Should Cortex learn from this decision?"),scopeOptions=el("div","review-scope-options");[["item_only","Just this candidate","Make this decision only. Do not use it to propose a general creation rule."],["policy_evidence","Teach from this example","Make this decision and count it as one example. Repeated examples still require testing and your later approval before becoming a standard."]].forEach(([value,label,description],scopeIndex)=>{const option=el("label","review-scope-option"),input=document.createElement("input"),optionCopy=el("span","");input.type="radio";input.name="creation-decision-scope";input.value=value;input.checked=scopeIndex===0;input.addEventListener("change",()=>updateCreationConfirmation(item));optionCopy.append(el("strong","",label),el("span","",description));option.append(input,optionCopy);scopeOptions.append(option);});scope.append(legend,scopeOptions);followup.append(scope);
      const noteStep=el("div","review-step"),noteHead=el("div","review-step-head");noteHead.append(el("span","review-step-number","3"),el("strong","","Anything else?"),el("span","","Optional. Leave blank if the choice says enough."));const note=el("textarea","review-reason-note");note.id="creation-decision-note";note.setAttribute("aria-label","Optional creation decision note");note.placeholder="Add context in your own words, if useful…";noteStep.append(noteHead,note);followup.append(noteStep);
      const final=el("div","review-final"),finalCopy=el("div","");finalCopy.append(el("strong","","Nothing changes until you confirm"),el("span","","Choose the outcome that matches what you want."));finalCopy.lastChild.id="creation-final-summary";const apply=el("button","primary-btn","Choose an outcome");apply.id="creation-apply-decision";apply.type="button";apply.disabled=true;apply.addEventListener("click",()=>submitCreationReviewDecision(item));final.append(finalCopy,apply);followup.append(final);decision.append(followup);if(app.data?.review_writes_enabled!==true)decision.append(el("div","measurement-boundary","This host is read-only. Set CORTEX_DASHBOARD_REVIEWS=1 to enable reviewed decisions."));host.append(decision);}
    async function submitCreationReviewDecision(item){const panel=document.getElementById("creation-decision-panel"),action=panel?.dataset.action||"",scope=document.querySelector('input[name="creation-decision-scope"]:checked')?.value||"item_only",note=document.getElementById("creation-decision-note")?.value.trim()||"",content=document.getElementById("creation-edit-content")?.value.trim()||"";if(!action){showToast("Choose what should happen first.",true);return;}if(action==="remember_edited"&&content.length<8){showToast("Finish the edited memory first.",true);return;}if(app.review.busy)return;app.review.busy=true;renderReviewInbox();try{const payload={proposal_id:item.proposal_id,action,decision_scope:scope,reason_text:note};if(action==="remember_edited")payload.content=content;const response=await authRequest("/api/review/creation",payload),result=response.result||response,reviewId=result.review_id||response.review_id||"",memoryId=result.memory_id||result.created_memory_id||result.created_memory_ids?.[0]||"",promoted=Boolean(result.memory_promoted),created=Boolean(result.memory_created);const receipts={remember:[promoted?"Legacy memory approved":created?"Memory created":"Memory approved",memoryId?`${promoted?"The preserved memory joined the active trained set":"The approved statement is now recallable"}. Memory ${shortId(memoryId)} is eligible for ordinary recall.`:"The approved statement is now recallable."],remember_edited:["Edited memory created",memoryId?`Only your edited statement became memory ${shortId(memoryId)}.`:"Only your edited statement became recallable."],reference:["Saved as lookup evidence","The source is available for exact lookup and excluded from normal recall."],reject:["Candidate dismissed","No memory was created. The decision remains in the audit trail."],needs_context:["Held for more context","No memory was created. Cortex can return with a better-supported candidate."]};const [title,copy]=receipts[action];app.review.creationReceipt={title,copy:scope==="policy_evidence"?`${copy} This also counted as one creation-training example.`:copy,reviewId};delete app.review.creationDrafts[item.proposal_id];app.review.busy=false;showToast(title);app.recallSets.loaded=false;await Promise.all([load(true),loadRecallSets(true)]);}catch(error){app.review.busy=false;showToast(`Could not save the creation decision: ${error.message}`,true);renderReviewInbox();}}
    async function undoCreationReviewDecision(reviewId){if(!reviewId||app.review.busy)return;app.review.busy=true;renderCreationReviewGuide();try{await authRequest("/api/review/undo",{review_id:reviewId});app.review.creationReceipt=null;showToast("Creation decision reversed. The candidate is back in the inbox.");app.recallSets.loaded=false;await Promise.all([load(true),loadRecallSets(true)]);}catch(error){showToast(`Could not undo the creation decision: ${error.message}`,true);}finally{app.review.busy=false;renderReviewInbox();}}
    function renderReviewInbox(){
      if(!app.data)return;renderPolicyTraining(app.data.policy_training||{});
      const inbox=app.data.review_inbox||{},all=inbox.items||[],items=reviewItems(),total=Number(inbox.total||all.length),nav=document.getElementById("review-nav-count"),totalHost=document.getElementById("review-inbox-total"),queue=document.getElementById("review-queue-list"),focus=document.getElementById("review-focus"),counts=inbox.counts||{},creationCount=all.filter(isCreationReview).length;
      renderCreationReviewGuide();renderConnectionReviewGuide(inbox);nav.textContent=total?formatNum(total):"";totalHost.replaceChildren(el("strong","",formatNum(total)),el("span","",total===1?"decision waiting":"decisions waiting"));
      document.querySelectorAll("[data-review-filter]").forEach(button=>{const category=button.dataset.reviewFilter,count=category==="all"?total:category==="creation"?Number(counts.creation??counts.new_memories??creationCount):Number(counts[category]||0),base=category==="creation"?"New memories":category==="all"?"All":category==="accuracy"?"Answer accuracy":humanize(category);button.textContent=`${base} · ${formatNum(count)}`;button.classList.toggle("active",category===app.review.filter);});
      const queueTitles={creation:"New memories waiting",connections:app.review.connectionPattern?"One connection pattern":"Connection pairs",all:"All pending decisions"};document.getElementById("review-queue-title").textContent=queueTitles[app.review.filter]||`${humanize(app.review.filter)} decisions`;document.getElementById("review-queue-count").textContent=`${formatNum(items.length)} in this view`;queue.replaceChildren();focus.replaceChildren();
      if(!items.length){const empty=el("div","review-empty"),inner=el("div","");const copy=app.review.filter==="creation"?"New candidates stay outside normal recall until an audited operator or automatic judge decision admits them.":app.review.filter==="all"?"There are no operator decisions waiting. New memory, cleanup, connection, conflict, claim, and answer-quality proposals will arrive here.":"Choose another filter or return after new review evidence is created.";inner.append(el("strong","","This part of the inbox is clear"),el("p","",copy));empty.append(inner);focus.append(empty);renderReviewLearning(inbox);return;}
      app.review.index=Math.max(0,Math.min(app.review.index,items.length-1));items.forEach((item,index)=>{const connection=isConnectionReview(item),creation=isCreationReview(item),review=item.connection_review||{},evidence=review.evidence_summary||{},creationData=creation?creationReviewData(item):null,button=el("button",`review-queue-item ${index===app.review.index?"active":""}`);button.type="button";const label=creation?"New memory":connection?(review.pattern_label||"Connection"):humanize(item.category),title=creation?truncate(creationData.proposedContent,100):connection?`${connectionMemoryTitle(item.memories?.[0])} ↔ ${connectionMemoryTitle(item.memories?.[1])}`:item.title,meta=creation?`${timeAgo(new Date(item.created_at))} · not recallable · seen ${formatNum(creationData.repeatCount)}×`:connection?`${formatNum(evidence.distinct_witnesses||0)} independent witnesses · ${formatNum(review.training?.denied||0)} denied / ${formatNum(review.training?.target||5)} · ${formatNum(review.training?.approved||0)} approved`:`${timeAgo(new Date(item.created_at))} · ${formatNum(item.memories?.length||0)} ${item.memories?.length===1?"memory":"memories"}`;button.append(el("span","",label),el("strong","",title),el("small","",meta));button.addEventListener("click",()=>{app.review.index=index;renderReviewInbox();requestAnimationFrame(()=>document.getElementById("review-focus").scrollIntoView({behavior:"smooth",block:"start"}));});queue.append(button);});renderReviewFocus(items[app.review.index],app.review.index,items.length);renderReviewLearning(inbox);
    }
    function reviewScopeOptions(item){
      const memories=item.memories||[],singleMemory=memories.length===1,duplicateCount=Number(item.exact_duplicate_count||0);
      const localLabel=item.item_type==="outcome"?"Only this answer":item.item_type==="conflict"||memories.length>1?"Only this review":"Only this memory";
      const localCopy="Make this one change. No general rule will be learned from it.";
      if(isConnectionReview(item))return[["policy_evidence","Teach this connection pattern","Create or reject only this edge now, and count the judgment toward the five examples needed to propose a tested standard."],["item_only",localLabel,"Create or reject only this edge. Keep the judgment out of connection-pattern training."]];
      const options=[["item_only",localLabel,localCopy]];
      if(singleMemory&&["proposal","clarity"].includes(item.item_type)&&duplicateCount>0)options.push(["exact_duplicates",`All ${formatNum(duplicateCount+1)} exact copies`,`Make the same change to every identical copy shown by Cortex. This will not teach a general rule.`]);
      options.push([
        "policy_evidence",
        "Teach Cortex from this",
        item.item_type==="clarity"
          ? "Save this as labeled evidence for the future role standard. It will not change similar memories or live recall now."
          : "Make this change and save it as one example for similar memories in the future."
      ]);
      return options;
    }
    function updateScopeAvailability(action){
      const input=document.querySelector('input[name="review-decision-scope"][value="exact_duplicates"]');
      if(!input)return;
      const disallowed=["rewrite","split"].includes(action);
      input.disabled=disallowed;
      const option=input.closest(".review-scope-option");
      if(option)option.style.opacity=disallowed?".45":"";
      if(disallowed&&input.checked){const local=document.querySelector('input[name="review-decision-scope"][value="item_only"]');if(local)local.checked=true;}
    }
    function clarityPreviewRecords(){
      const preview=app.review.preview;
      return preview?preview.records.map(record=>({content:String(record.content||"").trim(),kind:record.kind})).filter(record=>record.content):[];
    }
    async function loadClarityPreview(item,action){
      const host=document.getElementById("review-clarity-preview");if(!host)return;
      host.replaceChildren();app.review.preview=null;app.review.previewLoading=true;
      const loading=el("div","clarity-preview");loading.append(el("h4","","Building deterministic preview…"),el("p","","No model call. The starting text is derived from the stored record; you edit and confirm."));host.append(loading);
      updateReviewConfirmation(item);
      try{
        const response=await authRequest("/api/refinery/preview",{kind:action,memory_id:item.memories[0].id});
        app.review.previewLoading=false;
        app.review.preview={action,memoryId:item.memories[0].id,records:(response.preview?.proposed_records||[]).map(record=>({content:String(record.content||""),kind:record.kind}))};
        renderClarityPreview(item,action);
      }catch(error){
        app.review.previewLoading=false;app.review.preview=null;host.replaceChildren();
        const fail=el("div","clarity-preview");fail.append(el("h4","","Could not build the preview"),el("p","",error.message));host.append(fail);
        updateReviewConfirmation(item);
      }
    }
    function renderClarityPreview(item,action){
      const host=document.getElementById("review-clarity-preview");if(!host||!app.review.preview)return;
      host.replaceChildren();
      const box=el("div","clarity-preview");
      box.append(el("h4","",action==="rewrite"?"Editable rewrite preview":"Editable split preview"),el("p","",action==="rewrite"?"Edit the statement until it stands alone. Confirming stores it as your explicit memory; the raw record stays unchanged as linked reference evidence.":"Edit each part until it stands alone, and remove parts that should not become memories. Confirming stores them as your explicit memories; the raw record stays unchanged as linked reference evidence."));
      app.review.preview.records.forEach((record,index)=>{
        const area=document.createElement("textarea");area.value=record.content;area.setAttribute("aria-label",`Proposed memory ${index+1}`);
        area.addEventListener("input",()=>{record.content=area.value;updateReviewConfirmation(item);});
        box.append(area);
        if(action==="split"&&app.review.preview.records.length>2){
          const remove=el("button","ghost-btn clarity-part-remove","Remove this part");remove.type="button";
          remove.addEventListener("click",()=>{app.review.preview.records.splice(index,1);renderClarityPreview(item,action);});
          box.append(remove);
        }
      });
      box.append(el("div","clarity-preview-deps",`Source dependency: every stored memory links back to raw record ${shortId(item.memories[0].id)}, which is kept as reference evidence.`));
      host.append(box);
      updateReviewConfirmation(item);
    }
    function selectedReviewScope(){return document.querySelector('input[name="review-decision-scope"]:checked')?.value||"item_only";}
    function reviewScopeImpact(item,scope,action=""){const duplicateCount=Number(item.exact_duplicate_count||0);if(scope==="exact_duplicates")return`All ${formatNum(duplicateCount+1)} exact copies will change. No rule will be learned.`;if(scope==="policy_evidence"&&isConnectionReview(item)){const training=item.connection_review?.training||{},approved=Number(training.approved||0),denied=Number(training.denied||0),target=Number(training.target||5);return action==="deny"?`This edge stays out. This is denial ${formatNum(denied+1)} of ${formatNum(target)} toward a stricter evidence standard; ${formatNum(approved)} approval${approved===1?" is":"s are"} preserved as counterevidence.`:`This edge is created with the relationship you chose. The approval becomes counterevidence against an overly strict rule; this pattern will have ${formatNum(approved+1)} approved and ${formatNum(denied)} denied.`;}if(scope==="policy_evidence")return item.item_type==="clarity"?"Only this item changes now. The label is saved for the shadow-tested role standard; live recall does not change.":"Only this item changes now; this becomes one training example";return isConnectionReview(item)?"Only this proposed edge changes. The connection-pattern policy will not learn from this choice.":"Only this item changes. No rule will be learned.";}
    function reviewActionLabel(item,action){return reviewActionDefinitions(item).find(([value])=>value===action)?.[1]||humanize(action);}
    function selectedReviewReason(){const panel=document.getElementById("review-decision-panel");return{code:panel?.dataset.reason||"",label:panel?.dataset.reasonLabel||""};}
    function renderReviewReasonSuggestions(item,action){const host=document.getElementById("review-reason-suggestions"),panel=document.getElementById("review-decision-panel");if(!host||!panel)return;host.replaceChildren();panel.dataset.reason="";panel.dataset.reasonLabel="";reviewReasonSuggestions(item,action).forEach(([code,label])=>{const button=el("button","review-reason-chip",label);button.type="button";button.dataset.reasonCode=code;button.setAttribute("aria-pressed","false");button.addEventListener("click",()=>{const selected=panel.dataset.reason===code;panel.dataset.reason=selected?"":code;panel.dataset.reasonLabel=selected?"":label;host.querySelectorAll(".review-reason-chip").forEach(chip=>chip.setAttribute("aria-pressed",String(!selected&&chip===button)));updateReviewConfirmation(item);});host.append(button);});}
    function updateReviewConfirmation(item){const panel=document.getElementById("review-decision-panel"),summary=document.getElementById("review-final-summary"),button=document.getElementById("review-apply-decision");if(!panel||!summary||!button)return;const action=panel.dataset.action||"",reason=selectedReviewReason(),note=document.getElementById("review-reason-note")?.value.trim()||"";if(!action){summary.textContent="Choose what should happen first.";button.disabled=true;button.textContent="Choose an action";return;}
      if(item.item_type==="clarity"&&["rewrite","split"].includes(action)){
        const records=clarityPreviewRecords(),minimum=action==="split"?2:1;
        const ready=app.review.preview&&app.review.preview.action===action&&records.length>=minimum&&records.every(record=>record.content.length>=8);
        if(!ready){summary.textContent=app.review.previewLoading?"Building the editable preview…":action==="split"?"Edit the preview: keep at least two standalone parts.":"Edit the preview until the statement reads clearly.";button.disabled=true;button.textContent="Complete the preview first";return;}
      }
      const actionLabel=reviewActionLabel(item,action),scope=selectedReviewScope(),reasonCopy=reason.label?` Reason: ${reason.label}.`:note?" Your note will be saved.":" No additional reason is required.";summary.textContent=`${actionLabel}. ${reviewScopeImpact(item,scope,action)}${reasonCopy}`;button.disabled=app.review.busy||app.data?.review_writes_enabled!==true;button.textContent=`Confirm: ${actionLabel}`;}
    function chooseReviewAction(item,action){const panel=document.getElementById("review-decision-panel"),followup=document.getElementById("review-decision-followup");if(!panel||!followup)return;panel.dataset.action=action;panel.querySelectorAll(".review-action-option").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.action===action)));followup.hidden=false;renderReviewReasonSuggestions(item,action);updateScopeAvailability(action);const impact=document.getElementById("review-scope-impact");if(impact)impact.textContent=reviewScopeImpact(item,selectedReviewScope(),action);
      if(item.item_type==="clarity"&&["rewrite","split"].includes(action)){loadClarityPreview(item,action);}
      else{app.review.preview=null;app.review.previewLoading=false;const previewHost=document.getElementById("review-clarity-preview");if(previewHost)previewHost.replaceChildren();}
      updateReviewConfirmation(item);requestAnimationFrame(()=>followup.scrollIntoView({behavior:"smooth",block:"nearest"}));}
    function reviewCopilotState(item){const key=String(item.proposal_id||"");if(!app.review.copilotByProposal[key])app.review.copilotByProposal[key]={messages:[],response:null,interpretationId:"",busy:false,draft:"",error:"",applied:false};return app.review.copilotByProposal[key];}
    function resetReviewCopilot(item){delete app.review.copilotByProposal[String(item.proposal_id||"")];renderReviewInbox();}
    function renderReviewCopilot(item){const config=app.data?.review_copilot||{},state=reviewCopilotState(item),section=el("section","review-copilot");section.id="review-copilot";const head=el("div","review-copilot-head"),copy=el("div",""),status=el("span",`review-copilot-status ${config.enabled?"ready":""}`,config.enabled?"Ready":"Unavailable");copy.append(el("div","eyebrow","Review Copilot"),el("h3","","Explain it naturally. The copilot will translate."),el("p","","Describe what feels right or wrong about this pair. The copilot can ask one question, then recommends the relationship and whether your thinking is one-off or a reusable pattern. It cannot apply anything."));head.append(copy,status);section.append(head);
      const thread=el("div","review-copilot-thread");state.messages.forEach(message=>thread.append(el("div",`review-copilot-message ${message.role}`,message.content)));if(state.error)thread.append(el("div","review-copilot-message assistant",state.error));section.append(thread);
      if(state.response?.recommendation){const recommendation=state.response.recommendation,card=el("div","review-copilot-recommendation"),cardHead=el("div","review-copilot-recommendation-head");cardHead.append(el("strong","","Recommendation preview"),el("span","",`${Math.round(Number(recommendation.confidence||0)*100)}% interpreter confidence`));card.append(cardHead);const grid=el("div","review-copilot-grid"),fields=[["What I heard",recommendation.heard],["Recommended action",recommendation.action_label],["Recommended reach",recommendation.scope_label],["Why",recommendation.rationale],["What changes now",recommendation.change_now],["What Cortex learns",recommendation.teaches_cortex]];if(recommendation.general_rule)fields.push(["Proposed general rule",recommendation.general_rule]);if(recommendation.caveat)fields.push(["Uncertainty",recommendation.caveat]);fields.forEach(([label,value])=>{const field=el("div","review-copilot-field");field.append(el("span","",label),el(label==="Recommended action"||label==="Recommended reach"?"strong":"p","",value||"None"));grid.append(field);});card.append(grid);if(recommendation.scope_adjustment)card.append(el("div","review-copilot-scope-adjustment",recommendation.scope_adjustment));const actions=el("div","review-copilot-actions"),use=el("button","primary-btn",state.applied?"Recommendation loaded below":"Use this recommendation"),talk=el("button","ghost-btn","Keep talking"),startOver=el("button","ghost-btn","Start over");use.type=talk.type=startOver.type="button";use.disabled=state.applied;use.addEventListener("click",()=>applyReviewCopilotRecommendation(item));talk.addEventListener("click",()=>document.getElementById("review-copilot-input")?.focus());startOver.addEventListener("click",()=>resetReviewCopilot(item));actions.append(use,talk,startOver);card.append(actions);section.append(card);}
      const compose=el("div","review-copilot-compose"),input=el("textarea","");input.id="review-copilot-input";input.setAttribute("aria-label",state.messages.length?"Reply to Review Copilot":"Tell the copilot what you think about this connection");input.placeholder=state.response?.mode==="clarify"?"Answer the copilot's question in your own words…":"Example: These share a project name, but they solve different problems. This keeps happening whenever two memories mention Python.";input.value=state.draft;input.disabled=!config.enabled||state.busy;input.addEventListener("input",()=>{state.draft=input.value;});input.addEventListener("keydown",event=>{if((event.metaKey||event.ctrlKey)&&event.key==="Enter")askReviewCopilot(item);});const send=el("button","primary-btn",state.busy?"Working…":state.response?.mode==="clarify"?"Send reply":"Ask the agent to interpret");send.type="button";send.disabled=!config.enabled||state.busy;send.addEventListener("click",()=>askReviewCopilot(item));compose.append(input,send);section.append(compose,el("div","review-copilot-privacy",config.enabled?`${config.privacy} Provider: ${config.provider} · Model: ${config.model}. You still confirm the final action below.`:config.reason||"Review Copilot is not configured on this host."));return section;}
    async function askReviewCopilot(item){const state=reviewCopilotState(item),text=String(state.draft||"").trim();if(!text){showToast("Tell the copilot what you think first.",true);return;}if(state.busy)return;state.busy=true;state.error="";renderReviewInbox();try{const response=await authRequest("/api/review/copilot",{proposal_id:item.proposal_id,operator_text:text,conversation:state.messages});state.messages.push({role:"user",content:text});const assistant=[response.message,response.question].filter(Boolean).join("\n\n");state.messages.push({role:"assistant",content:assistant||"I translated your explanation into a preview."});state.response=response;state.interpretationId=response.interpretation_id||"";state.draft="";state.applied=false;state.busy=false;renderReviewInbox();requestAnimationFrame(()=>document.getElementById(response.mode==="clarify"?"review-copilot-input":"review-copilot")?.scrollIntoView({behavior:"smooth",block:"nearest"}));}catch(error){state.busy=false;state.error=`The copilot could not interpret that yet: ${error.message}`;renderReviewInbox();}}
    function applyReviewCopilotRecommendation(item){const state=reviewCopilotState(item),recommendation=state.response?.recommendation;if(!recommendation)return;chooseReviewAction(item,recommendation.action_key);const scope=document.querySelector(`input[name="review-decision-scope"][value="${recommendation.decision_scope}"]`);if(scope&&!scope.disabled)scope.checked=true;const panel=document.getElementById("review-decision-panel"),reasonButton=document.querySelector(`[data-reason-code="${recommendation.reason_code}"]`);if(panel){panel.dataset.reason=recommendation.reason_code||"";panel.dataset.reasonLabel=recommendation.reason_label||"";}document.querySelectorAll(".review-reason-chip").forEach(chip=>chip.setAttribute("aria-pressed",String(chip===reasonButton)));const note=document.getElementById("review-reason-note");if(note)note.value=state.messages.filter(message=>message.role==="user").map(message=>message.content).join(" ").slice(0,1000);const impact=document.getElementById("review-scope-impact");if(impact)impact.textContent=reviewScopeImpact(item,recommendation.decision_scope,recommendation.action_key);state.applied=true;updateReviewConfirmation(item);requestAnimationFrame(()=>document.getElementById("review-final-summary")?.scrollIntoView({behavior:"smooth",block:"nearest"}));}
    function matchingCopilotInterpretation(item,action,apiAction,apiReason,decisionScope){const state=reviewCopilotState(item),recommendation=state.response?.recommendation;if(!state.applied||!state.interpretationId||!recommendation)return"";return recommendation.action_key===action&&recommendation.api_action===apiAction&&recommendation.reason_code===apiReason&&recommendation.decision_scope===decisionScope?state.interpretationId:"";}
    function renderReviewFocus(item,index,total){
      if(isCreationReview(item)){renderCreationReviewFocus(item,index,total);return;}
      const host=document.getElementById("review-focus"),head=el("div","review-focus-head"),copy=el("div",""),meta=el("div","review-sequence",`${index+1} of ${formatNum(total)}`);
      copy.append(el("div","eyebrow",`${humanize(item.category)} · ${humanize(item.proposal_kind||item.item_type)}`),el("h2","",item.title),el("p","",item.question));head.append(copy,meta);host.append(head);
      const connection=isConnectionReview(item);if(connection)host.append(renderConnectionBrief(item));
      const details=item.details||{},stats=el("div","review-evidence-grid"),connectionEvidence=item.connection_review?.evidence_summary||{},training=item.connection_review?.training||{},values=connection?[["Independent witnesses",formatNum(connectionEvidence.distinct_witnesses||0)],["Required to propose",formatNum(connectionEvidence.required_witnesses||2)],["Pattern labels",`${formatNum(training.denied||0)} denied · ${formatNum(training.approved||0)} approved`],["Waiting since",formatDate(item.created_at)]]:[["Proposal confidence",item.score?`${Math.round(Number(item.score)*100)}%`:"Not scored"],["Evidence records",formatNum(item.evidence_count||0)],["Proposed by",item.run_id?`Sleep ${shortId(item.run_id)}`:item.item_type==="outcome"?"Observed task":"Cortex integrity"],["Waiting since",formatDate(item.created_at)]];
      values.forEach(([label,value])=>{const stat=el("div","review-evidence-stat");stat.append(el("span","",label),el("strong","",value));stats.append(stat);});host.append(stats);
      const memorySection=el("section","review-section");memorySection.append(el("h3","",connection?"Readable memories with raw evidence on demand":item.memories?.length>1?"Complete memories being compared":"Complete memory and provenance"),el("p","",connection?"Decide from the readable statements. Expand raw text only when you need to audit the stored record or provenance.":"Nothing below is truncated. Open the full record for versions, dependencies, and every existing link."));const stack=el("div","review-memory-stack");(item.memories||[]).forEach((memory,memoryIndex)=>stack.append(renderReviewMemory(memory,memoryIndex,item.memories.length,connection)));memorySection.append(stack);host.append(memorySection);
      const evidenceSection=el("section","review-section");evidenceSection.append(el("h3","",connection?"Preserved proposal evidence":"Why Cortex surfaced this"));const explanation=el("div","review-witness-list"),proposal=el("div","review-witness",connection?`Cortex's proposal: ${item.question}`:item.question);explanation.append(proposal);if(connection){(item.evidence||[]).forEach((witness,witnessIndex)=>explanation.append(el("div","review-witness",`Independent witness ${witnessIndex+1} · ${humanize(witness.evidence_kind)} · ${formatDate(witness.created_at)}${witness.episode_id?` · replay ${shortId(witness.episode_id)}`:""}`)));}else{Object.entries(details).forEach(([key,value])=>explanation.append(el("div","review-witness",`${humanize(key)}: ${typeof value==="object"?JSON.stringify(value):value}`)));(item.evidence||[]).forEach((witness,witnessIndex)=>explanation.append(el("div","review-witness",`Witness ${witnessIndex+1}: ${humanize(witness.evidence_kind)} · score ${Number(witness.score||0).toFixed(3)}${witness.episode_id?` · episode ${witness.episode_id}`:""} · ${formatDate(witness.created_at)}`)));}if(!(item.evidence||[]).length&&!Object.keys(details).length)explanation.append(el("div","review-witness","No additional supporting record was preserved. Treat that absence as evidence against a strong automatic action."));evidenceSection.append(explanation);host.append(evidenceSection);
      if(connection)host.append(renderReviewCopilot(item));
      const decision=el("section","review-decision");decision.id="review-decision-panel";decision.dataset.action="";decision.dataset.reason="";decision.dataset.reasonLabel="";decision.append(el("h3","",connection?"Confirm or decide manually":"Make one decision"),el("p","",connection?"Use the copilot recommendation above or choose for yourself. Nothing changes until you confirm here.":"Start with the result you want. You can choose reach and reason after that."));
      const actionStep=el("div","review-step"),actionHead=el("div","review-step-head");actionHead.append(el("span","review-step-number","1"),el("strong","","What should happen?"),el("span","","Pick the closest result."));const actionOptions=el("div","review-action-options");reviewActionDefinitions(item).forEach(([action,label,effect])=>{const button=el("button",`review-action-option ${action.includes("trash")||["harmful","corrected"].includes(action)?"danger":""}`);button.type="button";button.dataset.action=action;button.setAttribute("aria-pressed","false");button.append(el("strong","",label),el("span","",effect));button.addEventListener("click",()=>chooseReviewAction(item,action));actionOptions.append(button);});const skip=el("button","ghost-btn",connection?"Skip this pair for now":"Skip this memory for now");skip.type="button";skip.disabled=app.review.busy;skip.addEventListener("click",()=>{app.review.index=(app.review.index+1)%Math.max(1,reviewItems().length);renderReviewInbox();});actionStep.append(actionHead,actionOptions,skip);decision.append(actionStep);
      app.review.preview=null;app.review.previewLoading=false;
      const followup=el("div","");followup.id="review-decision-followup";followup.hidden=true;
      const previewHost=el("div","");previewHost.id="review-clarity-preview";followup.append(previewHost);
      const scopeChoices=reviewScopeOptions(item),initialScope=scopeChoices[0]?.[0]||"item_only",scope=el("fieldset","review-scope review-step"),legend=el("legend","",connection?"2 · Should this decision teach the pattern?":"2 · Where should this apply?"),scopeOptions=el("div","review-scope-options"),impact=el("div","review-scope-impact",reviewScopeImpact(item,initialScope));impact.id="review-scope-impact";scope.append(legend);scopeChoices.forEach(([value,label,description],scopeIndex)=>{const option=el("label","review-scope-option"),input=document.createElement("input"),optionCopy=el("span","");input.type="radio";input.name="review-decision-scope";input.value=value;input.checked=scopeIndex===0;input.setAttribute("aria-label",label);input.addEventListener("change",()=>{impact.textContent=reviewScopeImpact(item,value,decision.dataset.action);updateReviewConfirmation(item);});optionCopy.append(el("strong","",label),el("span","",description));option.append(input,optionCopy);scopeOptions.append(option);});scope.append(scopeOptions,impact);followup.append(scope);
      const reasonStep=el("div","review-step"),reasonHead=el("div","review-step-head");reasonHead.append(el("span","review-step-number","3"),el("strong","","Want to say why?"),el("span","","Optional. If none fit, leave this blank."));const reasonSuggestions=el("div","review-reason-suggestions");reasonSuggestions.id="review-reason-suggestions";const note=el("textarea","review-reason-note");note.id="review-reason-note";note.setAttribute("aria-label","Optional reason in your own words");note.placeholder="Your own reason, if you want to add one…";note.addEventListener("input",()=>updateReviewConfirmation(item));reasonStep.append(reasonHead,reasonSuggestions,note);followup.append(reasonStep);
      const final=el("div","review-final"),finalCopy=el("div","");finalCopy.append(el("strong","","Ready when you are"),el("span","", "Choose what should happen first."));finalCopy.lastChild.id="review-final-summary";const apply=el("button","primary-btn","Choose an action");apply.id="review-apply-decision";apply.type="button";apply.disabled=true;apply.addEventListener("click",()=>submitReviewDecision(item,decision.dataset.action));final.append(finalCopy,apply);followup.append(final);decision.append(followup);
      if(app.data?.review_writes_enabled!==true)decision.append(el("div","measurement-boundary","This host is read-only. Set CORTEX_DASHBOARD_REVIEWS=1 to enable reviewed decisions."));host.append(decision);
    }
    async function submitReviewDecision(item,action){
      const reasonText=document.getElementById("review-reason-note")?.value.trim()||"",selected=selectedReviewReason(),reason=selected.code||(reasonText?"own_words":"operator_choice"),decisionScope=selectedReviewScope();
      if(!action){showToast("Choose what should happen first.",true);return;}if(app.review.busy)return;
      const proposedRecords=item.item_type==="clarity"&&["rewrite","split"].includes(action)?clarityPreviewRecords():[];
      if(item.item_type==="clarity"&&["rewrite","split"].includes(action)&&!proposedRecords.length){showToast("Complete the editable preview before confirming.",true);return;}
      app.review.busy=true;renderReviewInbox();
      try{
        const connectionActions={approve_same_subject:["approve","same_subject"],approve_a_supports_b:["approve","a_supports_b"],approve_b_supports_a:["approve","b_supports_a"],approve_useful_together:["approve","useful_together"],approve_reinforcement:["approve","reinforce_existing"]},translated=connectionActions[action],apiAction=translated?.[0]||action,apiReason=translated?.[1]||reason;
        let path,payload;if(item.item_type==="proposal"){path="/api/review/proposal";payload={proposal_id:item.proposal_id,action:apiAction,reason_code:apiReason,reason_text:reasonText,decision_scope:decisionScope};const interpretationId=matchingCopilotInterpretation(item,action,apiAction,apiReason,decisionScope);if(interpretationId)payload.copilot_interpretation_id=interpretationId;}else if(item.item_type==="clarity"){path="/api/refinery/action";payload={action,memory_id:item.memories[0].id,reason_code:reason,reason_text:reasonText,decision_scope:decisionScope};if(proposedRecords.length)payload.proposed_records=proposedRecords;}else if(item.item_type==="inference"){path="/api/review/inference";payload={memory_id:item.memories[0].id,resolution:action,reason_code:reason,reason_text:reasonText,decision_scope:decisionScope};}else if(item.item_type==="conflict"){path="/api/review/conflict";payload={first_id:item.memories[0].id,second_id:item.memories[1].id,resolution:action,reason_code:reason,reason_text:reasonText,decision_scope:decisionScope};}else{path="/api/outcome/label";payload={task_id:item.task.task_id,outcome:action,reason_code:reason,reason_text:reasonText,decision_scope:decisionScope};}
        const response=await authRequest(path,payload);app.review.busy=false;const affected=Number(response?.result?.affected_memory_ids?.length||0),created=Number(response?.result?.created_memory_ids?.length||0),edge=response?.result?.edge||null;if(edge){app.review.lastConnection=edge;showToast(`Connection created: ${connectionRelationLabel(edge.relation)}. It is now visible on the memory map.`);}
        const clarityTraining=decisionScope==="policy_evidence"?" The label was also saved for shadow-tested role training; live recall was not changed by the label.":"";
        if(item.item_type==="clarity"&&created)showToast(`Saved. ${formatNum(created)} readable ${created===1?"memory":"memories"} created; the raw record stays as linked reference evidence.${clarityTraining} Undo is in the decision history.`);
        else if(item.item_type==="clarity")showToast(`Saved. ${reviewActionLabel(item,action)} applied to ${formatNum(Math.max(1,affected))} record${affected>1?"s":""}.${clarityTraining} Undo is in the decision history.`);
        else if(!edge&&decisionScope==="policy_evidence")showToast(isConnectionReview(item)?"Connection rejected. This pattern gained one training example.":"Saved. This becomes one training example.");else if(!edge&&decisionScope==="exact_duplicates")showToast(`Saved. ${formatNum(affected||Number(item.exact_duplicate_count||0)+1)} exact copies changed; no rule was learned.`);else if(!edge)showToast("Saved. Only this item changed; no rule was learned.");await load(true);if(edge&&app.review.autoMap)showConnectionOnMap(edge);
      }catch(error){app.review.busy=false;showToast(`Could not save decision: ${error.message}`,true);renderReviewInbox();}
    }
    async function undoReviewDecision(reviewId){try{await authRequest("/api/review/undo",{review_id:reviewId});showToast("Decision reversed; the proposal is back in the inbox.");await load(true);}catch(error){showToast(`Could not reverse decision: ${error.message}`,true);}}
    function reviewReasonDisplay(value){return value==="operator_choice"?"No reason added":value==="own_words"?"Reason in your own words":humanize(value||"operator signal");}
    function renderReviewLearning(inbox){
      const signals=document.getElementById("review-learning-signals"),history=document.getElementById("review-history-list"),rows=inbox.recent_decisions||[],active=rows.filter(row=>!row.reversed_at),counts=app.data?.policy_training?.counts||{},training=Number(counts.decisions||0),oneOff=Number(counts.item_only_decisions||0)+Number(counts.exact_duplicate_decisions||0);document.getElementById("review-decision-total").textContent=`${formatNum(training)} training · ${formatNum(oneOff)} one-off`;signals.replaceChildren();history.replaceChildren();const learning=inbox.learning_signals||[];
      if(!learning.length)signals.append(emptyState("No training examples yet","One-off and exact-duplicate decisions are still audited. Reviews marked Teach Cortex appear here; Clarity labels stay shadow-only until the role policy passes evaluation."));
      learning.slice(0,6).forEach(item=>{const [type,action,reason]=String(item.signal||"").split(":"),card=el("article","review-learning-card"),effect=type==="refinery"?"Saved as role-label evidence. Live role-aware recall remains shadow-only.":"Repetition makes this pattern eligible for standards review.";card.append(el("strong","",`${formatNum(item.count)} × ${reviewReasonDisplay(reason)}`),el("span","",`${humanize(type)} · ${humanize(action)}. ${effect}`));signals.append(card);});
      const scopeLabels={item_only:"This review only",exact_duplicates:"Exact duplicates",policy_evidence:"Teach Cortex"};active.slice(0,8).forEach(item=>{const row=el("div","review-history-row"),copy=el("div","");copy.append(el("strong","",`${humanize(item.action)} · ${reviewReasonDisplay(item.reason_code)}`),el("span","",` ${humanize(item.item_type)} · ${scopeLabels[item.decision_scope]||"Legacy training evidence"} · ${formatDate(item.created_at)}`));row.append(copy,el("span","",item.reason_text||"No additional note"));if(item.proposal_id||item.item_type==="refinery"){const undo=el("button","ghost-btn","Undo");undo.type="button";undo.addEventListener("click",()=>undoReviewDecision(item.review_id));row.append(undo);}else row.append(el("span","","Preserved in source workflow"));history.append(row);});
    }

    function renderTrainingGovernance(){
      if(!app.data)return;
      const neighborhood=app.data.review_inbox?.neighborhood_training||{},evaluations=neighborhood.evaluations||[],schemaHost=document.getElementById("schema-evaluation-list");document.getElementById("schema-training-total").textContent=`${formatNum(evaluations.length)} evaluated`;document.getElementById("schema-training-rule").textContent=neighborhood.rule||"Named project and service neighborhoods require repeated approved memories across independent contexts.";schemaHost.replaceChildren();
      if(!evaluations.length)schemaHost.append(emptyState("No named neighborhood candidates","Cortex still uses broad Projects, Services, Tool use, and Credential references. A named map area appears only after it earns enough evidence."));
      evaluations.slice(0,12).forEach(item=>{const row=el("article",`schema-evaluation ${item.decision||"held"}`),copy=el("div","");copy.append(el("strong","",`${humanize(item.category)}: ${item.display_name}`),el("p","",`${item.reason} Evidence: ${formatNum(item.memory_count||0)} memories across ${formatNum(item.context_count||0)} contexts.`));row.append(copy,el("span","",humanize(item.decision)));schemaHost.append(row);});
      const dataset=app.data.learning_experiences||{},counts=dataset.counts||{},total=Object.values(counts).reduce((sum,value)=>sum+Number(value||0),0),experienceHost=document.getElementById("experience-stat-grid");document.getElementById("experience-training-total").textContent=`${formatNum(total)} examples`;experienceHost.replaceChildren();[["Candidates noticed",counts.candidate_observed||0],["Memories selected",counts.memory_selected||0],["Actually used",counts.memory_used||0],["Outcomes labeled",counts.outcome_labeled||0]].forEach(([label,value])=>{const card=el("div","experience-stat");card.append(el("span","",label),el("strong","",formatNum(value)));experienceHost.append(card);});document.getElementById("experience-claim-boundary").textContent=dataset.claim_boundary||"The default export contains no memory text and trains Cortex policies, not the agent’s base model.";
      const filter=document.getElementById("decision-log-filter")?.value||"all",entries=(app.data.decision_log||[]).filter(item=>filter==="all"||item.category===filter),ledger=document.getElementById("decision-ledger-list");ledger.replaceChildren();if(!entries.length)ledger.append(emptyState("No matching decisions yet","Admission, review, map, policy, recall-set, and lifecycle decisions will appear here with their reason and effect."));entries.slice(0,40).forEach(item=>{const row=el("article","decision-ledger-row"),kind=el("div","decision-ledger-kind",`${humanize(item.category)}\n${humanize(item.action)}`),copy=el("div","decision-ledger-copy"),status=item.reversed_at?"Reversed":item.reversible?"Reversible":"Recorded";copy.append(el("strong","",item.summary||humanize(item.action)),el("p","",`${humanize(item.scope||"item only")} · ${status} · affected ${formatNum(item.affected_ids?.length||0)}`));const details=el("details",""),summary=el("summary","","See before, after, and IDs"),pre=el("pre","",JSON.stringify({affected_ids:item.affected_ids||[],before:item.before||{},after:item.after||{}},null,2));details.append(summary,pre);copy.append(details);const meta=el("div","decision-ledger-meta",`${item.actor||"cortex"}\n${formatDate(item.created_at)}`);row.append(kind,copy,meta);ledger.append(row);});
    }

    function policySelectorText(selector){if(selector?.src_kind&&selector?.dst_kind){const endpoint=prefix=>`${humanize(selector[`${prefix}_source_category`]||"matching")} ${humanize(selector[`${prefix}_kind`]||"memory")}`;return`${endpoint("src")} ↔ ${endpoint("dst")}`;}return Object.entries(selector||{}).map(([key,value])=>`${humanize(key)} = ${humanize(value)}`).join(" · ")||"All matching evidence";}
    function policyChangeText(change){return Object.entries(change||{}).map(([key,value])=>`${humanize(key)}: ${typeof value==="number"&&key.includes("adjustment")?signed(value,3):humanize(value)}`).join(" · ")||"No core adjustment";}
    function goToTrainingReviews(domain="") {const filters={connection:"connections",retention:"cleanup",admission:"creation",retrieval:"accuracy"};app.review.filter=filters[domain]||"creation";app.review.index=0;app.graph.highlightEdge=null;setView("review");renderReviewInbox();requestAnimationFrame(()=>document.querySelector(".review-filter-bar")?.scrollIntoView({behavior:"smooth",block:"start"}));}
    function requestPolicyConfirmation(kind,id,scope="") {app.policy.pending={kind,id,scope};renderPolicyTraining(app.data?.policy_training||{});}
    function cancelPolicyConfirmation(){app.policy.pending=null;renderPolicyTraining(app.data?.policy_training||{});}
    async function runPolicyAction(action,payload={}){if(app.policy.busy)return;app.policy.busy=true;renderPolicyTraining(app.data?.policy_training||{});try{const response=await authRequest("/api/policy/action",{action,...payload});app.policy.busy=false;app.policy.pending=null;const filtered=Number(response?.result?.pending_backlog_effect?.filtered||0),messages={compile:"Latest reviews compiled into policy evidence.",replay:"Evidence replay complete. Nothing changed in live Cortex yet.",start_shadow:"Shadow observation started. Keep reviewing matching examples; live policy is still unchanged.",promote:filtered?`Policy activated. ${formatNum(filtered)} matching weak connection proposals left the inbox; rollback will restore them.`:"Policy activated. Cortex is now monitoring the version.",reject:"Policy proposal rejected and preserved in the audit trail.",rollback:"Policy rolled back. Any connection proposals it filtered were restored when still eligible."};showToast(messages[action]||"Policy action saved.");await load(true);}catch(error){app.policy.busy=false;showToast(`Could not update the policy: ${error.message}`,true);renderPolicyTraining(app.data?.policy_training||{});}}
    function handleTrainingNextAction(){const training=app.data?.policy_training||{},action=training.next_action?.action||"review",candidates=training.candidates||[];if(action==="review"){goToTrainingReviews();return;}if(action==="replay"){const candidate=candidates.find(item=>item.stage==="replay_ready");if(candidate)runPolicyAction("replay",{candidate_id:candidate.candidate_id});return;}if(action==="shadow"){const candidate=candidates.find(item=>item.stage==="shadow_ready");if(candidate)runPolicyAction("start_shadow",{candidate_id:candidate.candidate_id});return;}document.getElementById("policy-lab-heading")?.scrollIntoView({behavior:"smooth",block:"start"});}
    function renderPolicyTraining(training){
      document.getElementById("policy-compile").disabled=app.policy.busy||app.data?.review_writes_enabled!==true;
      const progress=Math.round(Number(training.overall_progress||0)*100),progressTrack=document.getElementById("training-progress-track");document.getElementById("training-progress-value").textContent=`${progress}%`;document.getElementById("training-progress-fill").style.width=`${progress}%`;progressTrack.setAttribute("aria-valuenow",String(progress));
      const stepHost=document.getElementById("training-steps");stepHost.replaceChildren();(training.steps||[]).forEach((step,index)=>{const card=el("article",`training-step ${step.status||"waiting"}`),badge=el("span","training-step-index",step.status==="complete"?"✓":String(index+1));card.append(badge,el("strong","",step.label),el("span","",`${step.description} ${formatNum(step.current||0)} / ${formatNum(step.target||0)}`));stepHost.append(card);});
      const next=training.next_action||{};document.getElementById("training-next-title").textContent=next.title||"Review real examples";document.getElementById("training-next-copy").textContent=next.description||"Your choices create the evidence used by the policy compiler.";const nextButton=document.getElementById("training-next-action"),nextLabels={review:"Start reviewing",replay:"Run evidence replay",shadow:"Start shadow observation",approve:"Review policy approval"};nextButton.textContent=nextLabels[next.action]||"Open training work";nextButton.disabled=app.policy.busy||app.data?.review_writes_enabled!==true;
      const candidates=training.candidates||[],candidateHost=document.getElementById("policy-candidate-list");candidateHost.replaceChildren();if(!candidates.length)candidateHost.append(emptyState("No proposed standard yet","Complete five consistent reviews of a similar pattern. Cortex will compile them here automatically without changing live behavior."));
      candidates.filter(item=>!["rejected","promoted"].includes(item.stage)).slice(0,20).forEach(candidate=>{
        const card=el("article",`policy-candidate ${candidate.stage}`),head=el("div","policy-candidate-head"),copy=el("div","");copy.append(el("div","eyebrow",`${humanize(candidate.domain)} policy · ${humanize(candidate.direction)}`),el("h3","",candidate.title),el("p","",candidate.explanation));head.append(copy,el("span","policy-stage",humanize(candidate.stage)));card.append(head);
        const shadow=candidate.shadow||{},replay=candidate.replay||{},gates=el("div","policy-gates"),values=[["Supporting reviews",`${formatNum(candidate.support_count||0)} / ${formatNum(training.thresholds?.scoped_support||5)}`],["Agreement",`${Math.round(Number(candidate.consistency||0)*100)}% / ${Math.round(Number(training.thresholds?.consistency||.8)*100)}%`],["Independent contexts",formatNum(candidate.context_count||0)],["Shadow decisions",`${formatNum(shadow.observation_count||0)} / ${formatNum(training.thresholds?.shadow_observations||3)}`]];values.forEach(([label,value])=>{const gate=el("div","policy-gate");gate.append(el("span","",label),el("strong","",value));gates.append(gate);});card.append(gates,el("div","policy-rule",`Applies when: ${policySelectorText(candidate.selector)}\nProposed core change: ${policyChangeText(candidate.change)}`));
        const notes=[];notes.push(replay.summary||"Evidence replay has not run yet.");if(candidate.stage==="shadow")notes.push(`Shadow mode is watching new matching decisions. ${formatNum(shadow.observation_count||0)} of ${formatNum(shadow.required_observations||3)} recorded.`);if(candidate.core_gate_ready)notes.push("This pattern also meets the broader core-promotion support and context gate.");card.append(el("div","policy-evidence-note",notes.join(" ")));
        const actions=el("div","policy-actions"),writable=app.data?.review_writes_enabled===true&&!app.policy.busy;
        if(candidate.stage==="collecting"||candidate.stage==="paused"){const review=el("button","ghost-btn","Review matching examples");review.type="button";review.disabled=!writable;review.addEventListener("click",()=>goToTrainingReviews(candidate.domain));actions.append(review);}
        if(candidate.stage==="replay_ready"){const replayButton=el("button","primary-btn","Run evidence replay");replayButton.type="button";replayButton.disabled=!writable;replayButton.addEventListener("click",()=>runPolicyAction("replay",{candidate_id:candidate.candidate_id}));actions.append(replayButton);}
        if(candidate.stage==="shadow_ready"){const shadowButton=el("button","primary-btn","Start shadow observation");shadowButton.type="button";shadowButton.disabled=!writable;shadowButton.addEventListener("click",()=>runPolicyAction("start_shadow",{candidate_id:candidate.candidate_id}));actions.append(shadowButton);}
        if(candidate.stage==="shadow"){const review=el("button","primary-btn","Review matching examples");review.type="button";review.disabled=!writable;review.addEventListener("click",()=>goToTrainingReviews(candidate.domain));actions.append(review);}
        if(candidate.stage==="ready"){const scoped=el("button","primary-btn","Approve scoped policy");scoped.type="button";scoped.disabled=!writable;scoped.addEventListener("click",()=>requestPolicyConfirmation("promote",candidate.candidate_id,"scoped"));actions.append(scoped);if(candidate.core_gate_ready){const core=el("button","ghost-btn","Approve as core standard");core.type="button";core.disabled=!writable;core.addEventListener("click",()=>requestPolicyConfirmation("promote",candidate.candidate_id,"core"));actions.append(core);}const reject=el("button","ghost-btn","Reject proposal");reject.type="button";reject.disabled=!writable;reject.addEventListener("click",()=>requestPolicyConfirmation("reject",candidate.candidate_id));actions.append(reject);}
        if(actions.childNodes.length)card.append(actions);
        const pending=app.policy.pending;if(pending?.id===candidate.candidate_id){const confirm=el("div","review-decision"),promoting=pending.kind==="promote",promotionCopy=candidate.domain==="connection"?"Matching future connection proposals will need one more independent witness. Still-pending pairs below that threshold leave this inbox immediately; rollback restores them when they are still eligible.":"The exact bounded adjustment shown above will begin affecting matching core decisions. The version remains auditable and can be rolled back at any time.";confirm.append(el("h3","",promoting?`Activate this ${pending.scope} policy?`:"Reject this proposed standard?"),el("p","",promoting?promotionCopy:"The proposal will leave the active training queue, but its reviews and rejection remain in the audit history."));const buttons=el("div","review-actions"),yes=el("button",promoting?"primary-btn":"ghost-btn",promoting?"Confirm activation":"Confirm rejection"),no=el("button","ghost-btn","Cancel");yes.type=no.type="button";yes.disabled=no.disabled=app.policy.busy;yes.addEventListener("click",()=>runPolicyAction(promoting?"promote":"reject",promoting?{candidate_id:candidate.candidate_id,activation_scope:pending.scope}:{candidate_id:candidate.candidate_id,reason:"operator rejected from Policy Lab"}));no.addEventListener("click",cancelPolicyConfirmation);buttons.append(yes,no);confirm.append(buttons);card.append(confirm);}
        candidateHost.append(card);
      });
      const versions=training.versions||[],active=versions.filter(item=>item.status==="active"),activeHost=document.getElementById("active-policy-list");document.getElementById("active-policy-total").textContent=`${formatNum(active.length)} active`;activeHost.replaceChildren();if(!active.length)activeHost.append(emptyState("No active operator-trained policy yet","Reviews can change individual items now. A core policy appears here only after compilation, replay, shadow observation, and your approval."));active.forEach(version=>{const row=el("article","active-policy"),copy=el("div",""),filtered=Number(version.evidence?.pending_backlog_effect?.filtered||0),effect=filtered?` · removed ${formatNum(filtered)} weak pending proposals (rollback restores eligible ones)`:"";copy.append(el("strong","",`${humanize(version.domain)} · ${humanize(version.activation_scope)} policy`),el("span","",`${policySelectorText(version.selector)} · ${policyChangeText(version.change)}${effect} · activated ${formatDate(version.activated_at)}`));const controls=el("div","review-actions"),pending=app.policy.pending;if(pending?.id===version.version_id&&pending.kind==="rollback"){const yes=el("button","ghost-btn danger","Confirm rollback"),no=el("button","ghost-btn","Cancel");yes.type=no.type="button";yes.disabled=no.disabled=app.policy.busy;yes.addEventListener("click",()=>runPolicyAction("rollback",{version_id:version.version_id,reason:"operator rollback from Policy Lab"}));no.addEventListener("click",cancelPolicyConfirmation);controls.append(yes,no);}else{const rollback=el("button","ghost-btn","Rollback");rollback.type="button";rollback.disabled=app.policy.busy||app.data?.review_writes_enabled!==true;rollback.addEventListener("click",()=>requestPolicyConfirmation("rollback",version.version_id));controls.append(rollback);}row.append(copy,controls);activeHost.append(row);});document.getElementById("policy-claim-boundary").textContent=training.claim_boundary||"Reviews generate proposals; explicit promotion is required before a policy can affect the core.";
    }

    function renderInsights() {
      if(!app.data)return;
      const d=app.data,memories=d.memories||[],active=memories.filter(memory=>memory.state==="active"),retrieved=active.filter(memory=>Number(memory.retrieved_count||0)>0),selected=active.filter(memory=>Number(memory.selected_count||0)>0),used=active.filter(memory=>Number(memory.used_count||0)>0),edges=graphAllEdges(),connectedIds=new Set(edges.flatMap(edge=>[edge.src_id,edge.dst_id])),connected=active.filter(memory=>connectedIds.has(memory.id)),tools=d.tools||[],attempts=tools.reduce((sum,tool)=>sum+toolAttempts(tool),0),successes=tools.reduce((sum,tool)=>sum+Number(tool.success_count||0),0);
      const coverage=percent(retrieved.length,active.length),useThrough=percent(used.length,Math.max(1,retrieved.length)),networkCoverage=percent(connected.length,active.length),toolRate=percent(successes,attempts);
      metricCards("insight-metrics",[["Recall coverage",`${coverage}%`,`${formatNum(retrieved.length)} of ${formatNum(active.length)} active memories have been recalled`,"↗"],["Use-through",`${useThrough}%`,`${formatNum(used.length)} recalled memories were confirmed used`,"✓"],["Network coverage",`${networkCoverage}%`,`${formatNum(connected.length)} active memories have at least one link`,"⌘"],["Tool reliability",attempts?`${toolRate}%`:"—",attempts?`${formatNum(successes)} successes across ${formatNum(attempts)} observations`:"Waiting for tool outcomes","◇"]],true);
      const sourceCounts=countBy(memories,"source_category"),documentCount=Number(sourceCounts.DOCUMENT_EXTRACTED||0),observations=[
        [coverage<25?"Cortex is still calibrating":"Recall is spreading through the index",coverage<25?`${100-coverage}% of active memories have not been requested yet. That is expected soon after a large vault import; useful coverage grows as Hermes handles more varied work.`:`${coverage}% of active memories have now participated in recall.`],
        ["The vault is the main knowledge source",`${percent(documentCount,memories.length)}% of memories were extracted from indexed documents. Live conversations and verified tool outcomes will gradually balance that mix.`],
        [networkCoverage<50?"Many memories are still isolated":"The association graph has broad coverage",networkCoverage<50?`${formatNum(active.length-connected.length)} active memories do not yet have a graph connection. Co-use and repeated context can create stronger links over time.`:`${networkCoverage}% of active memories are connected to at least one neighbor.`],
        [attempts?"Tool learning has evidence to work with":"Tool learning needs more outcomes",attempts?`${formatNum(attempts)} tool outcomes currently support ${formatNum(tools.length)} learned strategies at ${toolRate}% observed success.`:"As Hermes calls tools, Cortex will retain task-specific successes and failures here."]
      ];
      const noteHost=document.getElementById("insight-observations");noteHost.replaceChildren();observations.forEach(([title,copy],index)=>{const item=el("div","insight-note");item.append(el("div","insight-index",String(index+1).padStart(2,"0")));const text=el("div","");text.append(el("strong","",title),el("p","",copy));item.append(text);noteHost.append(item);});
      renderFunnel([["Available",active.length],["Recalled",retrieved.length],["Selected",selected.length],["Used",used.length]],active.length);
      renderMemoryRanking("underused-list",active.filter(memory=>!Number(memory.retrieved_count||0)).sort((a,b)=>signalScore(b)-signalScore(a)).slice(0,6),memory=>`${humanize(memory.kind)} · signal ${Math.round(signalScore(memory)*100)}`);
      renderMemoryRanking("hub-list",[...active].sort((a,b)=>connectionCount(b.id)-connectionCount(a.id)).slice(0,6),memory=>`${connectionCount(memory.id)} connections · ${humanize(memory.kind)}`);
      renderInsightSources(sourceCounts,memories.length);
      renderInsightTools(tools);
      renderBenchmark(d.benchmarks||{});
      renderOutcomeLab(d.outcome_lab||{},d.evaluations||{});
      renderCapacityImpactChart();
      renderTrendChart();
    }

    function renderBenchmark(benchmark) {
      const runs=Array.isArray(benchmark.runs)?benchmark.runs:[],active=benchmark.active||null,latest=benchmark.latest||null,delta=benchmark.delta||null,result=benchmark.latest_result||{},summary=result.dashboard_summary||latest||{},button=document.getElementById("benchmark-start"),status=document.getElementById("benchmark-status"),progress=document.getElementById("benchmark-progress"),current=active||runs[0]||null;
      button.disabled=Boolean(active);button.textContent=active?"Benchmark running…":"Run benchmark";
      status.textContent=active?`${humanize(active.phase)} · ${Math.round(Number(active.progress||0))}%`:latest?`Score ${Number(latest.score||0).toFixed(1)}`:current?.status==="failed"?"Last run failed":"Not run";
      progress.hidden=!(active||current?.status==="failed");
      if(!progress.hidden){const progressValue=Math.max(0,Math.min(100,Number((active||current).progress||0)));document.getElementById("benchmark-progress-title").textContent=active?humanize(active.phase||"running"):"Benchmark stopped";document.getElementById("benchmark-progress-message").textContent=String((active||current).error||(active||current).message||"Benchmark is running.");document.getElementById("benchmark-progress-value").textContent=`${Math.round(progressValue)}%`;document.getElementById("benchmark-progress-fill").style.width=`${progressValue}%`;}
      const summaryHost=document.getElementById("benchmark-summary");summaryHost.replaceChildren();
      const stats=latest?[
        ["Overall score",`${Number(latest.score||0).toFixed(1)} / 100`,delta?`${signed(delta.score,1)} vs previous compatible run`:"First compatible baseline"],
        ["Recall@6",`${(Number(latest.recall_at_k||0)*100).toFixed(1)}%`,`${formatNum(latest.queries||0)} labeled questions at ${formatNum(latest.corpus_memories||0)} memories`],
        ["MRR",Number(latest.mrr||0).toFixed(3),"1.000 means the correct memory ranked first every time"],
        ["p95 retrieval",`${Number(latest.p95_ms||0).toFixed(1)} ms`,`${Number(latest.p95_ms||0)<=50?"Inside":"Above"} the 50 ms local gate`]
      ]:[
        ["Overall score","—","Run the fixed suite to create a baseline"],["Recall@6","—","Correct memory present in the first six"],["MRR","—","How close the answer ranks to first"],["p95 retrieval","—","Local retrieval overhead, excluding the model"]
      ];
      stats.forEach(([label,value,note])=>{const card=el("div","benchmark-stat");card.append(el("span","",label),el("strong","",value),el("small","",note));summaryHost.append(card);});
      renderBenchmarkHistory(runs,latest);
      const recommendationHost=document.getElementById("benchmark-recommendations"),recommendations=Array.isArray(summary.recommendations)?summary.recommendations:[];recommendationHost.replaceChildren();
      if(!recommendations.length)recommendationHost.append(emptyState("Run the benchmark first","Measured guidance will separate retrieval coverage, ranking, latency, and context efficiency."));
      recommendations.forEach(item=>{const card=el("article",`benchmark-recommendation ${item.state||"watch"}`);card.append(el("span","",`${item.area||"Measured lever"} · ${item.state||"watch"}`),el("strong","",item.title||"Review this metric"),el("p","",item.explanation||""),el("small","",item.action||""));recommendationHost.append(card);});
      const runHost=document.getElementById("benchmark-run-list");runHost.replaceChildren();
      if(!runs.length)runHost.append(emptyState("No benchmark history yet","The first completed run becomes the baseline for later code and host changes."));
      runs.slice(0,8).forEach(run=>{const row=el("div","benchmark-run-row"),identity=el("div","");identity.append(el("strong","",formatDate(run.started_at)),el("span","",`${humanize(run.status)} · suite v${run.suite_version}`));row.append(identity,el("strong","",run.status==="completed"?Number(run.score||0).toFixed(1):"—"),el("span","",run.status==="completed"?`${(Number(run.recall_at_k||0)*100).toFixed(1)}% recall`:humanize(run.phase)),el("span","",run.status==="completed"?`${Number(run.p95_ms||0).toFixed(1)} ms p95`:String(run.error||run.message||"")));runHost.append(row);});
      const method=document.getElementById("benchmark-method"),environment=result.environment||{},suite=result.dashboard_suite||{};method.replaceChildren(el("strong","","Overall score: "),document.createTextNode("85% retrieval quality and 15% local p95 speed. Retrieval quality is 65% recall@6 and 35% MRR. Speed gets full credit at 50 ms p95 or faster. Context size stays a separate guardrail so returning nothing cannot game the score."));
      if(latest){method.append(document.createElement("br"),document.createElement("br"),el("strong","","Latest run: "),document.createTextNode(`suite v${latest.suite_version} · ${formatNum(latest.corpus_memories)} memories · ${formatNum(latest.queries)} labeled queries · Python ${environment.python||"unknown"}${suite.seed!==undefined?` · seed ${suite.seed}`:""}. Compare only compatible suite versions on similar host load.`));}
      if(active)scheduleBenchmarkPoll();
    }

    function renderBenchmarkHistory(runs,latest) {
      const svg=document.getElementById("benchmark-history-chart"),legend=document.getElementById("benchmark-legend"),note=document.getElementById("benchmark-chart-note"),version=latest?.suite_version||"1",rows=runs.filter(row=>row.status==="completed"&&String(row.suite_version)===String(version)).slice(0,24).reverse();svg.replaceChildren();legend.replaceChildren();document.getElementById("benchmark-run-total").textContent=`${formatNum(rows.length)} compatible ${rows.length===1?"run":"runs"}`;
      const series=[{label:"Overall score",color:cssVar("--violet"),note:latest?Number(latest.score||0).toFixed(1):"—"},{label:"Recall@6",color:cssVar("--green"),note:latest?`${(Number(latest.recall_at_k||0)*100).toFixed(1)}%`:"—"},{label:"p95 retrieval",color:cssVar("--amber"),note:latest?`${Number(latest.p95_ms||0).toFixed(1)} ms`:"—"}];series.forEach(item=>{const entry=el("span","trend-legend-item"),swatch=el("i","");swatch.style.background=item.color;entry.append(swatch,el("span","",item.label),el("strong","",item.note));legend.append(entry);});
      const W=900,H=430,L=64,R=26,topT=34,topH=205,bottomT=292,bottomH=78,plotW=W-L-R,x=index=>rows.length===1?L+plotW/2:L+index*plotW/(rows.length-1),qualityY=value=>topT+(100-value)*topH/100,maxLatency=Math.max(100,...rows.map(row=>Number(row.p95_ms||0))),latencyY=value=>bottomT+(maxLatency-value)*bottomH/maxLatency;
      const desc=svgEl("desc");desc.textContent="Version-compatible synthetic benchmark runs. The upper panel shows score and recall on a 0 to 100 scale; the lower panel shows local p95 retrieval milliseconds.";svg.append(desc);
      if(!rows.length){svg.append(svgText(W/2,185,"Run the benchmark to create the first measured point","chart-title","middle"),svgText(W/2,214,"The suite uses synthetic facts and does not read production memory","chart-label","middle"));note.textContent="One run gives a measured baseline. Eight compatible runs are recommended before reading the line as a trend.";return;}
      for(let index=0;index<5;index++){const value=100-index*25,y=topT+index*topH/4;svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,class:"chart-grid"}),svgText(L-10,y+4,`${value}`,"chart-label","end"));}
      const recallTarget=qualityY(98);svg.append(svgEl("line",{x1:L,y1:recallTarget,x2:W-R,y2:recallTarget,class:"benchmark-target"}),svgText(W-R,recallTarget-7,"98% recall target","chart-label","end"));
      svg.append(svgText(L,topT-13,"Quality · 0–100","chart-label"),svgEl("line",{x1:L,y1:bottomT+bottomH,x2:W-R,y2:bottomT+bottomH,class:"chart-axis"}),svgText(L,bottomT-14,`Speed · p95 milliseconds (0–${Math.ceil(maxLatency)})`,"chart-label"));
      const latencyTarget=latencyY(50);svg.append(svgEl("line",{x1:L,y1:latencyTarget,x2:W-R,y2:latencyTarget,class:"benchmark-target"}),svgText(W-R,latencyTarget-7,"50 ms gate","chart-label","end"));
      const scorePoints=rows.map((row,index)=>({x:x(index),y:qualityY(Number(row.score||0)),value:Number(row.score||0),row})),recallPoints=rows.map((row,index)=>({x:x(index),y:qualityY(Number(row.recall_at_k||0)*100),value:Number(row.recall_at_k||0)*100,row})),latencyPoints=rows.map((row,index)=>({x:x(index),y:latencyY(Number(row.p95_ms||0)),value:Number(row.p95_ms||0),row}));
      if(rows.length>=8){[[scorePoints,"--violet","benchmark-score-line"],[recallPoints,"--green","benchmark-recall-line"],[latencyPoints,"--amber","benchmark-latency-line"]].forEach(([points,color,className])=>svg.append(svgEl("path",{d:points.map((point,index)=>`${index?"L":"M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),stroke:cssVar(color),class:className})));}
      [[scorePoints,"--violet","score"],[recallPoints,"--green","recall"],[latencyPoints,"--amber","latency"]].forEach(([points,color,label])=>points.forEach(point=>{const dot=svgEl("circle",{cx:point.x,cy:point.y,r:4.2,fill:cssVar("--surface"),stroke:cssVar(color),"stroke-width":"2.2"}),title=svgEl("title");title.textContent=`${formatDate(point.row.started_at)} · ${label} ${label==="latency"?`${point.value.toFixed(1)} ms`:point.value.toFixed(1)}`;dot.append(title);svg.append(dot);}));
      const labelEvery=Math.max(1,Math.ceil(rows.length/6));rows.forEach((row,index)=>{if(index%labelEvery===0||index===rows.length-1)svg.append(svgText(x(index),H-18,shortDate(row.started_at),"chart-label","middle"));});
      note.textContent=rows.length===1?"This is one measured baseline, not a trend. Re-run after code or host changes with similar load.":rows.length<8?`${rows.length} compatible runs are shown as individual comparisons. Reach eight before interpreting a trend line.`:"Lines connect compatible suite-version runs. Host load can move latency, so read quality and speed together.";
    }

    async function startBenchmark() {
      const button=document.getElementById("benchmark-start");button.disabled=true;
      try{const benchmark=await authRequest("/api/benchmark/start",{suite:"standard"});app.data.benchmarks=benchmark;renderBenchmark(benchmark);showToast("Benchmark started on this Cortex host; it usually takes about 10–20 seconds.");scheduleBenchmarkPoll();}
      catch(error){button.disabled=false;showToast(`Could not start benchmark: ${error.message}`,true);}
    }

    function scheduleBenchmarkPoll(){if(benchmarkPollTimer)return;benchmarkPollTimer=setTimeout(()=>{benchmarkPollTimer=null;pollBenchmarkStatus();},800);}

    async function pollBenchmarkStatus(){try{const response=await fetch("/api/benchmark/status",{cache:"no-store",credentials:"same-origin"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const benchmark=await response.json();app.data.benchmarks=benchmark;renderBenchmark(benchmark);if(benchmark.active)scheduleBenchmarkPoll();else await load(true);}catch(error){showToast(`Benchmark status unavailable: ${error.message}`,true);}}

    function renderOutcomeLab(lab,evaluation) {
      const summary=document.getElementById("outcome-summary"),tasks=Array.isArray(lab.tasks)?lab.tasks:[],eligible=Number(lab.eligible_tasks||0),labeled=Number(lab.labeled_tasks||0),positive=Number(lab.positive_count||0),negative=Number(lab.negative_count||0),coverage=Number(lab.label_coverage||0),observed=lab.observed_helpfulness,writeEnabled=app.data?.review_writes_enabled===true;
      summary.replaceChildren();[
        ["Label coverage",eligible?`${(coverage*100).toFixed(1)}%`:"—",`${formatNum(labeled)} of ${formatNum(eligible)} used recall tasks`],
        ["Observed helpfulness",observed===null||observed===undefined?"—":`${(Number(observed)*100).toFixed(1)}%`,`${formatNum(positive)} positive · ${formatNum(negative)} negative labels`],
        ["Private cases",`${formatNum(evaluation.case_count||lab.evaluation_case_count||0)} / ${formatNum(evaluation.minimum_cases||lab.evaluation_min_cases||8)}`,evaluation.ready?"Ready for paired evaluation":"Helpful or validated tasks become cases"],
        ["Causal status","Observational","Labels describe outcomes; they do not isolate the cause"]
      ].forEach(([label,value,note])=>{const card=el("div","outcome-stat");card.append(el("span","",label),el("strong","",value),el("small","",note));summary.append(card);});
      document.getElementById("outcome-task-total").textContent=`${formatNum(eligible)} eligible`;
      document.getElementById("outcome-write-status").textContent=writeEnabled?"Audited labels enabled":"Read-only on this host";
      const host=document.getElementById("outcome-task-list");host.replaceChildren();
      if(!tasks.length){host.append(emptyState("No attributed tasks yet","Use the agent on a memory-bearing task. Once the response uses recalled evidence, it becomes eligible for an outcome label."));}
      tasks.slice(0,12).forEach(task=>{const card=el("article","outcome-task"),head=el("div","outcome-task-head"),query=String(task.query||"Private recall task"),current=String(task.label_outcome||"");head.append(el("strong","",truncate(query,150)),el("span","",`${humanize(task.task_type||"general")} · ${timeAgo(new Date(task.created_at))}`));card.append(head,el("p","",`${formatNum(task.used_count||0)} attributed of ${formatNum(task.selected_count||0)} selected · ${humanize(task.recall_mode||"unknown")} recall · ${formatNum(task.estimated_tokens||0)} estimated tokens`));const memories=el("div","outcome-memory-strip");(task.memories||[]).slice(0,5).forEach(memory=>{const button=el("button","",truncate(cleanMemory(memory.content),72));button.type="button";button.title=cleanMemory(memory.content);button.addEventListener("click",()=>selectMemory(memory.memory_id));memories.append(button);});card.append(memories);if(task.why_recalled){const why=el("details","outcome-why"),summary=el("summary","","Why these memories were recalled"),list=el("div","outcome-why-list");(task.why_recalled.candidates||[]).slice(0,12).forEach(candidate=>{const row=el("div",`outcome-why-row ${candidate.selected?"selected":"rejected"}`);row.append(el("strong","",`${candidate.selected?"Used in context":"Not selected"} · ${humanize(candidate.kind||"memory")} · score ${Number(candidate.score||0).toFixed(3)}`),el("span","",candidate.reason||"No selection reason recorded."),el("p","",truncate(cleanMemory(candidate.content_preview||candidate.memory_id||"Memory"),150)));list.append(row);});why.append(summary,el("p","",task.why_recalled.reason||"Cortex ran bounded retrieval for this task."),list);card.append(why);}const actions=el("div","outcome-actions");[["helpful","Helpful"],["validated","Validated"],["harmful","Harmful"],["corrected","Corrected"]].forEach(([outcome,label])=>{const button=el("button",`outcome-choice ${["harmful","corrected"].includes(outcome)?"negative":""} ${current===outcome?"active":""}`,label);button.type="button";button.disabled=!writeEnabled;button.addEventListener("click",()=>labelOutcomeTask(task.task_id,outcome,button));actions.append(button);});if(current){const undo=el("button","outcome-choice","Undo label");undo.type="button";undo.disabled=!writeEnabled;undo.addEventListener("click",()=>undoOutcomeTask(task.task_id,undo));actions.append(undo);}card.append(actions);host.append(card);});
      renderPrivateEvaluation(evaluation||{});
    }

    async function labelOutcomeTask(taskId,outcome,button){button.disabled=true;try{await authRequest("/api/outcome/label",{task_id:taskId,outcome});showToast(`${humanize(outcome)} outcome recorded with an audit trail.`);await load(true);}catch(error){button.disabled=false;showToast(`Could not record outcome: ${error.message}`,true);}}
    async function undoOutcomeTask(taskId,button){button.disabled=true;try{await authRequest("/api/outcome/undo",{task_id:taskId});showToast("Outcome label reversed; the prior record remains in the audit trail.");await load(true);}catch(error){button.disabled=false;showToast(`Could not undo outcome: ${error.message}`,true);}}

    function renderPrivateEvaluation(evaluation){
      const runs=Array.isArray(evaluation.runs)?evaluation.runs:[],active=evaluation.active||null,latest=evaluation.latest||null,button=document.getElementById("evaluation-start"),progress=document.getElementById("evaluation-progress"),ready=Boolean(evaluation.ready);
      button.disabled=Boolean(active)||!ready;button.textContent=active?"Evaluation running…":ready?"Run private evaluation":`Label ${Math.max(0,Number(evaluation.minimum_cases||8)-Number(evaluation.case_count||0))} more positive tasks`;
      progress.hidden=!(active||runs[0]?.status==="failed");if(!progress.hidden){const current=active||runs[0],value=Math.max(0,Math.min(100,Number(current.progress||0)));document.getElementById("evaluation-progress-title").textContent=active?humanize(current.phase||"running"):"Evaluation stopped";document.getElementById("evaluation-progress-message").textContent=String(current.error||current.message||"Comparing the paired retrieval conditions.");document.getElementById("evaluation-progress-value").textContent=`${Math.round(value)}%`;document.getElementById("evaluation-progress-fill").style.width=`${value}%`;}
      const summary=document.getElementById("evaluation-summary");summary.replaceChildren();const fields=latest?[
        ["Adaptive hit@6",`${(Number(latest.adaptive_hit_at_k||0)*100).toFixed(1)}%`],["Fixed hit@6",`${(Number(latest.fixed_hit_at_k||0)*100).toFixed(1)}%`],["Context p50 Δ",`${signed(latest.context_delta_p50||0,0)} tok`],["Latency p95 Δ",`${signed(latest.latency_delta_p95||0,1)} ms`]
      ]:[["Adaptive hit@6","—"],["Fixed hit@6","—"],["Context p50 Δ","—"],["Latency p95 Δ","—"]];fields.forEach(([label,value])=>{const card=el("div","private-evaluation-stat");card.append(el("span","",label),el("strong","",value));summary.append(card);});
      document.getElementById("evaluation-boundary").textContent=String(evaluation.claim_boundary||"This tests Cortex retrieval mechanics on private real history. It does not measure complete-agent accuracy or inference speed.");
      const runHost=document.getElementById("evaluation-runs");runHost.replaceChildren();if(!runs.length){runHost.append(emptyState("No private evaluation yet",ready?"The labeled set is ready for its first paired run.":"Positive task labels build the private case set without exposing their queries."));}runs.slice(0,6).forEach(run=>{const row=el("div","private-run"),copy=el("div","");copy.append(el("strong","",formatDate(run.started_at)),el("span","",`${humanize(run.status)} · ${formatNum(run.case_count)} private cases`));row.append(copy,el("span","",run.status==="completed"?`${(Number(run.adaptive_recall_at_k||0)*100).toFixed(1)}% adaptive recall`:humanize(run.phase||run.status)));runHost.append(row);});if(active)scheduleEvaluationPoll();
    }

    async function startPrivateEvaluation(){const button=document.getElementById("evaluation-start");button.disabled=true;try{const evaluation=await authRequest("/api/evaluation/start",{suite:"real_history"});app.data.evaluations=evaluation;renderOutcomeLab(app.data.outcome_lab||{},evaluation);showToast("Private paired evaluation started on this Cortex host.");scheduleEvaluationPoll();}catch(error){button.disabled=false;showToast(`Could not start evaluation: ${error.message}`,true);}}
    function scheduleEvaluationPoll(){if(evaluationPollTimer)return;evaluationPollTimer=setTimeout(()=>{evaluationPollTimer=null;pollEvaluationStatus();},800);}
    async function pollEvaluationStatus(){try{const response=await fetch("/api/evaluation/status",{cache:"no-store",credentials:"same-origin"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const evaluation=await response.json();app.data.evaluations=evaluation;renderOutcomeLab(app.data.outcome_lab||{},evaluation);if(evaluation.active)scheduleEvaluationPoll();else await load(true);}catch(error){showToast(`Private evaluation status unavailable: ${error.message}`,true);}}

    function signed(value,digits=1){const number=Number(value||0);return`${number>=0?"+":""}${number.toFixed(digits)}`;}

    function renderCapacityImpactChart() {
      if(!app.data)return;
      const range=Math.max(30,Number(document.getElementById("capacity-impact-range").value||90)),rows=(app.data.capacity_impact_by_day||[]).slice(-range).map(row=>{const positive=Number(row.helpful_outcomes||0),negative=Number(row.harmful_outcomes||0),labeled=positive+negative;return{...row,capacity:Number(row.stored_capacity||0),recalls:Number(row.recall_runs||0),tokens:Number(row.avg_context_tokens||0),resolved:Number(row.resolved_outcomes||0),labeled,accuracy:labeled?positive/labeled*100:null};}),svg=document.getElementById("capacity-impact-chart"),legend=document.getElementById("capacity-impact-legend"),summary=document.getElementById("capacity-impact-summary");svg.replaceChildren();legend.replaceChildren();summary.replaceChildren();
      const labeledRows=rows.filter(row=>row.accuracy!==null),labeledTotal=labeledRows.reduce((sum,row)=>sum+row.labeled,0),positiveTotal=labeledRows.reduce((sum,row)=>sum+Number(row.helpful_outcomes||0),0),accuracy=labeledTotal?positiveTotal/labeledTotal*100:null,latest=rows[rows.length-1]||{},first=rows[0]||{},capacityGrowth=Number(latest.capacity||0)-Number(first.capacity||0),avgTokens=weightedAverage(rows.filter(row=>row.recalls&&row.tokens),"tokens","recalls"),correlation=pearson(labeledRows.map(row=>[row.capacity,row.accuracy]));[[accuracy===null?"—":`${accuracy.toFixed(1)}%`,`${formatNum(labeledTotal)} labeled helpful/harmful outcomes`],[formatNum(latest.capacity||0),`${capacityGrowth>=0?"+":""}${formatNum(capacityGrowth)} stored memories in view`],[avgTokens?`${Math.round(avgTokens)} tok`:"—","average context per recall"],[correlation===null?"Need more days":`${correlation>=0?"+":""}${correlation.toFixed(2)} r`,`${labeledRows.length} labeled days · descriptive only`]].forEach(([value,label])=>{const card=el("div","capacity-impact-stat");card.append(el("strong","",value),el("span","",label));summary.append(card);});
      const series=[{label:"Memory helpfulness",color:cssVar("--green"),note:labeledTotal?`${accuracy.toFixed(1)}% across ${formatNum(labeledTotal)} labels`:"no labels"},{label:"Stored capacity",color:cssVar("--violet"),note:`${formatNum(latest.capacity||0)} memories`}];series.forEach(item=>{const entry=el("span","trend-legend-item"),swatch=el("i","");swatch.style.background=item.color;entry.append(swatch,el("span","",item.label),el("strong","",item.note));legend.append(entry);});
      if(!rows.length){svg.append(svgText(450,150,"Capacity history begins after Cortex records its first day","chart-label","middle"));return;}
      const W=900,H=330,L=50,R=56,T=28,B=42,plotW=W-L-R,plotH=H-T-B,maxCapacity=Math.max(1,...rows.map(row=>row.capacity)),x=index=>L+index*plotW/Math.max(1,rows.length-1),capacityY=value=>T+(maxCapacity-value)*plotH/maxCapacity,accuracyY=value=>T+(100-value)*plotH/100;
      const desc=svgEl("desc");desc.textContent="Stored memory count and daily observed memory helpfulness. Missing accuracy days have no helpful or harmful labels.";svg.append(desc);
      for(let index=0;index<5;index++){const y=T+index*plotH/4,rate=100-index*25,cap=Math.round(maxCapacity*(4-index)/4);svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,class:"chart-grid"}),svgText(L-9,y+4,`${rate}%`,"chart-label","end"),svgText(W-R+9,y+4,formatNum(cap),"chart-label","start"));}
      const capacityPoints=rows.map((row,index)=>({x:x(index),y:capacityY(row.capacity),...row})),capacityPath=capacityPoints.map((point,index)=>`${index?"L":"M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),capacityArea=`${capacityPath} L${capacityPoints[capacityPoints.length-1].x},${H-B} L${capacityPoints[0].x},${H-B} Z`;svg.append(svgEl("path",{d:capacityArea,fill:cssVar("--violet"),class:"capacity-area"}),svgEl("path",{d:capacityPath,stroke:cssVar("--violet"),class:"capacity-line"}));
      const accuracySegments=chartSegments(rows,row=>row.accuracy!==null,(row,index)=>({x:x(index),y:accuracyY(row.accuracy),...row}));accuracySegments.forEach(segment=>svg.append(svgEl("path",{d:segment.map((point,index)=>`${index?"L":"M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),stroke:cssVar("--green"),class:"accuracy-line"})));labeledRows.forEach(row=>{const index=rows.indexOf(row),dot=svgEl("circle",{cx:x(index),cy:accuracyY(row.accuracy),r:Math.min(7,3+Math.sqrt(row.labeled)),fill:cssVar("--surface"),stroke:cssVar("--green"),"stroke-width":"2"}),title=svgEl("title");title.textContent=`${row.day}: ${row.accuracy.toFixed(1)}% helpful across ${row.labeled} labeled outcomes · ${row.capacity} stored memories`;dot.append(title);svg.append(dot);});
      const labelEvery=Math.max(1,Math.ceil(rows.length/6));rows.forEach((row,index)=>{if(index%labelEvery===0||index===rows.length-1)svg.append(svgText(x(index),H-13,shortDay(row.day),"chart-label","middle"));});svg.append(svgText(L,T-10,"Outcome-backed usefulness","chart-label"),svgText(W-R,T-10,"Stored memories","chart-label","end"));
    }
    function chartSegments(rows,present,project){const segments=[];let current=[];rows.forEach((row,index)=>{if(present(row))current.push(project(row,index));else if(current.length){segments.push(current);current=[];}});if(current.length)segments.push(current);return segments;}
    function weightedAverage(rows,valueKey,weightKey){const weight=rows.reduce((sum,row)=>sum+Number(row[weightKey]||0),0);return weight?rows.reduce((sum,row)=>sum+Number(row[valueKey]||0)*Number(row[weightKey]||0),0)/weight:0;}
    function pearson(pairs){if(pairs.length<3)return null;const meanX=pairs.reduce((sum,pair)=>sum+pair[0],0)/pairs.length,meanY=pairs.reduce((sum,pair)=>sum+pair[1],0)/pairs.length,numerator=pairs.reduce((sum,pair)=>sum+(pair[0]-meanX)*(pair[1]-meanY),0),denX=Math.sqrt(pairs.reduce((sum,pair)=>sum+(pair[0]-meanX)**2,0)),denY=Math.sqrt(pairs.reduce((sum,pair)=>sum+(pair[1]-meanY)**2,0));return denX&&denY?numerator/(denX*denY):null;}

    function renderTrendChart() {
      if(!app.data)return;
      const svg=document.getElementById("trend-chart"),legend=document.getElementById("trend-legend"),range=Math.max(7,Number(document.getElementById("trends-range").value||30)),source=new Map((app.data.activity_trends||[]).map(row=>[row.day,row])),days=[],today=new Date();
      for(let offset=range-1;offset>=0;offset--){const day=new Date(Date.UTC(today.getUTCFullYear(),today.getUTCMonth(),today.getUTCDate()-offset)),key=day.toISOString().slice(0,10),row=source.get(key)||{};days.push({day:key,memories_made:Number(row.memories_made||0),connections_made:Number(row.connections_made||0),memories_pruned:Number(row.memories_pruned||0),tool_calls:Number(row.tool_calls||0)});}
      const series=[
        {key:"memories_made",label:"Memories made",color:cssVar("--violet")},
        {key:"connections_made",label:"Connections made",color:cssVar("--mint")},
        {key:"memories_pruned",label:"Cooled / pruned",color:cssVar("--amber")},
        {key:"tool_calls",label:"Tool calls",color:cssVar("--blue")},
      ];
      svg.replaceChildren();legend.replaceChildren();
      series.forEach(item=>{const total=days.reduce((sum,day)=>sum+day[item.key],0),entry=el("span","trend-legend-item"),swatch=el("i","");swatch.style.background=item.color;entry.append(swatch,el("span","",item.label),el("strong","",formatNum(total)));legend.append(entry);});
      const desc=svgEl("desc");desc.textContent=`Daily Cortex activity across the last ${range} days.`;svg.append(desc);
      const W=900,H=300,L=44,R=18,T=18,B=38,max=Math.max(1,...days.flatMap(day=>series.map(item=>day[item.key]))),plotH=H-T-B,plotW=W-L-R;
      for(let index=0;index<5;index++){const y=T+index*plotH/4,value=Math.round(max*(4-index)/4);svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,class:"chart-grid"}),svgText(L-8,y+4,value,"chart-label","end"));}
      series.forEach(item=>{const points=days.map((day,index)=>({x:L+index*plotW/Math.max(1,days.length-1),y:T+(max-day[item.key])*plotH/max,value:day[item.key],day:day.day})),path=points.map((point,index)=>`${index?"L":"M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");svg.append(svgEl("path",{d:path,fill:"none",stroke:item.color,"stroke-width":"2.4","stroke-linecap":"round","stroke-linejoin":"round"}));if(days.length<=30)points.forEach(point=>{const dot=svgEl("circle",{cx:point.x,cy:point.y,r:point.value?3.2:2,fill:point.value?item.color:cssVar("--surface"),stroke:item.color,"stroke-width":"1.4"}),title=svgEl("title");title.textContent=`${point.day}: ${point.value} ${item.label.toLowerCase()}`;dot.append(title);svg.append(dot);});});
      const labelEvery=Math.max(1,Math.ceil(days.length/6));days.forEach((day,index)=>{if(index%labelEvery===0||index===days.length-1)svg.append(svgText(L+index*plotW/Math.max(1,days.length-1),H-11,shortDay(day.day),"chart-label","middle"));});
    }

    function renderFunnel(stages,total) {const host=document.getElementById("recall-funnel");host.replaceChildren();stages.forEach(([label,value])=>{const row=el("div","funnel-row"),track=el("div","bar-track"),fill=el("div","bar-fill");fill.style.width=`${value/Math.max(1,total)*100}%`;track.append(fill);row.append(el("span","",label),track,el("strong","",`${formatNum(value)} · ${percent(value,total)}%`));host.append(row);});}
    function renderMemoryRanking(id,rows,meta) {const host=document.getElementById(id);host.replaceChildren();if(!rows.length){host.append(emptyState("Not enough evidence yet","This signal will appear as Cortex gathers more usage."));return;}rows.forEach(memory=>{const item=el("div","signal-item");item.tabIndex=0;const copy=el("div","");copy.append(el("div","signal-title",truncate(cleanMemory(memory.content),105)),el("div","small-meta",meta(memory)));item.append(copy,el("div","signal-score",Math.round(signalScore(memory)*100)));item.addEventListener("click",()=>selectMemory(memory.id));item.addEventListener("keydown",e=>{if(e.key==="Enter")selectMemory(memory.id);});host.append(item);});}
    function renderInsightSources(counts,total) {const host=document.getElementById("insight-source-list");host.replaceChildren();const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]),max=Math.max(1,...entries.map(entry=>entry[1]));entries.forEach(([name,count])=>{const row=el("div","source-item"),track=el("div","bar-track"),fill=el("div","bar-fill");fill.style.width=`${count/max*100}%`;track.append(fill);row.append(el("div","",humanize(name)),track,el("div","source-count",`${percent(count,total)}%`));host.append(row);});}
    function renderInsightTools(tools) {const host=document.getElementById("insight-tool-list");host.replaceChildren();const rows=[...tools].sort((a,b)=>toolAttempts(b)-toolAttempts(a)).slice(0,8),max=Math.max(1,...rows.map(tool=>toolAttempts(tool)));if(!rows.length){host.append(emptyState("No tool observations yet","This view will fill as Hermes completes tool-assisted tasks."));return;}rows.forEach(tool=>{const attempts=toolAttempts(tool),rate=percent(tool.success_count,attempts),row=el("div","source-item"),track=el("div","bar-track"),fill=el("div","bar-fill");fill.style.width=`${attempts/max*100}%`;track.append(fill);row.append(el("div","",tool.tool_name),track,el("div","source-count",`${rate}%`));host.append(row);});}

    function renderEvidenceHierarchy(hierarchy){
      const claims=Array.isArray(hierarchy.supported_claims)?hierarchy.supported_claims:[],levels=Array.isArray(hierarchy.levels)?hierarchy.levels:[],flow=document.getElementById("hierarchy-flow"),host=document.getElementById("supported-claim-list");flow.replaceChildren();host.replaceChildren();document.getElementById("hierarchy-summary-count").textContent=`${formatNum(claims.length)} visible supported claims`;
      const values=[formatNum(hierarchy.raw_evidence_count||0),formatNum(hierarchy.dependency_count||0),formatNum(hierarchy.summary_candidates||0)];levels.forEach((level,index)=>{const card=el("article","hierarchy-level");card.append(el("span","",`${String(index+1).padStart(2,"0")} · ${humanize(level.level||"level")}`),el("strong","",`${values[index]||"0"} · ${level.label||"Evidence level"}`),el("p","",index===0?"Original memories remain inspectable and are never replaced by a summary.":index===1?"A claim qualifies only when active dependency links point to its evidence.":"Repeated clean structure can become a navigation candidate; it is not treated as new truth."));flow.append(card);});
      if(!levels.length){[["01 · raw",`${formatNum(hierarchy.raw_evidence_count||0)} original memories`],["02 · claims",`${formatNum(hierarchy.dependency_count||0)} active evidence links`],["03 · bundles",`${formatNum(hierarchy.summary_candidates||0)} approval candidates`]].forEach(([label,value])=>{const card=el("article","hierarchy-level");card.append(el("span","",label),el("strong","",value));flow.append(card);});}
      if(!claims.length){host.append(emptyState("No explicit supported claims yet","Cortex has raw evidence, but this view only promotes claims with recorded dependency links."));}
      claims.slice(0,8).forEach(claim=>{const card=el("article","supported-claim"),button=el("button","",truncate(cleanMemory(claim.content),150));button.type="button";button.title=cleanMemory(claim.content);button.addEventListener("click",()=>selectMemory(claim.id));card.append(button,el("div","small-meta",`${formatNum(claim.evidence_count||0)} sources · ${claim.dirty?"support changed; review required":"support current"}`));const evidence=el("div","supported-claim-evidence");(claim.evidence||[]).slice(0,5).forEach(item=>{const source=el("button","",truncate(cleanMemory(item.content),76));source.type="button";source.title=cleanMemory(item.content);source.addEventListener("click",()=>selectMemory(item.id));evidence.append(source);});card.append(evidence);host.append(card);});
      document.getElementById("hierarchy-boundary").textContent=String(hierarchy.claim_boundary||"Candidates stay outside recall until approval and retain citations to their source memories.");
    }

    function renderLearning(){
      if(!app.data)return;const research=app.data.research||{};
      renderExperimentCenter(research.experiments||{});
      renderAgentEvaluation(research.agent_evaluation||{});
      renderFailureExplorer();
      renderSleepTrials(research.sleep_trials||{});
      renderSummaryCandidates(research.summary_candidates||[]);
      renderProspective(research.prospective||[]);
      renderReconsolidation(research.reconsolidation||{});
    }

    function renderExperimentCenter(experiments){
      const active=experiments.active||null,writeEnabled=app.data?.review_writes_enabled===true,status=document.getElementById("experiment-status"),toggle=document.getElementById("experiment-toggle"),host=document.getElementById("experiment-conditions"),boundary=document.getElementById("experiment-boundary"),taskHost=document.getElementById("experiment-task-list");status.textContent=active?"Collecting randomized tasks":"Stopped";toggle.textContent=active?"Stop experiment":"Start experiment";toggle.disabled=!writeEnabled;host.replaceChildren();taskHost.replaceChildren();
      const definitions={adaptive:"Attention gate, learned budget, and metacognitive policy.",fixed:"Always uses the configured top-k and token budget.",no_memory:"Withholds all Cortex memory and learned tool guidance."};
      (experiments.condition_stats||[]).forEach(item=>{const labels=Number(item.explicit_labels||0),accuracy=item.accuracy==null?"—":`${(Number(item.accuracy)*100).toFixed(1)}%`,card=el("article","condition-card");card.append(el("span","",humanize(item.condition)),el("strong","",`${accuracy} explicit accuracy`),el("p","",`${formatNum(item.assignments||0)} assigned · ${formatNum(labels)} explicitly labeled · ${item.ci_low==null?"interval waits for labels":`${(Number(item.ci_low)*100).toFixed(0)}–${(Number(item.ci_high)*100).toFixed(0)}% 95% interval`} · ${definitions[item.condition]||"Controlled condition"}`));host.append(card);});
      if(!host.children.length)RECALL_CONDITION_NAMES.forEach(([condition,copy])=>{const card=el("article","condition-card");card.append(el("span","",humanize(condition)),el("strong","","No assignments"),el("p","",copy));host.append(card);});
      boundary.classList.toggle("ready",Boolean(experiments.causal_ready));boundary.textContent=String(experiments.claim||"Collect explicit labels in every arm before interpreting condition differences causally.");
      const tasks=(experiments.recent_assignments||[]).filter(item=>item.task_id&&item.task_completed_at).slice(0,12);if(!tasks.length){taskHost.append(emptyState("No completed randomized tasks yet",active?"Use the agent normally. Each new task will appear here after its turn sync completes.":"Start the experiment to assign future agent tasks before recall."));return;}tasks.forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head");head.append(el("strong","",truncate(item.query_preview||`Private ${humanize(item.task_type)} task`,150)),el("span","",`${humanize(item.condition)} · ${timeAgo(new Date(item.task_started_at||item.created_at))}`));row.append(head,el("p","",`${formatNum(item.memory_count||0)} recalled memories · ${formatNum(item.context_tokens||0)} context tokens · ${item.outcome_source==="dashboard"?`explicitly ${humanize(item.outcome)}`:"awaiting an explicit outcome"}`));const actions=el("div","lab-row-actions");["helpful","validated","harmful","corrected"].forEach(outcome=>{const button=el("button",item.outcome===outcome&&item.outcome_source==="dashboard"?"primary-btn":"ghost-btn",humanize(outcome));button.type="button";button.disabled=!writeEnabled;button.addEventListener("click",()=>labelOutcomeTask(item.task_id,outcome,button));actions.append(button);});row.append(actions);taskHost.append(row);});
    }
    const RECALL_CONDITION_NAMES=[["adaptive","Adaptive policy chooses when and how much to recall."],["fixed","Fixed policy always uses the configured recall size."],["no_memory","Cortex evidence and guidance are withheld for the task."]];
    async function toggleRecallExperiment(){const active=Boolean(app.data?.research?.experiments?.active),button=document.getElementById("experiment-toggle");button.disabled=true;try{await authRequest("/api/experiment/control",{action:active?"stop":"start"});showToast(active?"Controlled recall experiment stopped.":"Controlled recall experiment started; future agent tasks will be assigned before recall.");await load(true);}catch(error){button.disabled=false;showToast(`Could not change experiment: ${error.message}`,true);}}

    function labMetric(host,label,value,detail){const card=el("div","lab-metric");card.append(el("span","",label),el("strong","",value),el("small","",detail));host.append(card);}
    function renderAgentEvaluation(evaluation){
      const summary=evaluation.summary||{},host=document.getElementById("agent-evaluation-summary"),labels=Number(summary.explicit_labels||0),accuracy=summary.accuracy==null?"—":`${(Number(summary.accuracy)*100).toFixed(1)}%`,toolCalls=Number(summary.tool_calls||0),toolRate=toolCalls?Number(summary.tool_successes||0)/toolCalls:null;host.replaceChildren();labMetric(host,"Explicit accuracy",accuracy,`${formatNum(labels)} operator-labeled tasks`);labMetric(host,"Observed tasks",formatNum(summary.tasks||0),`${formatNum(summary.completed||0)} completed turn-sync records`);labMetric(host,"Tool success",toolRate==null?"—":`${(toolRate*100).toFixed(1)}%`,`${formatNum(toolCalls)} observed tool results`);labMetric(host,"Corrections",formatNum(summary.corrections||0),"Explicit corrected outcomes");document.getElementById("agent-evaluation-samples").textContent=`${formatNum(labels)} labeled tasks`;renderAgentAccuracyChart(evaluation.by_day||[]);
    }
    function renderAgentAccuracyChart(rows){
      const svg=document.getElementById("agent-accuracy-chart"),days=[...rows].sort((a,b)=>String(a.day).localeCompare(String(b.day))).slice(-90);svg.replaceChildren();const desc=svgEl("desc");desc.textContent="Daily explicitly labeled agent accuracy, with task count as background bars.";svg.append(desc);if(days.length<8){svg.append(svgText(450,138,`Need ${Math.max(0,8-days.length)} more observed days for an honest trend line`,`chart-label`,`middle`),svgText(450,160,`${days.length} daily points recorded; exact totals remain in the cards above.`,`chart-label`,`middle`));return;}
      const W=900,H=300,L=48,R=18,T=22,B=38,plotW=W-L-R,plotH=H-T-B,maxTasks=Math.max(1,...days.map(row=>Number(row.tasks||0))),x=index=>L+index*plotW/Math.max(1,days.length-1),y=value=>T+(100-value)*plotH/100;for(let i=0;i<5;i++){const yy=T+i*plotH/4,rate=100-i*25;svg.append(svgEl("line",{x1:L,y1:yy,x2:W-R,y2:yy,class:"chart-grid"}),svgText(L-8,yy+4,`${rate}%`,"chart-label","end"));}
      const barWidth=Math.max(2,Math.min(12,plotW/days.length*.55));days.forEach((row,index)=>{const height=Number(row.tasks||0)/maxTasks*plotH*.32,bar=svgEl("rect",{x:x(index)-barWidth/2,y:H-B-height,width:barWidth,height,fill:cssVar("--violet"),opacity:.18}),title=svgEl("title");title.textContent=`${row.day}: ${row.tasks} observed tasks`;bar.append(title);svg.append(bar);});const segments=chartSegments(days,row=>row.accuracy!=null,(row,index)=>({x:x(index),y:y(Number(row.accuracy)*100),row}));segments.forEach(segment=>svg.append(svgEl("path",{d:segment.map((point,index)=>`${index?"L":"M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),fill:"none",stroke:cssVar("--green"),"stroke-width":"2.6","stroke-linecap":"round","stroke-linejoin":"round"})));days.forEach((row,index)=>{if(row.accuracy==null)return;const dot=svgEl("circle",{cx:x(index),cy:y(Number(row.accuracy)*100),r:Math.min(7,3+Math.sqrt(Number(row.explicit_labels||0))),fill:cssVar("--surface"),stroke:cssVar("--green"),"stroke-width":"2"}),title=svgEl("title");title.textContent=`${row.day}: ${(Number(row.accuracy)*100).toFixed(1)}% across ${row.explicit_labels} explicit labels · ${row.tasks} tasks`;dot.append(title);svg.append(dot);});const every=Math.max(1,Math.ceil(days.length/6));days.forEach((row,index)=>{if(index%every===0||index===days.length-1)svg.append(svgText(x(index),H-11,shortDay(row.day),"chart-label","middle"));});
    }

    function renderFailureExplorer(){
      if(!app.data)return;const failures=app.data.research?.failures||[],causeSelect=document.getElementById("failure-type"),taskSelect=document.getElementById("failure-task-type"),causes=[...new Set(failures.map(item=>String(item.cause||"unclassified_negative")))].sort(),taskTypes=[...new Set(failures.map(item=>String(item.task_type||"general")))].sort();syncSelectOptions(causeSelect,causes,app.failureCause,"All causes");syncSelectOptions(taskSelect,taskTypes,app.failureTaskType,"All task types");const filtered=failures.filter(item=>(app.failureCause==="all"||item.cause===app.failureCause)&&(app.failureTaskType==="all"||item.task_type===app.failureTaskType)),host=document.getElementById("failure-list");document.getElementById("failure-total").textContent=`${formatNum(filtered.length)} of ${formatNum(failures.length)} failures`;host.replaceChildren();if(!filtered.length){host.append(emptyState("No matching failures","Negative explicit outcomes and observed tool errors will appear here with a bounded diagnostic."));return;}filtered.slice(0,50).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head");head.append(el("strong","",truncate(item.query_preview||"Private agent task",150)),el("span","",`${humanize(item.task_type||"general")} · ${timeAgo(new Date(item.started_at))}`));row.append(head,el("p","",`${item.cause_label}. ${item.lever}`),el("p","small-meta",`${humanize(item.recall_condition||item.recall_mode||"unknown")} · ${formatNum(item.memory_count||0)} memories · ${formatNum(item.context_tokens||0)} tokens · ${formatNum(item.tool_successes||0)}/${formatNum(item.tool_calls||0)} tools succeeded`));const actions=el("div","lab-row-actions");["helpful","validated","harmful","corrected"].forEach(outcome=>{const button=el("button",item.outcome===outcome?"primary-btn":"ghost-btn",humanize(outcome));button.type="button";button.disabled=app.data.review_writes_enabled!==true;button.addEventListener("click",()=>labelOutcomeTask(item.task_id,outcome,button));actions.append(button);});row.append(actions);host.append(row);});
    }
    function syncSelectOptions(select,values,current,allLabel){const previous=current||"all";select.replaceChildren();const all=el("option","",allLabel);all.value="all";select.append(all);values.forEach(value=>{const option=el("option","",humanize(value));option.value=value;select.append(option);});select.value=[...select.options].some(option=>option.value===previous)?previous:"all";}

    function renderSleepTrials(data){const trials=data.trials||[],host=document.getElementById("sleep-trial-list"),button=document.getElementById("sleep-trial-start");button.disabled=app.data?.review_writes_enabled!==true;host.replaceChildren();if(!trials.length){host.append(emptyState("No matched apply trial yet","Run shadow Sleep until at least two equally supported association proposals exist, then start a randomized matched trial."));return;}trials.slice(0,12).forEach(trial=>{const row=el("article","lab-row"),head=el("div","lab-row-head");head.append(el("strong","",`${formatNum(trial.pair_count)} matched proposal pairs`),el("span","",`${humanize(trial.status)} · ${formatDate(trial.started_at)}`));row.append(head);const arms=el("div","condition-grid");(trial.arms||[]).forEach(arm=>{const card=el("article","condition-card"),rate=arm.positive_rate==null?"—":`${(Number(arm.positive_rate)*100).toFixed(1)}%`;card.append(el("span","",humanize(arm.assignment)),el("strong","",`${rate} positive`),el("p","",`${formatNum(arm.items)} proposal units · ${formatNum(arm.labeled_tasks)} explicitly labeled future tasks · ${formatNum(arm.negative_tasks)} negative`));arms.append(card);});row.append(arms,el("div",`causal-boundary ${trial.causal_ready?"ready":""}`,trial.causal_ready?"Both randomized arms have enough explicit outcomes for an initial causal estimate.":`This remains an active trial, not proof. Each arm needs ${formatNum(trial.minimum_labeled_per_arm||8)} explicit future-task outcomes.`));if(["collecting","completed"].includes(String(trial.status))){const actions=el("div","lab-row-actions"),undo=el("button","ghost-btn","Undo treatment links");undo.type="button";undo.disabled=app.data.review_writes_enabled!==true;undo.addEventListener("click",()=>undoSleepTrial(trial.trial_id,undo));actions.append(undo);row.append(actions);}host.append(row);});}
    async function startSleepTrial(){const button=document.getElementById("sleep-trial-start");button.disabled=true;try{const result=await authRequest("/api/sleep/trial/start",{pair_limit:8});showToast(`Started ${result.result?.pair_count||0}-pair Sleep trial; control links were withheld.`);await load(true);}catch(error){button.disabled=false;showToast(`Could not start Sleep trial: ${error.message}`,true);}}
    async function undoSleepTrial(trialId,button){button.disabled=true;try{await authRequest("/api/sleep/trial/undo",{trial_id:trialId});showToast("Treatment links restored to their pre-trial state.");await load(true);}catch(error){button.disabled=false;showToast(`Could not undo trial: ${error.message}`,true);}}

    function renderSummaryCandidates(candidates){const host=document.getElementById("summary-candidate-list"),writeEnabled=app.data?.review_writes_enabled===true;document.getElementById("summary-generate").disabled=!writeEnabled;host.replaceChildren();const proposed=candidates.filter(item=>item.status==="proposed");if(!proposed.length){host.append(emptyState("No summary candidates awaiting review","Generate extractive candidates from supported memory pairs. Nothing enters recall until approval."));}proposed.slice(0,20).forEach(candidate=>{const row=el("article","lab-row"),head=el("div","lab-row-head");head.append(el("strong","",candidate.title),el("span","",`${formatNum(candidate.sources?.length||0)} citations`));row.append(head,el("p","",candidate.summary_text));const citations=el("div","citation-list");(candidate.sources||[]).forEach(source=>{const citation=el("div","citation-item"),button=el("button","",`M:${shortId(source.memory_id)}`);button.type="button";button.addEventListener("click",()=>selectMemory(source.memory_id));citation.append(button,el("span","",`${source.claim_text} Source: ${truncate(cleanMemory(source.excerpt),180)}`));citations.append(citation);});row.append(citations);const actions=el("div","lab-row-actions");[["approve","Approve into recall","primary-btn"],["reject","Reject candidate","ghost-btn"]].forEach(([action,label,className])=>{const button=el("button",className,label);button.type="button";button.disabled=!writeEnabled;button.addEventListener("click",()=>reviewSummaryCandidate(candidate.candidate_id,action,button));actions.append(button);});row.append(actions);host.append(row);});}
    async function generateSummaryCandidates(){const button=document.getElementById("summary-generate");button.disabled=true;try{const result=await authRequest("/api/summary/generate",{limit:6});showToast(`${formatNum(result.result?.created||0)} cited summary candidates created; none are recallable yet.`);await load(true);}catch(error){button.disabled=false;showToast(`Could not generate candidates: ${error.message}`,true);}}
    async function reviewSummaryCandidate(candidateId,action,button){button.disabled=true;try{await authRequest("/api/summary/review",{candidate_id:candidateId,action});showToast(action==="approve"?"Summary approved with source dependencies preserved.":"Summary candidate rejected.");await load(true);}catch(error){button.disabled=false;showToast(`Could not review summary: ${error.message}`,true);}}

    function renderProspective(items){const host=document.getElementById("prospective-list"),open=items.filter(item=>["open","overdue"].includes(item.display_status)).length,writeEnabled=app.data?.review_writes_enabled===true;document.getElementById("prospective-open").textContent=`${formatNum(open)} open`;document.querySelectorAll("#prospective-form input,#prospective-form button").forEach(node=>node.disabled=!writeEnabled);host.replaceChildren();if(!items.length){host.append(emptyState("No tracked commitments","Add a future intention to give it an explicit lifecycle instead of leaving it as ordinary semantic memory."));return;}items.slice(0,40).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head"),status=String(item.display_status||item.status||"open"),statusEl=el("span",status==="overdue"?"status-overdue":"",humanize(status));head.append(el("strong","",truncate(cleanMemory(item.content),160)),statusEl);row.append(head,el("p","",item.due_at?`Due ${formatDate(item.due_at)} · updated ${timeAgo(new Date(item.updated_at))}`:"No due date · remains open until explicitly resolved"));const actions=el("div","lab-row-actions");if(item.status!=="completed")actions.append(prospectiveButton("Complete",item,"completed",writeEnabled));if(item.status!=="abandoned")actions.append(prospectiveButton("Abandon",item,"abandoned",writeEnabled));if(item.status!=="open")actions.append(prospectiveButton("Reopen",item,"open",writeEnabled));row.append(actions);host.append(row);});}
    function prospectiveButton(label,item,status,enabled){const button=el("button",status==="completed"?"primary-btn":"ghost-btn",label);button.type="button";button.disabled=!enabled;button.addEventListener("click",()=>updateProspective(item.memory_id,status,button));return button;}
    async function createProspectiveMemory(event){event.preventDefault();const form=event.currentTarget,button=form.querySelector("button"),due=form.due_at.value?new Date(form.due_at.value).toISOString():null;button.disabled=true;try{await authRequest("/api/prospective/create",{content:form.content.value,due_at:due});form.reset();showToast("Prospective memory added with an open lifecycle.");await load(true);}catch(error){button.disabled=false;showToast(`Could not add prospective memory: ${error.message}`,true);}}
    async function updateProspective(memoryId,status,button){button.disabled=true;try{await authRequest("/api/prospective/update",{memory_id:memoryId,status});showToast(`Prospective memory marked ${status}.`);await load(true);}catch(error){button.disabled=false;showToast(`Could not update prospective memory: ${error.message}`,true);}}

    function renderReconsolidation(data){const events=data.events||[],counts=data.counts||{},summary=document.getElementById("reconsolidation-summary"),host=document.getElementById("reconsolidation-list");document.getElementById("reconsolidation-total").textContent=`${formatNum(events.length)} corrections`;summary.replaceChildren();labMetric(summary,"After recall",formatNum(data.after_recall||0),"Corrections with prior retrieval or use");labMetric(summary,"Awaiting reuse",formatNum(counts.pending||0),"Corrected but not yet reexposed");labMetric(summary,"Stabilized",formatNum(counts.stabilized||0),"Explicit positive outcome after correction");labMetric(summary,"Failed",formatNum(counts.failed||0),"Negative outcome after correction");host.replaceChildren();if(!events.length){host.append(emptyState("No reconsolidation events yet","Correcting a memory preserves its prior version and starts replacement tracking."));return;}events.slice(0,30).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head"),reuse=item.time_to_reuse_hours==null?"not reused yet":`reused after ${formatDurationHours(item.time_to_reuse_hours)}`,stabilized=item.time_to_stabilize_hours==null?(item.outcome?`outcome ${humanize(item.outcome)}`:"awaiting explicit outcome"):`stabilized after ${formatDurationHours(item.time_to_stabilize_hours)}`;head.append(el("strong","",truncate(cleanMemory(item.corrected_content||item.content),160)),el("span","",`${humanize(item.status)} · ${timeAgo(new Date(item.corrected_at))}`));row.append(head,el("p","",`Previous version: ${truncate(cleanMemory(item.prior_content||"Unavailable"),190)}`),el("p","small-meta",`${item.trigger_access_at?"Corrected after recall":"No prior recall recorded"} · ${reuse} · ${stabilized}`));row.addEventListener("click",()=>selectMemory(item.memory_id));host.append(row);});}
    function formatDurationHours(hours){const value=Number(hours||0);return value<1?`${Math.max(0,Math.round(value*60))} min`:value<48?`${value.toFixed(1)} h`:`${(value/24).toFixed(1)} d`;}

    function renderCognition() {
      if(!app.data)return;
      const d=app.data,summary=d.recall_summary||{},runs=d.recall_runs||[],modes=d.recall_modes||[],budgets=d.recall_budgets||[],days=[...(d.recall_by_day||[])].reverse(),abstained=Number(summary.abstained||0),runCount=Number(summary.count||0),abstention=percent(abstained,Math.max(1,runCount));
      metricCards("cognition-metrics",[
        ["Recall decisions",d.stats.recall_runs||0,`${formatNum(abstained)} of the latest ${formatNum(runCount)} abstained`,"⌁"],
        ["Context avoided",`${abstention}%`,"Turns where Cortex injected zero memory tokens","○"],
        ["Median context",`${Math.round(Number(summary.p50_tokens||0))} tok`, `p95 ${Math.round(Number(summary.p95_tokens||0))} estimated tokens`,"≋"],
        ["Recall preparation",`${Number(summary.p50_prepare_ms||0).toFixed(1)} ms`,`p95 ${Number(summary.p95_prepare_ms||0).toFixed(1)} ms`,"↯"]
      ],true);
      document.getElementById("recall-total").textContent=`${formatNum(d.stats.recall_runs||0)} runs`;
      const modeDescriptions={none:"No durable context needed",lean:"Small evidence check",focused:"Personal, current, or decision context",procedural:"Memory plus tool evidence",deep:"Temporal or multi-memory reasoning",fixed:"Adaptive gate disabled"},modeHost=document.getElementById("recall-mode-list");modeHost.replaceChildren();
      if(!modes.length)modeHost.append(emptyState("No recall decisions yet","Use Hermes normally; attention-gate evidence will appear here."));
      modes.forEach(mode=>{const row=el("div","mode-row"),count=Number(mode.count||0),skipped=Number(mode.abstained||0);row.append(el("div","mode-name",humanize(mode.mode)),el("div","mode-copy",`${modeDescriptions[mode.mode]||"Adaptive recall plan"} · avg ${Math.round(Number(mode.avg_tokens||0))} tokens`),el("div","mode-stat",`${formatNum(count)} · ${percent(skipped,Math.max(1,count))}% empty`));modeHost.append(row);});
      renderBudgetLearning(budgets);
      renderScoringHealth(d.scoring_health||{});
      renderAttentionLearning(d.attention_learning||{});
      renderEvidenceHierarchy(d.evidence_hierarchy||{});
      renderSleep(d.sleep_runs||[],d.sleep_proposals||[],d.sleep_runtime||{});
      const strip=document.getElementById("recall-strip"),maxTokens=Math.max(1,...days.map(day=>Number(day.estimated_tokens||0)));strip.replaceChildren();
      if(!days.length)strip.append(emptyState("No token history yet","The chart begins after Cortex handles recall decisions."));
      days.forEach(day=>{const bar=el("div","recall-day");bar.style.height=`${Math.max(3,Number(day.estimated_tokens||0)/maxTokens*150)}px`;bar.title=`${day.day}: ${formatNum(day.estimated_tokens||0)} estimated tokens · ${formatNum(day.abstained||0)} abstained`;strip.append(bar);});
      document.getElementById("recall-window").textContent=days.length?`${days[0].day} — ${days[days.length-1].day}`:"No samples";

      const lifecycle=(d.lifecycle_events||[]).map(item=>({...item,event_kind:"state",event_time:item.created_at})),regrets=(d.pruning_regrets||[]).map(item=>({...item,event_kind:"regret",event_time:item.created_at})),consolidations=(d.consolidation_runs||[]).filter(item=>!Number(item.dry_run)).map(item=>({...item,event_kind:"consolidation",event_time:item.created_at})),events=[...lifecycle,...regrets,...consolidations].sort((a,b)=>Date.parse(b.event_time)-Date.parse(a.event_time)).slice(0,10),lifeHost=document.getElementById("lifecycle-feed");lifeHost.replaceChildren();
      if(!events.length)lifeHost.append(emptyState("No lifecycle changes yet","Cortex is healthy; shadow previews do not alter a memory."));
      events.forEach(event=>{const row=el("div","lifecycle-row"),label=event.event_kind==="state"?`${humanize(event.from_state)} → ${humanize(event.to_state)}`:event.event_kind==="regret"?(Number(event.restored)?"Restored":"Regret found"):"Consolidated",copy=event.event_kind==="consolidation"?`${formatNum(event.member_count)} memories folded across ${formatNum(event.cluster_count)} clusters`:truncate(cleanMemory(event.content||event.query||event.reason||"Lifecycle evidence recorded"),100),meta=event.event_kind==="state"?(event.reason||"State updated"):event.event_kind==="regret"?`Recall score ${Math.round(Number(event.score||0)*100)}%`:"Reversible run";row.append(el("div","lifecycle-state",label));const text=el("div","lifecycle-copy",copy);text.append(el("small","",meta));row.append(text,el("div","small-meta",timeAgo(new Date(event.event_time))));lifeHost.append(row);});

      const workflowHost=document.getElementById("workflow-feed"),workflows=d.tool_workflows||[];workflowHost.replaceChildren();
      if(!workflows.length)workflowHost.append(emptyState("No repeated workflows yet","A path appears only after the same tool sequence works across distinct tasks."));
      workflows.slice(0,10).forEach(workflow=>{const steps=parseJson(workflow.steps_json,[]),chain=steps.map(step=>`${step.tool}(${(step.argument_keys||[]).join(",")})`).join(" → ")||"Recorded workflow",attempts=Number(workflow.success_count||0)+Number(workflow.failure_count||0),rate=percent(workflow.success_count,Math.max(1,attempts)),row=el("div","lifecycle-row");row.append(el("div","lifecycle-state",humanize(workflow.task_type)));const copy=el("div","lifecycle-copy",chain);copy.append(el("small","",`${formatNum(workflow.distinct_tasks||0)} distinct tasks · ${rate}% observed success`));row.append(copy,el("div","small-meta",timeAgo(new Date(workflow.updated_at))));workflowHost.append(row);});
    }

    function renderScoringHealth(report) {
      const current=report.current||{},shadow=report.shadow||{},rate=value=>value===null||value===undefined?"—":`${(Number(value)*100).toFixed(1)}%`;
      metricCards("scoring-health-metrics",[
        ["Selection precision",rate(current.precision),`${formatNum(current.used_injections||0)} used of ${formatNum(current.resolved_injections||0)} resolved injections`,"◎"],
        ["Helpfulness",rate(current.helpfulness),`${formatNum(current.helpful_ratings||0)} Helpful or Essential ratings`,"+"],
        ["Token waste",rate(current.waste_rate),`${formatNum(current.wasted_tokens||0)} of ${formatNum(current.resolved_tokens||0)} resolved tokens`,"∿"],
        ["False positives",rate(current.false_positive_rate),`${formatNum(current.false_positive_ratings||0)} negative of ${formatNum(current.evaluated_ratings||0)} evaluated ratings`,"×"]
      ],true);
      const status=document.getElementById("scoring-policy-status"),observations=Number(shadow.resolved_observations||0),days=Number(shadow.observed_days||0);status.textContent=shadow.ready_for_selection_experiment?`Shadow ready · ${formatNum(observations)} resolved`:`Shadow · ${formatNum(observations)}/50 · ${days}/7 days`;
      document.getElementById("scoring-health-boundary").textContent=report.claim_boundary||"The candidate policy records scores only; it does not change live recall.";
      const signals=report.signals||[],signalHost=document.getElementById("scoring-signal-list");signalHost.replaceChildren();
      if(!signals.some(item=>Number(item.used_samples||0)+Number(item.unused_samples||0)>0))signalHost.append(emptyState("No resolved signal evidence yet","After the agent finishes memory-backed tasks, contribution shares will separate used from unused selections."));
      signals.filter(item=>Number(item.used_samples||0)+Number(item.unused_samples||0)>0).slice(0,8).forEach(item=>{const row=el("div","mode-row"),used=rate(item.used_contribution_share),unused=rate(item.unused_contribution_share),shift=Number(item.shadow_weight||0)-Number(item.weight||0),shiftCopy=Math.abs(shift)<.0001?"unchanged":`${shift>0?"+":""}${shift.toFixed(2)} in shadow`;row.append(el("div","mode-name",humanize(item.signal)),el("div","mode-copy",`Used share ${used} · unused share ${unused}`),el("div","mode-stat",shiftCopy));signalHost.append(row);});
      const rows=report.weekly||[],svg=document.getElementById("scoring-health-chart"),legend=document.getElementById("scoring-health-legend"),W=900,H=300,L=68,R=28,T=28,B=54,plotW=W-L-R,plotH=H-T-B;svg.replaceChildren();legend.replaceChildren();
      const desc=svgEl("desc");desc.textContent="Weekly resolved selection precision and estimated token waste rates.";svg.append(desc);
      [0,.25,.5,.75,1].forEach(value=>{const y=T+(1-value)*plotH;svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,stroke:cssVar("--border"),"stroke-width":1}));const label=svgEl("text",{x:L-12,y:y+4,"text-anchor":"end",fill:cssVar("--muted"),"font-size":12});label.textContent=`${Math.round(value*100)}%`;svg.append(label);});
      if(!rows.length){const label=svgEl("text",{x:W/2,y:H/2,"text-anchor":"middle",fill:cssVar("--muted"),"font-size":15});label.textContent="No resolved weekly scoring evidence yet";svg.append(label);}
      const x=index=>rows.length<=1?L+plotW/2:L+index/(rows.length-1)*plotW,y=value=>T+(1-Number(value))*plotH,series=[{field:"precision",label:"Selection precision",color:cssVar("--green")},{field:"waste_rate",label:"Token waste",color:cssVar("--amber")}];
      series.forEach(item=>{let prior=null;rows.forEach((row,index)=>{const value=row[item.field];if(value===null||value===undefined){prior=null;return;}const point={x:x(index),y:y(value)};if(prior)svg.append(svgEl("line",{x1:prior.x,y1:prior.y,x2:point.x,y2:point.y,stroke:item.color,"stroke-width":4,"stroke-linecap":"round"}));svg.append(svgEl("circle",{cx:point.x,cy:point.y,r:5,fill:item.color,stroke:cssVar("--surface"),"stroke-width":2}));prior=point;});const entry=el("span","trend-legend-item"),swatch=el("i","");swatch.style.background=item.color;entry.append(swatch,el("span","",item.label));legend.append(entry);});
      rows.forEach((row,index)=>{if(index!==0&&index!==rows.length-1&&rows.length>4)return;const label=svgEl("text",{x:x(index),y:H-18,"text-anchor":"middle",fill:cssVar("--muted"),"font-size":12});label.textContent=String(row.week||"").replace(/^\d{4}-/,"");svg.append(label);});
    }

    function renderAttentionLearning(report) {
      const summary=report.summary||{},gate=report.promotion_gate||{},weights=report.weights||[],rate=value=>value===null||value===undefined?"—":`${(Number(value)*100).toFixed(1)}%`,resolved=Number(summary.resolved_count||0),days=Number(summary.observed_days||0);
      metricCards("attention-learning-metrics",[
        ["Resolved topic turns",formatNum(resolved),`${formatNum(summary.observation_count||0)} total shadow decisions`,"⌁"],
        ["Attributed use",rate(summary.attributed_use_rate),`${formatNum(summary.used_count||0)} used · ${formatNum(summary.ignored_count||0)} ignored`,"◎"],
        ["Learned topic modes",formatNum(weights.filter(item=>Number(item.sample_count||0)>=4).length),"At least four resolved samples per topic and mode","◇"],
        ["Recent shadow changes",formatNum(summary.recent_shadow_changes||0),"Would change mode or token budget; never applied","↯"]
      ],true);
      const status=document.getElementById("attention-learning-status");status.textContent=gate.ready?`Eligible for trial · ${formatNum(resolved)} outcomes`:`Shadow · ${formatNum(resolved)}/50 · ${days}/7 days`;
      document.getElementById("attention-learning-boundary").textContent=report.claim_boundary||"Topic recommendations are observational and never change live recall.";
      const host=document.getElementById("attention-learning-topics");host.replaceChildren();
      if(!weights.length)host.append(emptyState("No resolved topic evidence yet","Enable attentional learning, then ordinary attributed turns will populate this shadow ledger."));
      weights.slice(0,12).forEach(item=>{const samples=Number(item.sample_count||0),delta=Number(item.effective_delta||0),row=el("div","mode-row"),signal=samples<4?"Learning":Math.abs(delta)<.001?"Hold":`${delta>0?"+":""}${Math.round(delta*100)}% budget`;row.append(el("div","mode-name",`${humanize(item.topic_key)} · ${humanize(item.mode)}`),el("div","mode-copy",`${formatNum(item.used_count||0)} used · ${formatNum(item.ignored_count||0)} ignored · ${formatNum(item.helpful_count||0)} helpful · ${formatNum(item.harmful_count||0)} harmful`),el("div","mode-stat",signal));host.append(row);});
      const gateHost=document.getElementById("attention-learning-gate");gateHost.replaceChildren();const checks=gate.checks||[];
      if(!checks.length)gateHost.append(emptyState("Promotion evidence unavailable","The shadow policy stays observational until every gate can be measured."));
      checks.forEach(check=>{const card=el("div",`gate-check ${check.passed?"pass":"fail"}`);card.append(el("span","",check.label||humanize(check.key)),el("strong","",`${check.passed?"Pass":"Hold"} · ${formatNum(check.value||0)} / ${formatNum(check.target||0)}`));gateHost.append(card);});
    }

    function renderBudgetLearning(budgets) {
      const host=document.getElementById("budget-learning-list"),total=budgets.reduce((sum,row)=>sum+Number(row.sample_count||0),0);host.replaceChildren();document.getElementById("budget-learning-total").textContent=`${formatNum(total)} resolved outcomes`;
      if(!budgets.length){host.append(emptyState("Waiting for resolved outcomes","Cortex holds its default budgets until enough recalled evidence has been used, ignored, confirmed, or corrected."));return;}
      budgets.slice(0,12).forEach(item=>{const samples=Number(item.sample_count||0),used=Number(item.used_count||0),helpful=Number(item.helpful_count||0),harmful=Number(item.harmful_count||0),ignored=Number(item.ignored_count||0),fill=Number(item.avg_estimated_tokens||0)/Math.max(1,Number(item.avg_requested_budget||0)),usedRate=used/Math.max(1,samples),ignoredRate=ignored/Math.max(1,samples);let signal="Hold";if(samples<8)signal="Learning";else if(harmful/Math.max(1,samples)>=.2)signal="Risk signal";else if(ignoredRate>=.65&&!helpful)signal="Reduce pressure";else if(helpful>=3&&usedRate>=.75&&fill>=.8)signal="Helpful saturation";const row=el("div","mode-row");row.append(el("div","mode-name",`${humanize(item.task_type)} · ${humanize(item.mode)}`),el("div","mode-copy",`${formatNum(samples)} outcomes · ${percent(used,samples)}% used · avg ${Math.round(Number(item.avg_estimated_tokens||0))}/${Math.round(Number(item.avg_requested_budget||0))} tokens`),el("div","mode-stat",signal));host.append(row);});
    }

    function renderSleep(runs,proposals,runtime={}) {
      const ordered=[...runs].sort(sleepNewest),latest=ordered[0];
      if(!app.sleepRunId||!ordered.some(run=>run.run_id===app.sleepRunId))app.sleepRunId=latest?.run_id||null;
      const selected=ordered.find(run=>run.run_id===app.sleepRunId)||latest;
      renderSleepSchedule(app.data?.sleep_schedule||{});
      renderSleepRunStats(ordered);
      renderSleepHistory(ordered);
      renderSleepProgress(runtime);
      const summaryHost=document.getElementById("sleep-cycle-summary"),metricHost=document.getElementById("sleep-metrics"),statusHost=document.getElementById("sleep-status");summaryHost.replaceChildren();metricHost.replaceChildren();
      if(!selected){
        statusHost.textContent="Waiting";
        document.getElementById("sleep-run-subtitle").textContent="No completed maintenance record exists yet.";
        summaryHost.append(emptyState("No Sleep cycles yet","Start a shadow run or wait for the scheduled window. The first report will appear here without changing live memory."));
        const waiting=el("div","sleep-metric sleep-metric-empty");waiting.append(el("span","","Offline evidence"),el("strong","","—"),el("small","","Replay, proposals, actual changes, and after-run observations will appear after the first cycle."));metricHost.append(waiting);
        document.getElementById("sleep-effects").replaceChildren();
        renderSleepHypotheses(null);
        renderSleepLedger(proposals,app.data?.sleep_edge_changes||[],app.data?.sleep_state_changes||[]);
        return;
      }
      const mode=String(selected.mode||"unknown").toLowerCase(),status=String(selected.status||"unknown").toLowerCase(),isFinished=["complete","completed","success","succeeded"].includes(status),isRunning=["running","started","in_progress"].includes(status),hasError=Boolean(selected.error),tone=hasError||status==="failed"||status==="error"?"failed":isFinished?"ready":isRunning?"running":"",statusLabel=hasError&&isFinished?"Completed with warning":isFinished?"Completed":isRunning?"Running":humanize(status),modeLabel=mode==="shadow"?"Shadow":mode==="apply"?"Apply":humanize(mode),duration=sleepDuration(selected);
      if(selected===latest&&!(["starting","running"].includes(String(runtime.status))))statusHost.textContent=`${modeLabel} · ${statusLabel}`;
      document.getElementById("sleep-run-subtitle").textContent=`Run ${shortId(selected.run_id)} · ${formatDate(selected.started_at)} · ${duration}`;
      const card=el("div","sleep-cycle-card"),head=el("div","sleep-cycle-head"),title=el("div","");title.append(el("div","sleep-kicker",`Selected ${modeLabel.toLowerCase()} cycle`),el("div","sleep-cycle-title",mode==="shadow"?"A complete preview; live memory stayed unchanged":"A gated apply cycle with reversible changes"),el("div","sleep-cycle-time",`Started ${formatDate(selected.started_at)} · finished in ${duration} · cutoff ${formatDate(selected.cutoff_at)}`));head.append(title,el("span",`sleep-status-chip ${tone}`,statusLabel));card.append(head);
      card.append(el("p","sleep-summary",mode==="shadow"?"Cortex replayed recorded evidence and wrote proposals only. Every item below says what would change; none of it entered normal recall.":"Cortex reviewed the same bounded evidence and committed only validated actions. Exact before-and-after rows appear below."));
      if(selected.error)card.append(el("div","sleep-error",`Cycle note: ${String(selected.error)}`));summaryHost.append(card);
      const report=selected.report||{},scanned=Number(selected.episodes_scanned||0),replayed=Number(selected.episodes_replayed||0),usage=Number(report.usage_tasks_replayed||0),evidence=Number(report.evidence_added||0),tokenBudget=Number(selected.reflection_token_budget||0),estimatedTokens=Number(selected.reflection_estimated_tokens||0);
      addSleepMetric(metricHost,"Episode replay",`${formatNum(replayed)} / ${formatNum(scanned)}`,scanned?`${percent(replayed,scanned)}% of eligible episodes revisited`:"No eligible episodes in this run");
      addSleepMetric(metricHost,"Outcome replay",formatNum(usage),`${formatNum(evidence)} new independent evidence ${evidence===1?"record":"records"}`);
      addSleepMetric(metricHost,"Runtime",duration,`${formatNum(selected.applied_changes||0)} reported applied changes`);
      addSleepMetric(metricHost,"Model reflection",tokenBudget?`${formatNum(estimatedTokens)} / ${formatNum(tokenBudget)} tok`:"Off",tokenBudget?humanize(selected.reflection_status||"unknown"):"Deterministic local analysis only");
      renderSleepEffects(selected,proposals);
      renderSleepHypotheses(selected);
      renderSleepLedger(proposals,app.data?.sleep_edge_changes||[],app.data?.sleep_state_changes||[]);
    }

    function sleepNewest(left,right){const a=Date.parse(left.completed_at||left.started_at||left.created_at||0),b=Date.parse(right.completed_at||right.started_at||right.created_at||0);return(Number.isFinite(b)?b:0)-(Number.isFinite(a)?a:0);}
    function sleepDuration(run){const start=Date.parse(run.started_at),end=Date.parse(run.completed_at);if(!Number.isFinite(start)||!Number.isFinite(end)||end<start)return String(run.status)==="running"?"in progress":"duration unavailable";const seconds=Math.max(0,Math.round((end-start)/1000));return seconds<60?`${seconds}s`:seconds<3600?`${Math.floor(seconds/60)}m ${seconds%60}s`:`${Math.floor(seconds/3600)}h ${Math.round(seconds%3600/60)}m`;}
    function addSleepMetric(host,label,value,copy){const card=el("div","sleep-metric");card.append(el("span","",label),el("strong","",value),el("small","",copy));host.append(card);}
    function renderSleepSchedule(schedule){const next=document.getElementById("sleep-next-run"),copy=document.getElementById("sleep-schedule-copy"),meta=document.getElementById("sleep-schedule-meta"),nextValue=String(schedule.next_run||"").trim(),parsed=Date.parse(nextValue);next.textContent=nextValue?(Number.isFinite(parsed)?formatDate(nextValue):nextValue):(schedule.schedule||"Nightly · 03:00–05:00 local window");copy.textContent=String(schedule.detail||"Starts nightly in a bounded idle window; manual shadow runs are available at any time.");meta.replaceChildren();[[schedule.active===true?"Timer active":schedule.active===false?"Timer inactive":"Timer state unknown",schedule.active===false?"bad":""],[schedule.enabled===true?"Enabled at login":schedule.enabled===false?"Not enabled":"Bundled default",""],[schedule.last_trigger?`Last trigger ${formatDate(schedule.last_trigger)}`:String(schedule.source||"Schedule source"),""]].forEach(([label,tone])=>meta.append(el("span",tone,label)));}
    function renderSleepRunStats(runs){const host=document.getElementById("sleep-run-stats"),completed=runs.filter(run=>String(run.status).toLowerCase()==="completed"),failed=runs.filter(run=>String(run.status).toLowerCase()==="failed"),durations=completed.map(run=>{const a=Date.parse(run.started_at),b=Date.parse(run.completed_at);return Number.isFinite(a)&&Number.isFinite(b)?Math.max(0,b-a):0;}).filter(Boolean),avg=durations.length?`${Math.round(durations.reduce((a,b)=>a+b,0)/durations.length/1000)}s`:"—",proposals=runs.reduce((total,run)=>total+sleepProposalCount(run),0),changes=runs.reduce((total,run)=>total+Number(run.applied_changes||0),0);host.replaceChildren();document.getElementById("sleep-run-total").textContent=`${formatNum(runs.length)} runs`;[[runs.length,"cycles recorded"],[runs.length?`${Math.round(completed.length/runs.length*100)}%`:"—",`${failed.length} failed`],[avg,"average runtime"],[proposals,`${changes} live changes`]].forEach(([value,label])=>{const card=el("div","sleep-run-stat");card.append(el("strong","",formatNum(value)),el("span","",label));host.append(card);});}
    function sleepProposalCount(run){return Number(run.association_proposals||0)+Number(run.interference_proposals||0)+Number(run.edge_decay_proposals||0)+Number(run.lifecycle_candidates||0)+Number(run.consolidation_candidates||0)+Number(run.dependency_candidates||0);}
    function renderSleepHistory(runs){const host=document.getElementById("sleep-run-history");host.replaceChildren();if(!runs.length){host.append(emptyState("No run history","Scheduled and manual cycles will appear here."));return;}runs.forEach(run=>{const button=el("button",`sleep-run-button ${run.run_id===app.sleepRunId?"active":""}`),top=el("div","sleep-run-button-top"),status=String(run.status||"unknown").toLowerCase();button.type="button";top.append(el("strong","",formatDate(run.started_at)),el("span",`sleep-run-button-status ${status==="failed"?"failed":""}`,`${humanize(run.mode)} · ${humanize(status)}`));button.append(top,el("small","",`${sleepDuration(run)} · ${formatNum(sleepProposalCount(run))} proposals · ${formatNum(run.applied_changes||0)} changed`));button.addEventListener("click",()=>{app.sleepRunId=run.run_id;renderSleep(app.data.sleep_runs||[],app.data.sleep_proposals||[],app.data.sleep_runtime||{});});host.append(button);});}
    function renderSleepEffects(run,proposals){const host=document.getElementById("sleep-effects"),runProposals=proposals.filter(item=>item.run_id===run.run_id),effects=app.data?.sleep_effects?.[run.run_id]||{},mode=String(run.mode||"shadow"),connectionIdeas=runProposals.filter(item=>["association","association_reinforcement"].includes(item.kind)).length,pruneIdeas=runProposals.filter(item=>["edge_downscale","lifecycle"].includes(item.kind)).length,reviewIdeas=runProposals.length-connectionIdeas-pruneIdeas,liveEdges=Number(effects.live_edge_changes||0),liveStates=Number(effects.live_state_changes||0),reversed=Number(effects.reversed_edge_changes||0)+Number(effects.reversed_state_changes||0),live=liveEdges+liveStates;host.replaceChildren();const direct=el("div",`sleep-effect-card ${mode==="shadow"?"proposed":"applied"}`),directCopy=mode==="shadow"?"This run observed and proposed. It did not change retrieval state, edge weights, or memory availability.":`${formatNum(liveEdges)} connection ${liveEdges===1?"change":"changes"} and ${formatNum(liveStates)} lifecycle ${liveStates===1?"change":"changes"} remain active; every one has stored before-and-after values.${reversed?` ${formatNum(reversed)} ${reversed===1?"change was":"changes were"} later reversed.`:""}`;direct.append(el("span","",mode==="shadow"?"Direct effect · proposal only":"Direct effect · committed"),el("strong","",mode==="shadow"?"0 live changes":`${formatNum(live)} live changes`),el("p","",directCopy));const signals=el("div","sleep-effect-signals"),signalRows=mode==="shadow"?[[connectionIdeas,"connect"],[pruneIdeas,"cool / archive"],[reviewIdeas,"review"]]:[[live,"still active"],[reversed,"reversed"],[run.applied_changes||0,"committed originally"]];signalRows.forEach(([count,label])=>signals.append(el("b","",`${formatNum(count)} ${label}`)));direct.append(signals);const observed=el("div","sleep-effect-card observed");observed.append(el("span","","Observed after this run"),el("strong","",`${formatNum(effects.recall_runs_after||0)} recalls`),el("p","","These are follow-up observations until the next Sleep cycle. They show what happened afterward, not proof that Sleep caused it."));const outcomes=el("div","sleep-effect-signals");[[effects.helpful_outcomes_after||0,"helpful"],[effects.harmful_outcomes_after||0,"harmful / corrected"],[effects.pruning_regrets_after||0,"pruning regret"]].forEach(([count,label])=>outcomes.append(el("b","",`${formatNum(count)} ${label}`)));observed.append(outcomes);host.append(direct,observed);}
    function sleepRecordCategory(kind){if(["association","association_reinforcement","edge_change","connection_added"].includes(kind))return"connections";if(["edge_downscale","lifecycle","state_change"].includes(kind))return"pruning";return"review";}
    function renderSleepHypotheses(run){const host=document.getElementById("sleep-hypotheses"),rows=run?(app.data?.sleep_hypotheses?.[run.run_id]||[]):[];host.replaceChildren();document.getElementById("sleep-hypothesis-count").textContent=`${formatNum(rows.length)} hypotheses`;if(!run){host.append(emptyState("No run selected","Choose a Sleep cycle to see the measurable predictions created by its proposals."));return;}if(!rows.length){host.append(emptyState("No hypotheses in this run","This cycle produced no proposal that could be followed through future attributed use."));return;}rows.slice(0,10).forEach(item=>{const card=el("article","sleep-hypothesis"),copy=el("div",""),meta=el("div","sleep-hypothesis-meta"),observations=Number(item.post_use_tasks||0),positive=Number(item.positive_tasks||0),negative=Number(item.negative_tasks||0);copy.append(el("strong","",item.hypothesis||"Future outcomes should test this proposal."),el("p","",`${item.success_metric||"Future explicit outcomes are the success metric."} ${truncate(String(item.rationale||""),120)}`));meta.append(el("span","",item.applied?"exposed · applied":"proposal only"),el("span","",`${formatNum(observations)} later uses`),el("span","",`${formatNum(positive)} positive · ${formatNum(negative)} negative`),el("span","",item.evidence_ready?"observational evidence ready":"needs 8 exposed observations"));card.append(copy,meta);host.append(card);});}

    function renderSleepLedger(proposals,edgeChanges,stateChanges){const host=document.getElementById("sleep-proposal-list"),runId=app.sleepRunId;host.replaceChildren();if(!runId){document.getElementById("sleep-proposal-total").textContent="0 records";host.append(emptyState("No run selected","Choose a cycle to inspect exactly what it touched."));return;}const proposalRecords=proposals.filter(item=>item.run_id===runId).map(item=>({...item,record_type:"proposal"})),edgeRecords=edgeChanges.filter(item=>item.run_id===runId).map(item=>({...item,record_type:"edge",kind:Number(item.prior_exists)?"edge_change":"connection_added",status:item.reversed_at?"reversed":"applied",rationale:Number(item.prior_exists)?`Changed ${humanize(item.relation)} weight from ${sleepWeight(item.prior_weight)} to ${sleepWeight(item.next_weight)}.`:`Created a ${humanize(item.relation)} connection at weight ${sleepWeight(item.next_weight)}.`,score:item.next_weight})),stateRecords=stateChanges.filter(item=>item.run_id===runId).map(item=>({...item,record_type:"state",kind:"state_change",status:item.reversed_at?"reversed":"applied",src_id:item.memory_id,src_kind:item.memory_kind,src_content:item.memory_content,rationale:`Changed memory availability from ${humanize(item.prior_state)} to ${humanize(item.next_state)}.`})),records=[...edgeRecords,...stateRecords,...proposalRecords].sort(sleepNewest).filter(item=>app.sleepFilter==="all"||sleepRecordCategory(item.kind)===app.sleepFilter);document.getElementById("sleep-proposal-total").textContent=`${formatNum(records.length)} records`;if(!records.length){host.append(emptyState("Nothing in this category","This run did not record a matching proposal or applied change."));return;}records.forEach(record=>renderSleepRecord(host,record));}
    function renderSleepRecord(host,record){const row=el("article","sleep-proposal-row"),kind=humanize(record.kind||"proposal"),copy=el("div","sleep-proposal-copy"),created=record.created_at&&Number.isFinite(Date.parse(record.created_at))?timeAgo(new Date(record.created_at)):"Time not recorded",status=String(record.status||"proposed").toLowerCase(),numericScore=Number(record.score),score=Number.isFinite(numericScore)?`${Math.round((numericScore<=1?numericScore*100:numericScore))}% signal`:"No score";copy.append(el("p","",String(record.rationale||"No rationale was recorded.")));const refs=el("div","sleep-memory-pair");if(record.src_id)refs.append(sleepMemoryRef(record.src_id,record.src_kind,record.src_content,"Memory A"));if(record.dst_id){refs.append(el("span","sleep-memory-arrow",sleepRecordCategory(record.kind)==="connections"?"↔":"→"),sleepMemoryRef(record.dst_id,record.dst_kind,record.dst_content,"Memory B"));}if(record.src_id)copy.append(refs);const details=record.record_type==="proposal"?sleepProposalDetails(record):record.record_type==="edge"?`${humanize(record.relation)} · evidence ${formatNum(record.prior_evidence_count||0)} → ${formatNum(record.next_evidence_count||0)}`:`${humanize(record.prior_state)} → ${humanize(record.next_state)}`;copy.append(el("div","sleep-change-detail",`${details} · ${created}`));const meta=el("div","sleep-proposal-meta");meta.append(el("span",`sleep-proposal-state ${status}`,humanize(status)),el("span","sleep-proposal-score",score));row.append(el("div","sleep-proposal-kind",kind),copy,meta);host.append(row);}
    function sleepMemoryRef(id,kind,content,label){const button=el("button","sleep-memory-ref");button.type="button";button.append(el("span","",`${label} · ${humanize(kind||"memory")} · ${shortId(id)}`),el("strong","",truncate(cleanMemory(content||"Memory text unavailable"),180)));button.addEventListener("click",()=>selectMemory(id));return button;}
    function sleepProposalDetails(record){const details=parseJson(record.details_json||"{}",{}),parts=[`${formatNum(record.evidence_count||0)} evidence`];if(details.distinct_witnesses)parts.push(`${formatNum(details.distinct_witnesses)} independent witnesses`);if(details.next_state)parts.push(`would move to ${humanize(details.next_state)}`);if(details.prior_weight!==undefined&&details.next_weight!==undefined)parts.push(`weight ${sleepWeight(details.prior_weight)} → ${sleepWeight(details.next_weight)}`);if(details.action)parts.push(`action ${humanize(details.action)}`);return parts.join(" · ");}
    function sleepWeight(value){const n=Number(value);return Number.isFinite(n)?n.toFixed(2):"none";}

    function renderSleepProgress(runtime) {
      const status=String(runtime.status||"idle").toLowerCase(),active=status==="starting"||status==="running",progress=Math.max(0,Math.min(100,Number(runtime.progress||0))),panel=document.getElementById("sleep-progress"),button=document.getElementById("sleep-start"),statusHost=document.getElementById("sleep-status"),steps=Array.isArray(runtime.steps)?runtime.steps:[];
      button.disabled=active;button.textContent=active?"Sleep running…":"Start shadow Sleep";
      panel.hidden=status==="idle";
      if(status==="idle")return;
      document.getElementById("sleep-progress-title").textContent=status==="completed"?"Sleep review complete":status==="failed"?"Sleep review stopped":humanize(runtime.phase||"working");
      document.getElementById("sleep-progress-message").textContent=String(runtime.message||"Cortex is reviewing stored evidence.");
      document.getElementById("sleep-progress-value").textContent=`${Math.round(progress)}%`;
      document.getElementById("sleep-progress-fill").style.width=`${progress}%`;
      const stepHost=document.getElementById("sleep-step-list");stepHost.replaceChildren();steps.slice(-5).forEach(step=>{const row=el("div","sleep-step");row.append(el("span","",humanize(step.phase||"step")),el("div","",String(step.message||"")));stepHost.append(row);});
      if(active){statusHost.textContent=`Shadow · ${Math.round(progress)}%`;scheduleSleepPoll();}
      else if(status==="completed")statusHost.textContent="Shadow · Completed";
      else if(status==="failed")statusHost.textContent="Shadow · Failed";
    }

    async function startSleepSession() {
      const button=document.getElementById("sleep-start");button.disabled=true;
      try{const runtime=await authRequest("/api/sleep/start",{mode:"shadow"});app.data.sleep_runtime=runtime;renderSleep(app.data.sleep_runs||[],app.data.sleep_proposals||[],runtime);showToast("Shadow Sleep started; watch the live review here.");scheduleSleepPoll();}
      catch(error){button.disabled=false;showToast(`Could not start Sleep: ${error.message}`,true);}
    }

    function scheduleSleepPoll(){if(sleepPollTimer)return;sleepPollTimer=setTimeout(()=>{sleepPollTimer=null;pollSleepStatus();},900);}

    async function pollSleepStatus(){try{const response=await fetch("/api/sleep/status",{cache:"no-store",credentials:"same-origin"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const runtime=await response.json();app.data.sleep_runtime=runtime;renderSleep(app.data.sleep_runs||[],app.data.sleep_proposals||[],runtime);if(["starting","running"].includes(String(runtime.status)))scheduleSleepPoll();else await load(true);}catch(error){showToast(`Sleep status unavailable: ${error.message}`,true);}}

    function renderMetacognition() {
      if(!app.data)return;
      const monitor=app.data.metacognition||{},summary=monitor.summary||{},labeled=Number(summary.labeled_count||0),negative=Number(summary.negative_count||0),ready=labeled>=8,ece=Number(summary.expected_calibration_error),brier=Number(summary.brier_score),mode=String(monitor.mode||"shadow").toLowerCase(),modeHost=document.getElementById("metacognition-mode");
      modeHost.replaceChildren(el("strong","",mode==="enforce"?"Enforced":"Shadow mode"),el("span","",mode==="enforce"?"Decisions affect recall":"Observing only"));
      metricCards("metacognition-metrics",[
        ["Calibration error",ready&&Number.isFinite(ece)?`${(ece*100).toFixed(1)}%`:"Learning",ready?"Expected calibration error · lower is better":`${Math.max(0,8-labeled)} more labeled outcomes needed`,"◎"],
        ["Harmful labels",labeled?`${percent(negative,labeled)}%`:"—",`${formatNum(negative)} harmful or corrected of ${formatNum(labeled)} labeled`,"!"],
        ["Brier score",ready&&Number.isFinite(brier)?brier.toFixed(3):"Learning",ready?"Probability error · 0 is perfect":"Shown after 8 explicit outcomes","◇"]
      ],true);
      renderCalibrationChart(monitor.calibration_bins||[],labeled);
      renderMetacognitiveDecisionMix(summary);
      renderMetacognitiveSources(monitor.by_source||[]);
      renderMetacognitionGate(monitor.enforcement_gate||{});
      renderMetacognitivePredictions();
    }

    function renderCalibrationChart(bins,labeled) {
      const svg=document.getElementById("calibration-chart"),note=document.getElementById("calibration-note"),sampleHost=document.getElementById("calibration-samples"),W=900,H=340,L=72,R=30,T=28,B=58,plotW=W-L-R,plotH=H-T-B;svg.replaceChildren();sampleHost.textContent=`${formatNum(labeled)} labels`;
      if(labeled<8){svg.append(svgText(W/2,145,"Calibration begins after 8 explicit outcomes","chart-title","middle"),svgText(W/2,174,`${Math.max(0,8-labeled)} more helpful, validated, harmful, or corrected labels needed`,"chart-label","middle"));note.textContent="Cortex is logging probabilities now, but it will not present a calibration score from an underpowered sample.";return;}
      for(let index=0;index<5;index++){const ratio=index/4,x=L+ratio*plotW,y=T+(1-ratio)*plotH,label=`${Math.round(ratio*100)}%`;svg.append(svgEl("line",{x1:L,y1:y,x2:W-R,y2:y,class:"chart-grid"}),svgText(L-10,y+4,label,"chart-label","end"),svgText(x,H-28,label,"chart-label","middle"));}
      svg.append(svgEl("line",{x1:L,y1:H-B,x2:W-R,y2:H-B,class:"chart-axis"}),svgEl("line",{x1:L,y1:T,x2:L,y2:H-B,class:"chart-axis"}),svgEl("path",{d:`M${L},${H-B} L${W-R},${T}`,class:"calibration-ideal"}),svgText(W-R-4,T+16,"Perfect calibration","chart-label","end"),svgText(L+plotW/2,H-7,"Predicted reliability","chart-label","middle"));
      const yLabel=svgText(16,T+plotH/2,"Observed helpfulness","chart-label","middle");yLabel.setAttribute("transform",`rotate(-90 16 ${T+plotH/2})`);svg.append(yLabel);
      const points=bins.map(row=>{const samples=Number(row.sample_count||0),predicted=Number(row.avg_probability||0),actual=Number(row.positive_count||0)/Math.max(1,samples);return{samples,predicted,actual,x:L+predicted*plotW,y:T+(1-actual)*plotH};}).filter(point=>point.samples>0).sort((a,b)=>a.predicted-b.predicted);
      if(points.length>1)svg.append(svgEl("path",{d:points.map((point,index)=>`${index?"L":"M"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "),class:"calibration-observed"}));
      points.forEach(point=>{const dot=svgEl("circle",{cx:point.x,cy:point.y,r:6,class:"calibration-point"}),title=svgEl("title");title.textContent=`${Math.round(point.predicted*100)}% predicted · ${Math.round(point.actual*100)}% helpful · ${point.samples} labels`;dot.append(title);svg.append(dot,svgText(point.x,Math.max(T+12,point.y-12),`n=${point.samples}`,"chart-label","middle"));});
      note.textContent="The dashed diagonal is perfect calibration. Points above it were too cautious; points below it were overconfident. Only explicit helpful, validated, harmful, and corrected outcomes are included.";
    }

    function renderMetacognitiveDecisionMix(summary) {
      const host=document.getElementById("metacognition-decision-mix"),rows=[["use",Number(summary.use_count||0)],["verify",Number(summary.verify_count||0)],["abstain",Number(summary.abstain_count||0)]],total=rows.reduce((count,row)=>count+row[1],0);host.replaceChildren();
      if(!total){host.append(emptyState("Waiting for recall decisions","The next memory-bearing the agent turn will create the first shadow judgment."));return;}
      rows.forEach(([name,count])=>{const row=el("div","decision-mix-row"),track=el("div","bar-track"),fill=el("div",`bar-fill decision-fill ${name}`);fill.style.width=`${count/Math.max(1,total)*100}%`;track.append(fill);row.append(el("span","",name),track,el("strong","",`${formatNum(count)} · ${percent(count,total)}%`));host.append(row);});
    }

    function renderMetacognitiveSources(rows) {
      const host=document.getElementById("metacognition-sources");host.replaceChildren();
      if(!rows.length){host.append(emptyState("No source judgments yet","Source-level calibration appears after Cortex evaluates recalled memories."));return;}
      rows.forEach(item=>{const labels=Number(item.labeled_count||0),positive=Number(item.positive_count||0),row=el("div","source-monitor-row"),copy=el("div","");copy.append(el("strong","",humanize(item.source_category)),el("span","",`${formatNum(item.prediction_count)} predictions`));row.append(copy,el("strong","",`${Math.round(Number(item.avg_probability||0)*100)}% predicted`),el("strong","",labels?`${percent(positive,labels)}% observed`:"Learning"),el("span","",`${formatNum(labels)} explicit labels`));host.append(row);});
    }

    function renderMetacognitionGate(gate){const panel=document.getElementById("metacognition-gate"),status=document.getElementById("metacognition-gate-status"),host=document.getElementById("metacognition-gate-checks"),checks=Array.isArray(gate.checks)?gate.checks:[],ready=Boolean(gate.ready);panel.classList.toggle("ready",ready);status.textContent=ready?"Eligible for controlled trial":"Shadow locked";host.replaceChildren();if(!checks.length){host.append(emptyState("Gate evidence unavailable","Metacognition will stay in shadow mode until the dashboard can verify every promotion requirement."));return;}checks.forEach(check=>{const value=check.value===null||check.value===undefined?"Learning":check.key==="labels"?formatNum(check.value):`${(Number(check.value)*100).toFixed(1)}%`,target=check.key==="labels"?formatNum(check.target):`${(Number(check.target)*100).toFixed(0)}%`,card=el("div",`gate-check ${check.passed?"pass":"fail"}`);card.append(el("span","",check.label||humanize(check.key)),el("strong","",`${check.passed?"Pass":"Hold"} · ${value} ${check.direction==="at_most"?"≤":"/"} ${target}`));host.append(card);});document.getElementById("metacognition-gate-boundary").textContent=String(gate.claim_boundary||"Passing this evidence gate permits only a controlled enforcement trial.");}

    function renderMetacognitivePredictions() {
      if(!app.data)return;
      const monitor=app.data.metacognition||{},rows=(monitor.recent_predictions||[]).filter(item=>app.metacognitionFilter==="all"||item.decision===app.metacognitionFilter).slice(0,60),host=document.getElementById("metacognition-predictions");host.replaceChildren();
      if(!rows.length){host.append(emptyState("No matching trust decisions","Use the agent normally, or choose another decision filter."));return;}
      rows.forEach(item=>{const row=el("article","prediction-row"),memory=el("button","prediction-memory"),features=item.features||{},signals=el("div","prediction-signals");memory.type="button";memory.append(el("strong","",truncate(cleanMemory(item.memory_content),150)),el("p","",`${item.reason} · ${humanize(item.source_category)} · ${humanize(item.task_type)} · ${timeAgo(new Date(item.created_at))}`));[[`match ${Math.round(Number(features.direct_relevance||0)*100)}%`],[`source ${Math.round(Number(features.source_prior||0)*100)}%`],[`stale ${Math.round(Number(features.stale_risk||0)*100)}%`],[item.calibration_scope==="prior"?"prior only":`${humanize(item.calibration_scope)} · n=${formatNum(item.calibration_samples)}`]].forEach(([label])=>signals.append(el("span","",label)));memory.append(signals);memory.addEventListener("click",()=>selectMemory(item.memory_id));const score=el("div","prediction-score");score.append(el("strong","",`${Math.round(Number(item.calibrated_probability||0)*100)}%`),el("span","","reliability"));const status=el("div","prediction-status");status.append(el("span",`decision-chip ${item.decision}`,item.decision),el("span",`decision-chip ${item.outcome}`,item.outcome));row.append(memory,score,status);host.append(row);});
    }

    function renderTools() {
      if(!app.data)return;
      const tools=app.data.tools||[], attempts=tools.reduce((n,t)=>n+Number(t.success_count||0)+Number(t.failure_count||0),0),success=tools.reduce((n,t)=>n+Number(t.success_count||0),0),failures=tools.reduce((n,t)=>n+Number(t.failure_count||0),0),rate=attempts?Math.round(success/attempts*100):0;
      metricCards("tool-metrics",[["Observed calls",app.data.stats.tool_executions,"Execution outcomes","◇"],["Learned strategies",tools.length,"Task and tool pairs","⌁"],["Success rate",`${rate}%`,`${formatNum(success)} successful`,"✓"],["Failures",failures,"Retained as warnings","!"]],true);
      const summary=document.getElementById("tool-summary");summary.replaceChildren();
      renderToolEvaluation(app.data.tool_evaluation||{});
      [...tools].sort((a,b)=>toolAttempts(b)-toolAttempts(a)).slice(0,8).forEach(tool=>{const total=toolAttempts(tool),pct=total?Math.round(Number(tool.success_count||0)/total*100):0,card=el("div","panel tool-card");card.append(el("div","tool-task",humanize(tool.task_type)),el("div","tool-name",tool.tool_name));const rel=el("div","reliability"),track=el("div","bar-track"),fill=el("div","bar-fill");fill.style.width=`${pct}%`;track.append(fill);rel.append(el("strong","",`${pct}%`),track);card.append(rel,el("div","small-meta",`${total} observations · ${Number(tool.failure_count||0)} failures`));summary.append(card);});
      const host=document.getElementById("tool-table");host.replaceChildren();
      if(!tools.length){host.append(emptyState("No reinforced tool strategies yet","Cortex needs repeated tool outcomes across distinct tasks before it generalizes."));return;}
      const table=document.createElement("table"),thead=document.createElement("thead"),hr=document.createElement("tr");["Task","Tool","Reliability","Success","Failure","Argument shape","Last signal"].forEach(h=>hr.append(el("th","",h)));thead.append(hr);table.append(thead);const tbody=document.createElement("tbody");
      const toolHeaders=["Task","Tool","Reliability","Success","Failure","Argument shape","Last signal"];
      tools.forEach(tool=>{const total=toolAttempts(tool),pct=total?Math.round(Number(tool.success_count||0)/total*100):0,row=document.createElement("tr"),keys=parseJson(tool.argument_keys,[]).join(", ")||"—";[humanize(tool.task_type),tool.tool_name,`${pct}%`,tool.success_count,tool.failure_count,keys,timeAgo(new Date(tool.updated_at))].forEach((value,index)=>{const cell=el("td","",value);cell.dataset.label=toolHeaders[index];row.append(cell);});tbody.append(row);});table.append(tbody);host.append(table);
    }

    function renderToolEvaluation(evaluation){const host=document.getElementById("tool-evaluation-summary"),ledger=document.getElementById("guidance-ledger"),exposures=Number(evaluation.exposures||0),resolved=Number(evaluation.resolved||0),followed=Number(evaluation.followed||0);host.replaceChildren();[
      [formatNum(exposures),`${formatNum(resolved)} guidance exposures have a recorded follow-through`],
      [evaluation.follow_rate===null||evaluation.follow_rate===undefined?"—":`${(Number(evaluation.follow_rate)*100).toFixed(1)}%`,"Resolved exposures followed with a matching tool or workflow"],
      [evaluation.followed_success_rate===null||evaluation.followed_success_rate===undefined?"—":`${(Number(evaluation.followed_success_rate)*100).toFixed(1)}%`,`${formatNum(evaluation.followed_success||0)} successes after ${formatNum(followed)} followed suggestions`],
      [formatNum(Number(evaluation.positive_task_outcomes||0)+Number(evaluation.negative_task_outcomes||0)),`${formatNum(evaluation.positive_task_outcomes||0)} positive · ${formatNum(evaluation.negative_task_outcomes||0)} negative explicit task outcomes`]
    ].forEach(([value,label])=>{const card=el("div","tool-evaluation-stat");card.append(el("strong","",value),el("span","",label));host.append(card);});ledger.replaceChildren();const rows=Array.isArray(evaluation.recent)?evaluation.recent:[];if(!rows.length){ledger.append(emptyState("Instrumentation starts with the next guided task","Historical tool calls remain visible above, but Cortex will not pretend old calls were exposed to guidance it did not log."));}rows.slice(0,10).forEach(item=>{const row=el("div","guidance-row"),identity=el("div","");identity.append(el("strong","",`${humanize(item.task_type)} · ${humanize(item.guidance_type)}`),el("span","",item.recommended_tool||item.guidance_key||"Recorded guidance"));row.append(identity,el("span","",item.resolved_at?(Number(item.followed)?"followed":"not followed"):"pending"),el("span","",item.success===null||item.success===undefined?"outcome pending":Number(item.success)?"execution succeeded":"execution failed"),el("span","",item.task_outcome?humanize(item.task_outcome):timeAgo(new Date(item.created_at))));ledger.append(row);});document.getElementById("tool-evaluation-boundary").textContent=String(evaluation.claim_boundary||"Follow-through is observational and does not prove guidance caused the result.");document.getElementById("tool-evaluation-status").textContent=evaluation.ready_for_comparison?"Comparison-ready · observational":"Observational · learning";}

    function brainMechanicsButton(label,payload,{primary=false,confirmation=""}={}){
      const button=el("button",primary?"primary-btn":"ghost-btn",label);button.type="button";
      button.addEventListener("click",async event=>{event.stopPropagation();if(confirmation&&!window.confirm(confirmation))return;button.disabled=true;try{await authRequest("/api/brain-mechanics/action",payload);showToast(`${label} saved`);await load(true);}catch(error){showToast(error.message,true);}finally{button.disabled=false;}});
      return button;
    }

    function brainMechanicsActions(items){
      const host=el("div","review-actions");items.filter(Boolean).forEach(item=>host.append(brainMechanicsButton(item.label,item.payload,item.options||{})));return host;
    }

    function renderAutoJudge() {
      if (!app.data) return;
      const d = app.data, aj = d.auto_judge || {};
      // Summary metrics row
      const metrics = document.getElementById("aj-summary-metrics");
      metrics.replaceChildren();
      const sum = aj.summary || {};
      [
        [`${sum.total || 0}`, "decisions made", "—"],
        [`${sum.remembered || 0}`, "remembered", "var(--mint)"],
        [`${sum.rejected || 0}`, "rejected", "var(--red)"],
        [`${sum.deferred || 0}`, "deferred (pending)", "var(--amber)"],
      ].forEach(([val, label, color]) => {
        const card = el("article", "metric-card");
        if (color !== "—") card.style.setProperty("--accent", color);
        card.append(el("strong", "metric-value", val), el("span", "metric-label", label));
        metrics.append(card);
      });

      const consolidation = d.semantic_consolidation || {};
      const consolidationCounts = consolidation.counts || {};
      const consolidationFeedback = consolidation.feedback || {};
      const consolidationDecisions = consolidation.decisions || [];
      const consolidationMetrics = document.getElementById("aj-consolidation-metrics");
      const consolidationList = document.getElementById("aj-consolidation-list");
      consolidationMetrics.replaceChildren();
      consolidationList.replaceChildren();
      labMetric(consolidationMetrics, "Proposed", formatNum(consolidationCounts.proposed || 0), "Model judgments awaiting an explicit gate");
      labMetric(consolidationMetrics, "Applied", formatNum(consolidationCounts.applied || 0), "New canonical memories with preserved source dependencies");
      labMetric(consolidationMetrics, "Linked", formatNum(consolidationCounts.linked || 0), "Pairs retained separately with an evidence-backed relation");
      labMetric(consolidationMetrics, "Reversed", formatNum(consolidationCounts.reversed || 0), "Applied merges whose source memories were restored");
      labMetric(consolidationMetrics, "Reviewed correctness", consolidationFeedback.correctness == null ? "Learning" : `${(Number(consolidationFeedback.correctness) * 100).toFixed(1)}%`, `${formatNum(consolidationFeedback.reviewed || 0)} human labels · target 80%`);
      document.getElementById("aj-consolidation-count").textContent = `${formatNum(consolidationCounts.proposed || 0)} proposals`;
      document.getElementById("aj-consolidation-boundary").textContent = String(consolidation.claim_boundary || "A model proposal is not an applied merge.");
      if (!consolidationDecisions.length) {
        consolidationList.append(emptyState("No semantic consolidation judgments yet", "Run Auto-Judge with --consolidate to record a bounded shadow batch."));
      } else {
        consolidationDecisions.slice(0, 12).forEach((item) => {
          const row = el("article", "lab-row");
          const head = el("div", "lab-row-head");
          head.append(
            el("strong", "", `${humanize(item.action)} · ${Math.round(Number(item.confidence || 0) * 100)}%`),
            el("span", "", `${humanize(item.status)} · ${timeAgo(new Date(item.created_at))}`)
          );
          row.append(
            head,
            el("p", "", `${truncate(cleanMemory(item.left_content || "Unavailable"), 130)} ↔ ${truncate(cleanMemory(item.right_content || "Unavailable"), 130)}`),
            el("p", "small-meta", `${item.reason || "No judgment reason recorded"}${item.feedback_label ? ` · reviewed ${humanize(item.feedback_label)}` : ""}`)
          );
          if(d.review_writes_enabled)row.append(brainMechanicsActions([
            item.status==="proposed"?{label:"Apply",payload:{mechanic:"consolidation",action:"apply",decision_id:item.decision_id},options:{primary:true,confirmation:"Apply this reversible consolidation?"}}:null,
            item.status==="applied"?{label:"Undo",payload:{mechanic:"consolidation",action:"undo",decision_id:item.decision_id}}:null,
            !item.feedback_label?{label:"Correct",payload:{mechanic:"consolidation",action:"feedback",decision_id:item.decision_id,label:"correct"}}:null,
            !item.feedback_label?{label:"Wrong",payload:{mechanic:"consolidation",action:"feedback",decision_id:item.decision_id,label:"wrong"},options:{confirmation:"Mark this judgment wrong? Applied merges will be undone."}}:null,
          ]));
          row.addEventListener("click", () => selectMemory(item.result_memory_id || item.left_id));
          consolidationList.append(row);
        });
      }

      const pruning=d.adaptive_pruning||{},pruningCounts=pruning.counts||{},pruningRegret=pruning.regret||{},pruningRows=pruning.decisions||[],pruningMetrics=document.getElementById("aj-pruning-metrics"),pruningList=document.getElementById("aj-pruning-list");
      pruningMetrics.replaceChildren();pruningList.replaceChildren();
      labMetric(pruningMetrics,"Proposed",formatNum(pruningCounts.proposed||0),"Awaiting operator review");
      labMetric(pruningMetrics,"Applied",formatNum(pruningCounts.applied||0),"Reversible lifecycle or strand changes");
      labMetric(pruningMetrics,"Reversed",formatNum(pruningCounts.reversed||0),"Restored manually or through regret");
      labMetric(pruningMetrics,"Regret",pruningRegret.rate==null?"Learning":`${(Number(pruningRegret.rate)*100).toFixed(1)}%`,`Target ≤ ${(Number(pruningRegret.target||.05)*100).toFixed(0)}%`);
      document.getElementById("aj-pruning-count").textContent=`${formatNum(pruningCounts.proposed||0)} proposals`;document.getElementById("aj-pruning-boundary").textContent=String(pruning.claim_boundary||"No pruning action hard-deletes memory content.");
      if(!pruningRows.length)pruningList.append(emptyState("No adaptive-pruning judgments yet","Enable the daily shadow pass or run Auto-Judge with --prune."));
      pruningRows.slice(0,12).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head");head.append(el("strong","",`${humanize(item.action)} · ${Math.round(Number(item.confidence||0)*100)}%`),el("span","",`${humanize(item.status)} · relevance ${Math.round(Number(item.relevance_score||0)*100)}%`));row.append(head,el("p","",truncate(cleanMemory(item.content||"Unavailable"),180)),el("p","small-meta",item.reason||"No reason recorded"));if(d.review_writes_enabled)row.append(brainMechanicsActions([item.status==="proposed"?{label:"Apply",payload:{mechanic:"pruning",action:"apply",decision_id:item.decision_id},options:{primary:true,confirmation:`Apply reversible ${humanize(item.action)}?`}}:null,item.status==="applied"?{label:"Undo",payload:{mechanic:"pruning",action:"undo",decision_id:item.decision_id}}:null]));row.addEventListener("click",()=>selectMemory(item.memory_id));pruningList.append(row);});

      const weights=d.scoring_weights||{},weightRows=weights.proposals||[],activeWeights=weights.active_profiles||[],weightMetrics=document.getElementById("aj-weights-metrics"),weightList=document.getElementById("aj-weights-list");
      weightMetrics.replaceChildren();weightList.replaceChildren();
      labMetric(weightMetrics,"Proposed",formatNum(weightRows.filter(item=>item.status==="proposed").length),"No retrieval effect");
      labMetric(weightMetrics,"Active",formatNum(activeWeights.length),"Explicitly approved task profiles");
      labMetric(weightMetrics,"Rolled back",formatNum(weightRows.filter(item=>["rolled_back","auto_reverted"].includes(item.status)).length),"History retained");
      labMetric(weightMetrics,"Bounds",`${Number(weights.bounds?.floor||.02).toFixed(2)}–${Number(weights.bounds?.ceiling||.30).toFixed(2)}`,"Every signal");
      document.getElementById("aj-weights-count").textContent=`${formatNum(weightRows.filter(item=>item.status==="proposed").length)} proposals`;document.getElementById("aj-weights-boundary").textContent=String(weights.claim_boundary||"A staged profile has no retrieval effect.");
      if(!weightRows.length)weightList.append(emptyState("No scoring-weight proposals yet","The weekly audit needs at least eight resolved selections for a task type."));
      weightRows.slice(0,12).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head"),baseline=item.baseline_weights||{},proposed=item.proposed_weights||{},changes=Object.keys(proposed).map(key=>[key,Number(proposed[key])-Number(baseline[key]||0)]).filter(([,delta])=>Math.abs(delta)>.0001).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,4).map(([key,delta])=>`${humanize(key)} ${delta>0?"+":""}${delta.toFixed(2)}`).join(" · ");head.append(el("strong","",`${humanize(item.task_type)} · ${Math.round(Number(item.confidence||0)*100)}%`),el("span","",`${humanize(item.status)}${item.explicit_confirmation_required?" · extra confirmation":""}`));row.append(head,el("p","",changes||"No effective change"),el("p","small-meta",`${item.reason||"No reason recorded"} · baseline precision ${item.baseline_precision==null?"learning":`${(Number(item.baseline_precision)*100).toFixed(1)}%`}`));if(d.review_writes_enabled)row.append(brainMechanicsActions([item.status==="proposed"?{label:item.explicit_confirmation_required?"Confirm & apply":"Apply",payload:{mechanic:"weights",action:"apply",proposal_id:item.proposal_id,confirm_large_change:Boolean(item.explicit_confirmation_required)},options:{primary:true,confirmation:item.explicit_confirmation_required?"At least one weight changes by 0.05 or more. Confirm this task-specific live profile?":"Apply this task-specific scoring profile?"}}:null,item.status==="proposed"?{label:"Reject",payload:{mechanic:"weights",action:"reject",proposal_id:item.proposal_id}}:null,item.status==="approved"?{label:"Rollback",payload:{mechanic:"weights",action:"rollback",proposal_id:item.proposal_id,reason:"operator requested dashboard rollback"},options:{confirmation:"Restore the proposal baseline weights?"}}:null]));weightList.append(row);});

      const recon=d.adaptive_reconsolidation||{},reconCounts=recon.counts||{},reconFeedback=recon.feedback||{},reconRows=recon.proposals||[],reconMetrics=document.getElementById("aj-recon-metrics"),reconList=document.getElementById("aj-recon-list");
      reconMetrics.replaceChildren();reconList.replaceChildren();
      labMetric(reconMetrics,"Proposed",formatNum(reconCounts.proposed||0),"Same-task review items");
      labMetric(reconMetrics,"Applied",formatNum(reconCounts.applied||0),"Versioned or evidenced changes");
      labMetric(reconMetrics,"Reversed",formatNum(reconCounts.reversed||0),"Prior state restored");
      labMetric(reconMetrics,"Reviewed accuracy",reconFeedback.correctness==null?"Learning":`${(Number(reconFeedback.correctness)*100).toFixed(1)}%`,`${formatNum(reconFeedback.reviewed||0)} labels · target 90%`);
      document.getElementById("aj-recon-count").textContent=`${formatNum(reconCounts.proposed||0)} proposals`;document.getElementById("aj-recon-boundary").textContent=String(recon.claim_boundary||"User corrections remain authoritative.");
      if(!reconRows.length)reconList.append(emptyState("No adaptive reconsolidation proposals yet","A memory must influence a task that creates or updates durable same-task evidence."));
      reconRows.slice(0,12).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head");head.append(el("strong","",`${humanize(item.action)} · ${Math.round(Number(item.confidence||0)*100)}%`),el("span","",`${humanize(item.status)}${item.protected_confirmation_required?" · protected":""}`));row.append(head,el("p","",`${truncate(cleanMemory(item.memory_content||"Unavailable"),110)} → ${truncate(cleanMemory(item.evidence_content||"Unavailable"),110)}`),el("p","small-meta",item.reason||"No reason recorded"));if(d.review_writes_enabled)row.append(brainMechanicsActions([item.status==="proposed"?{label:item.protected_confirmation_required?"Confirm protected change":"Apply",payload:{mechanic:"reconsolidation",action:"apply",proposal_id:item.proposal_id,confirm_protected:Boolean(item.protected_confirmation_required)},options:{primary:true,confirmation:item.protected_confirmation_required?"This changes protected identity or preference memory. Confirm explicitly?":"Apply this versioned, reversible reconsolidation?"}}:null,item.status==="applied"?{label:"Undo",payload:{mechanic:"reconsolidation",action:"undo",proposal_id:item.proposal_id}}:null,!item.feedback_label?{label:"Correct",payload:{mechanic:"reconsolidation",action:"feedback",proposal_id:item.proposal_id,label:"correct"}}:null,!item.feedback_label?{label:"Wrong",payload:{mechanic:"reconsolidation",action:"feedback",proposal_id:item.proposal_id,label:"wrong"},options:{confirmation:"Mark wrong and undo if applied?"}}:null]));row.addEventListener("click",()=>selectMemory(item.memory_id));reconList.append(row);});

      const schemas=d.schema_formation||{},schemaCounts=schemas.counts||{},schemaFeedback=schemas.feedback||{},schemaRows=schemas.proposals||[],schemaMetrics=document.getElementById("aj-schema-metrics"),schemaList=document.getElementById("aj-schema-list");
      schemaMetrics.replaceChildren();schemaList.replaceChildren();
      labMetric(schemaMetrics,"Proposed",formatNum(schemaCounts.proposed||0),"Abstractions awaiting review");
      labMetric(schemaMetrics,"Applied",formatNum(schemaCounts.applied||0),"Sources remain live");
      labMetric(schemaMetrics,"Dirty",formatNum(schemaRows.filter(item=>Number(item.result_dirty)).length),"Source correction requires reevaluation");
      labMetric(schemaMetrics,"Reviewed accuracy",schemaFeedback.accuracy==null?"Learning":`${(Number(schemaFeedback.accuracy)*100).toFixed(1)}%`,`${formatNum(schemaFeedback.reviewed||0)} labels · target 70%`);
      document.getElementById("aj-schema-count").textContent=`${formatNum(schemaCounts.proposed||0)} proposals`;document.getElementById("aj-schema-boundary").textContent=String(schemas.claim_boundary||"A cluster and abstraction are review evidence, not a proven rule.");
      if(!schemaRows.length)schemaList.append(emptyState("No schema proposals yet","The weekly pass requires at least three independently used sources."));
      schemaRows.slice(0,12).forEach(item=>{const row=el("article","lab-row"),head=el("div","lab-row-head"),evidence=item.evidence||{};head.append(el("strong","",`${humanize(item.action)} · ${Math.round(Number(item.confidence||0)*100)}%`),el("span","",`${humanize(item.status)}${Number(item.result_dirty)?" · dirty":""}`));row.append(head,el("p","",truncate(cleanMemory(item.abstract_content||item.reason||"No abstraction proposed"),190)),el("p","small-meta",`${formatNum(evidence.source_count||item.included_source_ids?.length||0)} sources · ${formatNum(evidence.distinct_tasks||0)} tasks · ${formatNum(evidence.distinct_task_types||0)} task types · ${formatNum(evidence.episode_count||0)} episodes`));if(d.review_writes_enabled)row.append(brainMechanicsActions([item.status==="proposed"?{label:item.action==="no_schema"?"Confirm no schema":"Apply schema",payload:{mechanic:"schemas",action:"apply",proposal_id:item.proposal_id},options:{primary:true,confirmation:item.action==="no_schema"?"Dismiss this cluster as no schema?":"Create this reversible schema and retain every source?"}}:null,item.status==="applied"?{label:"Undo",payload:{mechanic:"schemas",action:"undo",proposal_id:item.proposal_id}}:null,!item.feedback_label?{label:"Correct",payload:{mechanic:"schemas",action:"feedback",proposal_id:item.proposal_id,label:"correct"}}:null,!item.feedback_label?{label:"Wrong",payload:{mechanic:"schemas",action:"feedback",proposal_id:item.proposal_id,label:"wrong"},options:{confirmation:"Mark wrong and archive the schema if applied?"}}:null]));if(item.result_memory_id)row.addEventListener("click",()=>selectMemory(item.result_memory_id));schemaList.append(row);});

      // Decision funnel
      const funnel = document.getElementById("aj-funnel");
      funnel.replaceChildren();
      const props = aj.proposals || {};
      const funnelTotal = document.getElementById("aj-funnel-total");
      funnelTotal.textContent = `${props.total || 0} proposals`;
      const funnelStages = [
        { label: "Proposals", count: props.total || 0, pct: 100 },
        { label: "Remembered", count: props.remembered || 0, pct: props.total ? Math.round((props.remembered || 0) / props.total * 100) : 0 },
        { label: "Rejected", count: props.rejected || 0, pct: props.total ? Math.round((props.rejected || 0) / props.total * 100) : 0 },
        { label: "Pending review", count: props.pending || 0, pct: props.total ? Math.round((props.pending || 0) / props.total * 100) : 0 },
      ];
      const funnelBar = el("div", "funnel-bar");
      const funnelLabels = el("div", "funnel-labels");
      funnelStages.forEach((s) => {
        const seg = el("div", "funnel-segment");
        seg.style.width = `${s.pct}%`;
        seg.style.background = s.label === "Remembered" ? "var(--mint)" : s.label === "Rejected" ? "var(--red)" : s.label === "Pending review" ? "var(--amber)" : "var(--blue)";
        seg.title = `${s.label}: ${s.count}`;
        funnelBar.append(seg);
        const lab = el("div", "funnel-label");
        lab.append(el("strong", "", `${s.count}`), el("span", "", s.label));
        funnelLabels.append(lab);
      });
      funnel.append(funnelBar, funnelLabels);

      // Orphan linking stats
      const orphanEl = document.getElementById("aj-orphan-stats");
      orphanEl.replaceChildren();
      const orph = aj.orphan_linking || {};
      document.getElementById("aj-orphan-total").textContent = orph.linked_percentage ? `${orph.linked_percentage}% linked` : "—";
      const orphanGrid = el("div", "orphan-stat-grid");
      [
        ["Total memories", `${orph.total_memories || "—"}`],
        ["Total edges", `${orph.total_edges || "—"}`],
        ["Orphan memories", `${orph.orphan_memories || "—"}`],
        ["Graph connected", orph.linked_percentage ? `${orph.linked_percentage}%` : "—"],
      ].forEach(([label, val]) => {
        const row = el("div", "orphan-stat-row");
        row.append(el("span", "orphan-stat-label", label), el("strong", "", val));
        orphanGrid.append(row);
      });
      orphanEl.append(orphanGrid);

      // Edge type breakdown
      const edgeTypes = document.getElementById("aj-edge-types");
      edgeTypes.replaceChildren();
      const edges = aj.edges || {};
      const edgeTitle = el("div", "panel-subtitle", `Edge types (${edges.total || 0} total)`);
      edgeTypes.append(edgeTitle);
      const edgeGrid = el("div", "edge-type-grid");
      const byRel = edges.by_relation || {};
      Object.entries(byRel).forEach(([rel, cnt]) => {
        const tag = el("span", "edge-type-tag");
        tag.textContent = `${rel}: ${cnt}`;
        edgeGrid.append(tag);
      });
      edgeTypes.append(edgeGrid);

      // Recent decisions table
      const recentList = document.getElementById("aj-recent-list");
      recentList.replaceChildren();
      const decisions = aj.recent_decisions || [];
      document.getElementById("aj-recent-count").textContent = `${decisions.length} decisions`;
      if (!decisions.length) {
        recentList.append(emptyState("No decisions yet", "The auto-judge will populate this as it runs."));
      } else {
        const table = el("div", "decision-table");
        decisions.forEach((dec) => {
          const row = el("div", "decision-row");
          const actionBadge = document.createElement("span");
          actionBadge.className = `decision-badge ${dec.action}`;
          actionBadge.textContent = dec.action;
          row.append(
            actionBadge,
            el("span", "decision-type", dec.item_type || "—"),
            el("span", "decision-time", timeAgo(new Date(dec.created_at))),
            el("span", "decision-reason", dec.reason || "")
          );
          table.append(row);
        });
        recentList.append(table);
      }

      // Daily activity chart
      const dailySvg = document.getElementById("aj-daily-svg");
      const daily = aj.daily_decisions || [];
      const W = 800, H = 200, PAD = 20;
      const innerW = W - 2 * PAD, innerH = H - 2 * PAD;
      const dayMap = {};
      daily.forEach((d) => {
        if (!dayMap[d[0]]) dayMap[d[0]] = { day: d[0], remember: 0, reject: 0 };
        if (d[1] === "remember") dayMap[d[0]].remember += d[2];
        else dayMap[d[0]].reject += d[2];
      });
      const days = Object.values(dayMap).sort((a, b) => a.day < b.day ? -1 : 1);
      let svgContent = "";
      if (days.length) {
        const maxVal = Math.max(...days.map((d) => d.remember + d.reject), 1);
        const barW = Math.max(10, Math.min(40, (innerW - days.length * 2) / days.length));
        days.forEach((d, i) => {
          const x = PAD + i * (barW + 3);
          const hR = (d.remember / maxVal) * innerH;
          const hJ = (d.reject / maxVal) * innerH;
          const dateLabel = d.day.slice(5);
          // remember bar (green)
          svgContent += `<rect x="${x}" y="${H - PAD - hR - hJ}" width="${barW}" height="${hR}" fill="var(--mint)" opacity="0.85" rx="2" />`;
          // reject bar (red) on top
          svgContent += `<rect x="${x}" y="${H - PAD - hJ}" width="${barW}" height="${hJ}" fill="var(--red)" opacity="0.7" rx="2" />`;
          // date label
          svgContent += `<text x="${x + barW/2}" y="${H - 4}" text-anchor="middle" font-size="9" fill="var(--text-dim)">${dateLabel}</text>`;
        });
        // Legend
        svgContent += `<rect x="${PAD}" y="${PAD}" width="10" height="10" fill="var(--mint)" rx="2" /><text x="${PAD + 14}" y="${PAD + 9}" font-size="10" fill="var(--text-main)">Remembered</text>`;
        svgContent += `<rect x="${PAD + 90}" y="${PAD}" width="10" height="10" fill="var(--red)" rx="2" /><text x="${PAD + 104}" y="${PAD + 9}" font-size="10" fill="var(--text-main)">Rejected</text>`;
      } else {
        svgContent = `<text x="${W/2}" y="${H/2}" text-anchor="middle" font-size="13" fill="var(--text-dim)">No decisions in the last 14 days</text>`;
      }
      dailySvg.innerHTML = svgContent;

      // Config display
      const configHost = document.getElementById("aj-config-display");
      configHost.replaceChildren();
      const cfg = aj.config || {};
      const cfgGrid = el("div", "config-grid");
      const configured=value=>["1","true","yes","on"].includes(String(value??"").toLowerCase())?"✓ Enabled":"Shadow off";
      [
        ["Enabled", cfg.enabled ? "✓ Yes" : "✗ No"],
        ["Model", cfg.model || "—"],
        ["Brain mechanics model", cfg.brain_mechanics_model || cfg.model || "—"],
        ["Brain mechanics timeout", `${cfg.brain_mechanics_timeout_seconds || "45"}s`],
        ["Links enabled", cfg.links_enabled ? "✓ Yes" : "✗ No"],
        ["Semantic consolidation",configured(cfg.consolidation_enabled)],
        ["Relevance pruning",configured(cfg.pruning_enabled)],
        ["Reconsolidation",`${configured(cfg.reconsolidation_enabled)} · ${cfg.lability_minutes||30} min`],
        ["Schema formation",configured(cfg.schemas_enabled)],
        ["Weight tuning",configured(cfg.weight_tuning_enabled)],
      ].forEach(([label, val]) => {
        const row = el("div", "config-row");
        row.append(el("span", "config-label", label), el("span", "config-value", val));
        cfgGrid.append(row);
      });
      configHost.append(cfgGrid);
    }

    function renderHealth() {
      if(!app.data)return;
      const d=app.data,a=d.audit||{},hygiene=d.memory_hygiene||{},memories=d.memories||[],agent=d.research?.agent_evaluation?.summary||{},experiment=d.research?.experiments||{},quarantined=Number(d.stats?.states?.quarantine ?? memories.filter(memory=>memory.state==="quarantine").length),dirty=Number(d.stats.dirty_memories||0),contradictions=Number(d.contradiction_count||0),unsupported=Number(a.unsupported_inferences||0),duplicates=Number(a.duplicate_groups||0),hygieneCandidates=Number(hygiene.candidate_count||0),pending=Number(d.stats.pending_usage||0),stuck=Number(a.stuck_pending_usage||0),structural=technicalIssueCount(a),decisionItems=dirty+contradictions+quarantined+unsupported+duplicates+hygieneCandidates,decisionTasks=[contradictions,quarantined,dirty,unsupported,duplicates,hygieneCandidates].filter(Boolean).length,maintenance=sleepHealthState(d),status=structural?"repair":"ready",hero=document.getElementById("health-hero");hero.replaceChildren();
      const attention=decisionTasks+Number(maintenance.needsAttention)+Number(stuck>0),statusCopy=structural?["×","Recall may be incomplete",`${formatNum(structural)} index or database checks failed. Pause decisions based on recall until the red repair task is clear.`,"Repair first"]:attention?["✓","Safe to recall",`The storage and search path are intact. ${formatNum(attention)} non-blocking ${attention===1?"area could":"areas could"} use attention; each is separated below.`,"Operational"]:["✓","Healthy and observable","Recall is structurally safe, the review queue is clear, the learning loop is caught up, and maintenance is current.","Healthy"],symbol=el("div","health-symbol",statusCopy[0]),copy=el("div","");copy.append(el("h2","",statusCopy[1]),el("p","small-meta",statusCopy[2]));hero.append(symbol,copy,el("div",`health-score-large ${status}`,statusCopy[3]));
      metricCards("health-metrics",[["Recall integrity",structural?"Blocked":"Passed",structural?`${formatNum(structural)} checks can affect retrieval`:"Database, FTS, features, and scores agree",structural?"!":"✓"],["Memory quality",decisionItems,decisionItems?`${formatNum(decisionTasks)} review queues; normal recall still works`:"No conflicting, blocked, or unsupported items","↔"],["Learning evidence",formatNum(agent.explicit_labels||0),experiment.active?"Controlled experiment collecting randomized tasks":pending?`${formatNum(pending)} automatic outcomes still resolving`:"Explicit labels power accuracy and causal gates","◎"],["Sleep maintenance",maintenance.label,maintenance.copy,"◐"]],true);
      renderMemoryFeedback(d);
      renderHealthActions({structural,dirty,contradictions,quarantined,unsupported,duplicates,hygieneCandidates,pending,stuck,maintenance});
      renderHealthAreas({structural,decisionTasks,pending,stuck,maintenance,documents:Number(d.stats.documents||0),chunks:Number(d.stats.document_chunks||0),explicitLabels:Number(agent.explicit_labels||0),experimentActive:Boolean(experiment.active)});
      renderHealthMaintenance(maintenance,d);
      const checkDefs=[["Recall integrity gate",structural],["Duplicate memory groups",duplicates],["Links without preserved evidence",a.unexplained_edges||0],["Invalid link evidence",a.invalid_edge_evidence||0],["Missing search-index rows",a.missing_fts_rows||0],["Orphan search-index rows",a.orphan_fts_rows||0],["Missing semantic features",a.missing_feature_memories||0],["Semantic feature-stat mismatches",a.feature_stat_mismatches||0],["Orphan semantic feature stats",a.orphan_feature_stats||0],["Missing context-index terms",a.missing_memory_context_terms||0],["Orphan context-index terms",a.orphan_memory_context_terms||0],["Invalid context-index terms",a.invalid_memory_context_terms||0],["Invalid score values",a.invalid_score_rows||0],["Invalid context metadata",a.invalid_context_metadata||0],["Context-dependent memories without scope",a.contextless_memories||0],["Invalid write decisions",a.invalid_memory_write_decisions||0],["Invalid context outcomes",a.invalid_memory_context_outcomes||0],["Orphan trace events",a.orphan_memory_trace_events||0],["Orphan write decisions",a.orphan_memory_write_decisions||0],["Orphan context outcomes",a.orphan_memory_context_outcomes||0],["Orphan document chunks",a.orphan_document_chunks||0],["Orphan Sleep proposals",a.orphan_sleep_proposals||0],["Orphan experiment assignments",a.orphan_experiment_assignments||0],["Orphan agent assignments",a.orphan_agent_assignments||0],["Invalid prospective trackers",a.orphan_prospective_items||0],["Stuck usage records",stuck]],checks=document.getElementById("health-checks");checks.replaceChildren();
      checkDefs.forEach(([label,value])=>{const row=el("div","health-row");row.append(el("span","",label),el("span",`health-value ${value?"warn":"good"}`,value?formatNum(value):"Clear"));checks.append(row);});
      const states=d.stats.states||{},total=Math.max(1,Object.values(states).reduce((n,v)=>n+v,0)),strip=document.getElementById("lifecycle-strip"),glossary=document.getElementById("health-glossary");strip.replaceChildren();glossary.replaceChildren();Object.entries(states).forEach(([state,count])=>{const part=el("span","");part.style.width=`${count/total*100}%`;part.style.background=stateColor(state);part.title=`${humanize(state)}: ${count}`;strip.append(part);});
      const definitions={active:"Available for the connected agent to recall.",cold:"Kept safely, but ranked lower until it becomes relevant again.",archived:"Retained for history but excluded from normal recall.",quarantine:"Blocked from recall because it needs review.",tombstoned:"Marked as removed while preserving an audit trail."};Object.entries(states).forEach(([state,count])=>{const row=el("div","glossary-row");row.append(el("strong","",`${humanize(state)} · ${formatNum(count)}`),el("p","",definitions[state]||"A managed Cortex lifecycle state."));glossary.append(row);});
      const details=document.getElementById("system-details");details.replaceChildren();[["Database",formatBytes(d.stats.db_bytes)],["Schema",`v${d.stats.schema_version}`],["Documents",d.stats.documents],["Document chunks",d.stats.document_chunks],["Versions",d.version_count],["Episodes",d.stats.episodes],["Generated",formatDate(d.generated_at)]].forEach(([label,value])=>{details.append(el("dt","",label),el("dd","",value));});
      renderHealthReviewer();
    }

    function renderMemoryFeedback(data){
      const trace=data.memory_trace_summary||{},writes=data.memory_write_summary||{},contexts=data.context_feedback||{},hygiene=data.memory_hygiene||{},precision=trace.observed_selection_precision,usefulness=trace.resolved_memory_usefulness,selected=Number(trace.selected_memories||0),resolved=Object.values(trace.ratings||{}).reduce((sum,value)=>sum+Number(value||0),0),decisions=writes.decisions||{},created=Number(decisions.created||0),updated=Number(decisions.updated||0),ignored=Number(decisions.ignored||0),hygieneReasons=hygiene.by_reason||{};
      metricCards("memory-feedback-metrics",[
        ["Selection precision",precision===null||precision===undefined?"Learning":`${(Number(precision)*100).toFixed(1)}%`,selected?`${formatNum(trace.influenced_memories||0)} attributed uses among ${formatNum(selected)} selected memories`:"No resolved selections yet","◎"],
        ["Resolved usefulness",usefulness===null||usefulness===undefined?"Learning":`${(Number(usefulness)*100).toFixed(1)}%`,resolved?`${formatNum(resolved)} rated memory outcomes`:"Needs explicit helpful, validated, corrected, or harmful outcomes","✓"],
        ["False positives",formatNum(trace.false_positive_memories||0),`${trace.false_positive_rate===null||trace.false_positive_rate===undefined?"Rate learning":`${(Number(trace.false_positive_rate)*100).toFixed(1)}% of rated memories`} · ${formatNum(trace.context_failures||0)} context failures`,"!"],
        ["Adaptive contexts",formatNum(contexts.buckets||0),`${formatNum(contexts.positive_buckets||0)} raised · ${formatNum(contexts.downweighted_buckets||0)} downweighted`,"↔"],
        ["Hygiene queue",formatNum(hygiene.candidate_count||0),`${formatNum(hygieneReasons.legacy_tool_telemetry||0)} legacy tool memories · ${formatNum(hygiene.links?.unexplained||0)} unexplained links`,"⌁"]
      ],true);
      document.getElementById("memory-feedback-storage").textContent=`${formatNum(created)} created · ${formatNum(updated)} updated · ${formatNum(ignored)} ignored`;
      document.getElementById("memory-feedback-boundary").textContent=String(trace.claim_boundary||"Selection and outcome evidence is observational. It does not prove that memory caused the answer or task result.");
    }

    function renderHealthActions({structural,dirty,contradictions,quarantined,unsupported,duplicates,hygieneCandidates,pending,stuck,maintenance}) {
      const host=document.getElementById("health-actions"),summary=document.getElementById("health-action-summary"),decisionTasks=[contradictions,quarantined,dirty,unsupported,duplicates].filter(Boolean).length,actions=[];host.replaceChildren();summary.replaceChildren();
      [[`${structural?1:0} blocking ${structural?"repair":"repairs"}`,structural?"bad":""],[`${decisionTasks} optional ${decisionTasks===1?"review":"reviews"}`,decisionTasks?"review":""],[`${pending} learning automatically`,pending?"auto":""],[maintenance.label,maintenance.needsAttention?"review":""]].forEach(([copy,tone])=>summary.append(el("span",`health-summary-chip ${tone}`,copy)));
      if(structural)actions.push({tone:"bad",priority:"Fix now",title:"Repair search and database integrity",count:`${formatNum(structural)} failed checks`,summary:"Search results may be incomplete. Resolve this before evaluating the quality of individual memories.",why:"Structural failures are the only Health category that can make ordinary recall incomplete.",steps:["Open the failed checks below and note their exact names.","Copy and run the audit command on the Cortex host.","Repair the reported index data or restore the latest known-good backup, then run the audit again."],complete:"The audit returns OK and every structural check reads Clear.",target:"checks",primary:"Show failed checks",instructions:"cortex-memory audit",copyLabel:"Copy audit command"});
      if(quarantined)actions.push({tone:"review",priority:"Needs your decision",title:"Review blocked memories",count:`${formatNum(quarantined)} blocked`,summary:"These memories cannot enter agent context until you decide whether they are safe and accurate.",why:"Cortex detected unsafe prompt-like content or another reason to block recall instead of guessing.",steps:["Open the blocked memories and read the quarantine reason and provenance.","If the underlying fact is valid, restate it as clean evidence through your agent.","If it is unsafe or untrustworthy, leave it quarantined; no deletion is required."],complete:"Every blocked item is either deliberately quarantined or replaced by a safe, sourced memory.",target:"quarantine",primary:"Open blocked memories",instructions:"Review the quarantined Cortex memories. Inspect each reason and source. Restate valid facts as clean, versioned evidence; leave unsafe or untrusted content blocked. Do not hard-delete the audit history.",copyLabel:"Copy instructions"});
      if(contradictions)actions.push({tone:"review",priority:"Simple review",title:"Choose between conflicting memories",count:`${formatNum(contradictions)} pairs`,summary:"Cortex will show you one pair at a time. You only choose which statement is current, whether both are valid, or whether to skip it.",why:"A conflict can mean an old fact was replaced, two time periods differ, or both claims are valid in different situations.",steps:["Press Review here.","Read the two memory cards and their dates.","Choose one plain-language answer. Cortex preserves the old history automatically."],complete:"The pair is marked contextual or the outdated memory is archived with an audit trail.",target:"contradictions",primary:"Review here"});
      if(dirty)actions.push({tone:"review",priority:"Needs your decision",title:"Re-check memories built from changed evidence",count:`${formatNum(dirty)} changed`,summary:"These derived memories depend on evidence that was corrected or replaced.",why:"Cortex lowers trust in downstream conclusions when their supporting evidence changes.",steps:["Open dependency links and inspect the changed source.","Confirm whether the derived memory is still true.","Correct or regenerate it if the conclusion no longer follows from active evidence."],complete:"Every derived memory points to current evidence or has been corrected, cooled, or archived reversibly.",target:"dependencies",primary:"Open dependency map",instructions:"Review Cortex memories whose supporting evidence changed. Verify each derived claim against active sources. Preserve it if still supported; otherwise create a versioned correction or allow it to cool. Keep provenance and history intact.",copyLabel:"Copy instructions"});
      if(unsupported)actions.push({tone:"review",priority:"Simple review",title:"Check inferred memories",count:`${formatNum(unsupported)} inferred`,summary:"Cortex will show one statement at a time. Tell it whether the statement is correct, should be archived, or should be skipped for now.",why:"Cortex refuses to treat an agent guess as durable truth until evidence or an explicit user decision supports it.",steps:["Press Review here.","Read the statement and its source.","Choose Correct, Archive, or Not sure."],complete:"The inference is explicitly confirmed or moved safely out of normal recall.",target:"unsupported",primary:"Review here"});
      if(duplicates)actions.push({tone:"review",priority:"Review in Sleep",title:"Inspect exact duplicate groups",count:`${formatNum(duplicates)} groups`,summary:"These do not damage recall, but consolidation can reduce repeated evidence competing for the same context budget.",why:"Duplicate content is a memory-quality issue, not an index-integrity failure.",steps:["Open the Sleep log.","Run a shadow cycle to preview consolidation candidates.","Keep the stronger canonical memory and preserve the source records."],complete:"The duplicate groups are consolidated or deliberately retained.",target:"sleep",primary:"Open Sleep log"});
      if(hygieneCandidates)actions.push({tone:"review",priority:"Review in Sleep",title:"Remove telemetry and placeholder noise from normal recall",count:`${formatNum(hygieneCandidates)} candidates`,summary:"These are deterministic tool-call records or low-information placeholders already preserved elsewhere. Cortex proposes a reversible cold/archive move, never deletion.",why:"Execution telemetry belongs in Tool notes, while evolving vault files should keep one versioned memory instead of many archived copies.",steps:["Open the Sleep log.","Run a shadow cycle and inspect each lifecycle reason.","Apply only through the reversible maintenance path after the reasons match the source."],complete:"Tool telemetry remains in its ledger but no longer appears as normal memory, and placeholder chunks are cooled for review.",target:"sleep",primary:"Open Sleep log"});
      if(maintenance.needsAttention)actions.push({tone:"review",priority:"Maintenance",title:maintenance.actionTitle,count:maintenance.label,summary:maintenance.actionCopy,why:"Sleep is the auditable maintenance path. A missing or failed cycle does not break recall, but it removes evidence that offline review is current.",steps:["Open the Sleep log and inspect the timer state.","Run a manual shadow cycle; it cannot change live memory.","Confirm the run completes and produces an evidence ledger."],complete:"A recent shadow cycle completed and the next scheduled window is visible.",target:"sleep",primary:"Open Sleep log"});
      if(stuck)actions.push({tone:"auto",priority:"Check adapter",title:"Close stale learning outcomes",count:`${formatNum(stuck)} older than one day`,summary:"These records no longer look like ordinary in-flight tasks. The connected harness may not be reporting outcomes.",why:"Stale feedback weakens budget learning but does not make stored recall unsafe.",steps:["Verify the harness reports task outcomes after recall.","Complete or cancel abandoned tasks through the normal adapter path.","Run the audit again and confirm Stuck usage records returns to Clear."],complete:"No pending usage record is older than one day.",instructions:"cortex-memory audit",copyLabel:"Copy audit command"});
      if(!actions.length)actions.push({tone:"good",priority:"Complete",title:"Nothing needs your attention",count:"0 tasks",summary:"Cortex is structurally healthy, its review queue is clear, and the learning loop is caught up.",why:"All current health gates have passed.",steps:["Continue normal use. Cortex will keep monitoring integrity, evidence, and outcomes."],complete:"No action is required.",target:null});
      actions.forEach((action,index)=>{const row=el("article",`health-action ${action.tone}`),number=el("div","health-action-index",String(index+1).padStart(2,"0")),body=el("div","health-action-copy"),head=el("div","health-action-head");head.append(el("strong","",action.title),el("span","health-priority",action.priority),el("span","health-action-count",action.count));body.append(head,el("p","",action.summary));const why=el("div","health-action-why");why.append(el("b","","Why it is flagged: "),document.createTextNode(action.why));body.append(why);const steps=el("ol","health-action-steps");action.steps.forEach(step=>steps.append(el("li","",step)));body.append(steps);const complete=el("div","health-action-complete");complete.append(el("b","","Resolved when: "),document.createTextNode(action.complete));body.append(complete);if(action.target||action.instructions){const buttons=el("div","health-action-buttons");if(action.target){const primary=el("button",index===0&&action.tone!=="auto"?"primary-btn":"ghost-btn",action.primary);primary.type="button";primary.addEventListener("click",()=>openHealthTarget(action.target));buttons.append(primary);}if(action.instructions){const copy=el("button","ghost-btn",action.copyLabel||"Copy instructions");copy.type="button";copy.addEventListener("click",()=>copyHealthInstructions(action.instructions));buttons.append(copy);}body.append(buttons);}row.append(number,body);host.append(row);});
    }
    function renderHealthAreas({structural,decisionTasks,pending,stuck,maintenance,documents,chunks,explicitLabels,experimentActive}) {const host=document.getElementById("health-areas");host.replaceChildren();const areas=[["01 · Recall integrity",structural?"bad":"good",structural?"Blocks safe recall":"Passed",structural?`${structural} database, search, feature, or score checks can make retrieval incomplete.`:"The database, full-text index, semantic features, and score ranges agree."],["02 · Memory quality",decisionTasks?"review":"good",decisionTasks?`${decisionTasks} review queues`:"Clear",decisionTasks?"Conflicts, blocked items, duplicates, or unsupported claims can be improved without stopping normal recall.":"No memory-quality queue needs a human decision."],["03 · Learning evidence",stuck?"review":pending?"auto":"good",stuck?"Adapter check":experimentActive?"Experiment active":`${formatNum(explicitLabels)} labels`,stuck?`${stuck} outcome records are stale; learning may lag.`:experimentActive?"Randomized tasks are being assigned before recall; only explicit labels unlock causal estimates.":pending?`${pending} automatic outcomes are still resolving; they do not count as causal evidence.`:"Accuracy and experiment gates use explicit outcomes; missing labels remain visible as missing data."],["04 · Sleep maintenance",maintenance.needsAttention?"review":"good",maintenance.label,`${maintenance.copy} Source coverage: ${documents} notes and ${formatNum(chunks)} searchable sections.`]];areas.forEach(([title,tone,status,copy])=>{const row=el("div",`health-area ${tone}`);row.append(el("span","health-area-dot"));const text=el("div","");text.append(el("strong","",title),el("p","",copy));row.append(text,el("span","health-area-status",status));host.append(row);});}
    function sleepHealthState(data){const runs=[...(data.sleep_runs||[])].sort(sleepNewest),latest=runs[0],schedule=data.sleep_schedule||{};if(!latest)return{label:"Not run yet",copy:"No offline maintenance report exists.",needsAttention:true,actionTitle:"Create the first maintenance proof",actionCopy:"Run a shadow cycle so Health can show replay, proposals, and zero live changes."};const status=String(latest.status||"").toLowerCase(),ended=Date.parse(latest.completed_at||latest.started_at),ageHours=Number.isFinite(ended)?(Date.now()-ended)/3600000:Infinity;if(status==="failed")return{label:"Last run failed",copy:`The latest cycle stopped ${timeAgo(new Date(latest.started_at))}. Recall still works.`,needsAttention:true,run:latest,actionTitle:"Inspect the failed Sleep cycle",actionCopy:"The last offline maintenance pass did not finish; its error and completed phases are preserved in the run log."};if(ageHours>36)return{label:"Overdue",copy:`The last completed cycle is ${Math.floor(ageHours)} hours old.`,needsAttention:true,run:latest,actionTitle:"Refresh offline maintenance",actionCopy:"The nightly maintenance record is older than its expected window. Check the timer, then run a shadow cycle."};return{label:"Current",copy:`Last ${humanize(latest.mode)} cycle completed ${timeAgo(new Date(latest.completed_at||latest.started_at))}.`,needsAttention:schedule.active===false,run:latest,actionTitle:"Enable the Sleep timer",actionCopy:"The latest run is current, but the installed schedule reports inactive."};}
    function renderHealthMaintenance(maintenance,data){const host=document.getElementById("health-maintenance"),run=maintenance.run,effects=run?data.sleep_effects?.[run.run_id]||{}:{};host.replaceChildren();const card=el("div",`sleep-effect-card ${maintenance.needsAttention?"proposed":"applied"}`);card.append(el("span","","Maintenance status"),el("strong","",maintenance.label),el("p","",maintenance.copy));const signals=el("div","sleep-effect-signals");if(run){signals.append(el("b","",`${humanize(run.mode)} · ${humanize(run.status)}`),el("b","",`${sleepDuration(run)} runtime`),el("b","",`${formatNum(effects.live_edge_changes||0)} live link changes`),el("b","",`${formatNum(effects.live_state_changes||0)} live state changes`));}else signals.append(el("b","","No run evidence"));card.append(signals);host.append(card);}
    async function copyHealthInstructions(text){try{await navigator.clipboard.writeText(text);showToast("Health instructions copied");}catch(_){const field=document.createElement("textarea");field.value=text;field.setAttribute("readonly","");field.style.position="fixed";field.style.opacity="0";document.body.append(field);field.select();document.execCommand("copy");field.remove();showToast("Health instructions copied");}}
    function openHealthTarget(target){if(target==="checks"){const details=document.querySelector("#view-health .health-technical");details.open=true;details.scrollIntoView({behavior:"smooth",block:"center"});return;}if(target==="sleep"){setView("cognition");requestAnimationFrame(()=>document.getElementById("cortex-sleep").scrollIntoView({behavior:"smooth",block:"start"}));return;}if(target==="contradictions"||target==="unsupported"){app.review.filter=target==="contradictions"?"conflicts":"claims";app.review.index=0;setView("review");renderReviewInbox();return;}if(target==="quarantine"){document.getElementById("memory-state").value="quarantine";app.explorerPage=1;setView("explorer");renderExplorer();return;}setView("graph");if(target==="dependencies")document.getElementById("graph-relation").value="dependency";renderActiveGraph();}

    function healthReviewItems(kind=app.healthReview.kind){const reviews=app.data?.health_reviews||{};return kind==="contradictions"?(reviews.contradictions||[]):kind==="unsupported"?(reviews.unsupported_inferences||[]):[];}
    function startHealthReview(kind){app.healthReview={kind,index:0,pending:null,busy:false};setView("health");renderHealthReviewer();requestAnimationFrame(()=>document.getElementById("health-reviewer").scrollIntoView({behavior:"smooth",block:"start"}));}
    function closeHealthReview(){app.healthReview={kind:null,index:0,pending:null,busy:false};document.getElementById("health-reviewer").hidden=true;}
    function nextHealthReview(){const items=healthReviewItems();if(!items.length)return;app.healthReview.index=(app.healthReview.index+1)%items.length;app.healthReview.pending=null;renderHealthReviewer();}
    function conflictMemory(item,prefix){return{id:item[`${prefix}_id`],content:item[`${prefix}_content`],kind:item[`${prefix}_kind`],source_category:item[`${prefix}_source_category`],source_ref:item[`${prefix}_source_ref`],observed_at:item[`${prefix}_observed_at`],valid_from:item[`${prefix}_valid_from`],valid_to:item[`${prefix}_valid_to`],confidence:item[`${prefix}_confidence`]};}
    function renderHealthMemoryCard(memory,label){const card=el("article","health-memory-card"),head=el("div","health-memory-label");head.append(el("span","",label),el("span","",shortId(memory.id)));card.append(head,el("div","health-memory-content",cleanMemory(memory.content)));const meta=el("div","health-memory-meta"),windowCopy=memory.valid_from||memory.valid_to?`${memory.valid_from||"open"} → ${memory.valid_to||"open"}`:"No time limit recorded";meta.append(el("span","",`Source: ${sourceName(memory)}`),el("span","",`Observed: ${formatDate(memory.observed_at)}`),el("span","",`Valid: ${windowCopy}`));card.append(meta);const details=el("button","ghost-btn","View full provenance");details.type="button";details.addEventListener("click",()=>selectMemory(memory.id));card.append(details);return card;}
    function healthChoice(label,action,secondary=false){const button=el("button",`health-choice ${secondary?"secondary":""}`,label);button.type="button";button.disabled=app.healthReview.busy||app.data?.review_writes_enabled!==true;button.addEventListener("click",()=>{if(action==="skip")nextHealthReview();else{app.healthReview.pending=action;renderHealthReviewer();}});return button;}
    function renderHealthReviewer(){const panel=document.getElementById("health-reviewer"),body=document.getElementById("health-review-body"),progress=document.getElementById("health-review-progress"),kind=app.healthReview.kind;if(!kind){panel.hidden=true;return;}panel.hidden=false;const items=healthReviewItems(kind),title=document.getElementById("health-review-title"),subtitle=document.getElementById("health-review-subtitle"),writable=app.data?.review_writes_enabled===true;title.textContent=kind==="contradictions"?"Resolve conflicting memories":"Check inferred memories";subtitle.textContent=writable?(kind==="contradictions"?"Compare two statements and choose the relationship. The graph is optional.":"Confirm what you know; archive guesses you do not want in normal recall."):"This installation is in read-only mode. Set CORTEX_DASHBOARD_REVIEWS=1 on the host to enable confirmed decisions.";progress.replaceChildren();body.replaceChildren();if(!items.length){progress.hidden=true;const done=el("div","health-review-done");done.append(el("strong","","This review queue is clear"),el("p","","Cortex has recorded your decisions and preserved the memory history."));const close=el("button","primary-btn","Return to Health");close.type="button";close.addEventListener("click",closeHealthReview);done.append(close);body.append(done);return;}progress.hidden=false;app.healthReview.index=Math.max(0,Math.min(app.healthReview.index,items.length-1));const track=el("div","health-review-progress-track"),fill=el("div","health-review-progress-fill");fill.style.width=`${(app.healthReview.index+1)/items.length*100}%`;track.append(fill);progress.append(track,el("span","",`${app.healthReview.index+1} of ${formatNum(items.length)}`));const item=items[app.healthReview.index],question=el("div","health-review-question");if(kind==="contradictions"){question.append(el("strong","","Which statement should Cortex use now?"),el("p","","Choose A or B if one replaced the other. Choose Both if they describe different times or situations."));const pair=el("div","health-review-pair"),first=conflictMemory(item,"src"),second=conflictMemory(item,"dst");pair.append(renderHealthMemoryCard(first,"Memory A"),el("div","health-review-versus","or"),renderHealthMemoryCard(second,"Memory B"));const choices=el("div","health-review-choices");choices.append(healthChoice("Memory A is current","keep_first"),healthChoice("Memory B is current","keep_second"),healthChoice("Both are valid in different situations","both_valid"),healthChoice("I’m not sure — show the next pair","skip",true));body.append(question,pair,choices);}else{question.append(el("strong","","Should Cortex treat this as something you confirmed?"),el("p","","Correct keeps it in normal recall as an explicit decision. Archive removes it from normal recall but preserves its history."));const memory={...item},choices=el("div","health-review-choices");choices.append(healthChoice("Yes — this is correct","confirm"),healthChoice("No — archive this guess","archive"),healthChoice("I’m not sure — show the next one","skip",true));body.append(question,renderHealthMemoryCard(memory,"Inferred memory"),choices);}if(app.healthReview.pending)renderHealthConfirmation(item,kind);}
    function renderHealthConfirmation(item,kind){const action=app.healthReview.pending,box=el("div","health-review-confirm"),copy=kind==="contradictions"?(action==="keep_first"?"Cortex will keep Memory A active and archive Memory B as superseded.":action==="keep_second"?"Cortex will keep Memory B active and archive Memory A as superseded.":"Cortex will keep both memories active and mark the relationship as contextual, not contradictory."):(action==="confirm"?"Cortex will record this as explicitly confirmed by you and keep it in normal recall.":"Cortex will archive this inference. It remains in history and can be restored later.");box.append(el("strong","","Confirm this decision"),el("p","",`${copy} The original content and audit trail are preserved.`));const buttons=el("div","health-action-buttons"),confirm=el("button","primary-btn",app.healthReview.busy?"Saving…":"Confirm decision"),cancel=el("button","ghost-btn","Cancel");confirm.type="button";cancel.type="button";confirm.disabled=app.healthReview.busy;cancel.disabled=app.healthReview.busy;confirm.addEventListener("click",()=>submitHealthDecision(item,kind,action));cancel.addEventListener("click",()=>{app.healthReview.pending=null;renderHealthReviewer();});buttons.append(confirm,cancel);box.append(buttons);document.getElementById("health-review-body").append(box);}
    async function submitHealthDecision(item,kind,resolution){if(app.healthReview.busy)return;app.healthReview.busy=true;renderHealthReviewer();try{const path=kind==="contradictions"?"/api/review/conflict":"/api/review/inference",payload=kind==="contradictions"?{first_id:item.src_id,second_id:item.dst_id,resolution}:{memory_id:item.id,resolution};await authRequest(path,payload);app.healthReview.busy=false;app.healthReview.pending=null;showToast("Decision saved; memory history preserved");await load(true);}catch(error){app.healthReview.busy=false;showToast(`Could not save review: ${error.message}`,true);renderHealthReviewer();}}

    function memoryRole(row){return String(row.record_role||"canonical");}
    function roleLabel(role){return({canonical:"Readable memory",reference:"Reference evidence",event:"Event",claim:"Unconfirmed claim"})[role]||humanize(role);}
    function roleViewMatches(row,view){
      if(view==="readable")return memoryRole(row)==="canonical";
      if(view==="reference")return memoryRole(row)==="reference";
      if(view==="clarity")return row.needs_clarity===true;
      return true;
    }
    function renderRoleViewBar(){
      const counts=app.data?.refinery?.view_counts||{};
      document.querySelectorAll("[data-role-view]").forEach(button=>{
        const view=button.dataset.roleView,count=counts[view],active=view===app.explorerRole;
        const badge=button.querySelector("span");if(badge)badge.textContent=count===undefined?"—":formatNum(count);
        button.classList.toggle("active",active);button.setAttribute("aria-selected",String(active));
      });
      const notes={
        readable:"Concise statements the agent can act on. Counts cover the complete audited index; the list below stays bounded for speed.",
        reference:"Documents, code, tables, configuration, and raw technical material. Raw content stays collapsed until you open it.",
        clarity:"Unresolved, mixed, contextless, unsupported, or status-like records. Review them in the Explorer or let the Auto-Judge handle them.",
        all:"The complete audited index, including events, claims, and archived history."
      };
      document.getElementById("role-view-note").textContent=notes[app.explorerRole]||notes.readable;
    }
    function usageEvidenceLine(memory){
      const helpful=Number(memory.helpful_count||0)+Number(memory.validated_count||0),harmful=Number(memory.harmful_count||0)+Number(memory.correction_count||0);
      let line=`${formatNum(memory.retrieved_count||0)} recalled · ${formatNum(memory.used_count||0)} used`;
      if(helpful)line+=` · ${formatNum(helpful)} helpful`;
      if(harmful)line+=` · ${formatNum(harmful)} harmful`;
      return line;
    }
    function renderMemoryCard(memory){
      const role=memoryRole(memory),card=el("article","memory-card");card.tabIndex=0;
      const head=el("div","memory-card-head");
      head.append(el("span",`role-chip role-${role}`,roleLabel(role)),el("span",`status-chip ${memory.state}`,humanize(memory.state)),el("span","chip",humanize(memory.kind)));
      head.append(el("span","card-date",shortDate(memory.observed_at)));
      card.append(head);
      const fallbackTitle=truncate(cleanMemory(memory.content),110);
      card.append(el("h3","memory-card-title",memory.display_title||fallbackTitle));
      const summary=memory.display_summary||truncate(cleanMemory(memory.content),300);
      if(summary&&summary!==memory.display_title)card.append(el("p","memory-card-summary",summary));
      const meta=el("div","memory-card-meta");
      const applies=el("span","");applies.append(el("strong","","Applies:"),document.createTextNode(` ${memory.applies_when||"Applies everywhere"}`));
      const source=el("span","");source.append(el("strong","","Source:"),document.createTextNode(` ${humanize(memory.source_category)} · ${sourceName(memory)}`));
      const kept=el("span","");kept.append(el("strong","","Kept because:"),document.createTextNode(` ${memory.retention_reason||"Retained by the storage quality checks."}`));
      const use=el("span","");use.append(el("strong","","Evidence:"),document.createTextNode(` ${usageEvidenceLine(memory)}`));
      meta.append(applies,source,kept,use);card.append(meta);
      const flags=(memory.readability_flags||[]).filter(flag=>flag!=="uncertain_language");
      if(memory.needs_clarity&&flags.length){
        const flagRow=el("div","memory-card-flags");
        flags.slice(0,5).forEach(flag=>flagRow.append(el("span","flag-chip",humanize(flag))));
        card.append(flagRow);
      }
      card.append(el("div","memory-mobile-meta",`${humanize(memory.kind)} · ${shortDate(memory.observed_at)} · tap for full memory`));
      if(role==="reference"){
        const raw=document.createElement("details");raw.className="raw-evidence";
        const summaryEl=document.createElement("summary");summaryEl.textContent="View raw evidence";
        const pre=document.createElement("pre");pre.textContent=String(memory.content||"");
        raw.append(summaryEl,pre);raw.addEventListener("click",event=>event.stopPropagation());
        raw.addEventListener("keydown",event=>event.stopPropagation());
        card.append(raw);
      }
      card.addEventListener("click",()=>selectMemory(memory.id));
      card.addEventListener("keydown",event=>{if(event.key==="Enter")selectMemory(memory.id);});
      return card;
    }
    function renderExplorer() {
      if(!app.data)return;
      renderRoleViewBar();
      const query=document.getElementById("memory-search").value.trim().toLowerCase(),kind=document.getElementById("memory-kind").value,state=document.getElementById("memory-state").value,source=document.getElementById("memory-source").value,sort=document.getElementById("memory-sort").value;
      let rows=(app.data.memories||[]).filter(row=>roleViewMatches(row,app.explorerRole)&&(!query||searchText(row).includes(query))&&(!kind||row.kind===kind)&&(!state||row.state===state)&&(!source||row.source_category===source));
      if(sort==="signal")rows.sort((a,b)=>signalScore(b)-signalScore(a));else if(sort==="used")rows.sort((a,b)=>Number(b.used_count||0)-Number(a.used_count||0));else if(sort==="confidence")rows.sort((a,b)=>Number(b.confidence||0)-Number(a.confidence||0));else rows.sort((a,b)=>Date.parse(b.observed_at)-Date.parse(a.observed_at));
      const pages=Math.max(1,Math.ceil(rows.length/app.pageSize));app.explorerPage=Math.min(app.explorerPage,pages);const start=(app.explorerPage-1)*app.pageSize,visible=rows.slice(start,start+app.pageSize),host=document.getElementById("memory-list");host.replaceChildren();host.removeAttribute("aria-busy");
      const fullTotal=Number(app.data?.refinery?.view_counts?.[app.explorerRole]);
      const totalNote=Number.isFinite(fullTotal)&&fullTotal>rows.length?` of ${formatNum(fullTotal)} total`:"";
      document.getElementById("result-count").textContent=`${formatNum(rows.length)} matching${totalNote}`;document.getElementById("page-label").textContent=`Page ${app.explorerPage} of ${pages}`;document.getElementById("page-prev").disabled=app.explorerPage<=1;document.getElementById("page-next").disabled=app.explorerPage>=pages;
      if(!visible.length){
        const copy={
          readable:"No readable memories match. New canonical statements appear here; raw material lives under Reference evidence.",
          reference:"No reference evidence matches. Vault sections, code, tables, and logs are filed here automatically.",
          clarity:"Nothing needs clarity right now. Flagged records arrive here until the Auto-Judge reviews them.",
          all:"Try a broader phrase or remove one of the filters."
        };
        host.append(emptyState("No records in this view",copy[app.explorerRole]||copy.all));return;
      }
      visible.forEach(memory=>host.append(renderMemoryCard(memory)));
    }

    async function selectTrace(taskId, { updateUrl = true } = {}) {
      if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.delete("memory");
        url.searchParams.set("trace", taskId);
        window.history.replaceState({}, "", url);
      }
      app.selectedId=null;app.selectedTrace=taskId;openDrawer("Recall trace");const body=document.getElementById("drawer-body");body.replaceChildren();body.append(emptyState("Loading recall trace…","Fetching the complete retrieval and attribution record."));
      try{
        const response=await fetch(`/api/trace?task_id=${encodeURIComponent(taskId)}`,{cache:"no-store"});
        if(!response.ok)throw new Error(`HTTP ${response.status}`);
        const payload=await response.json(),trace=payload.trace||{},candidates=trace.candidate_memories||[],influence=new Map((trace.influence||[]).map(item=>[item.memory_id,item])),selected=candidates.filter(item=>item.selected),attributed=selected.filter(item=>influence.get(item.memory_id)?.influenced),feedback=payload.feedback||{};
        body.replaceChildren();
        const summary=el("section","trace-summary"),chips=el("div","chip-row");
        chips.append(el("span","chip",humanize(trace.recall_mode||"unknown")+" recall"),el("span","chip",humanize(trace.outcome||"pending")),el("span","chip",timeAgo(new Date(trace.created_at))));
        summary.append(chips,el("h2","",truncate(cleanMemory(trace.goal||"Private memory-bearing task"),190)),el("p","",trace.retrieval_reason||"Cortex ran bounded retrieval for this task."));
        const counts=el("div","trace-counts");
        [[candidates.length,"Considered"],[selected.length,"Recalled"],[attributed.length,"Attributed"]].forEach(([value,label])=>{const item=el("div","trace-count");item.append(el("strong","",formatNum(value)),el("span","",label));counts.append(item);});
        summary.append(counts);body.append(summary);
        const list=detailSection("Memory-by-memory evidence");
        if(!candidates.length)list.append(el("p","small-meta","Cortex recorded no memory candidates for this task."));
        candidates.forEach(candidate=>{
          const item=el("article","trace-memory"),used=influence.get(candidate.memory_id),status=used?.influenced?"Attributed to answer":candidate.selected?"Recalled; no use evidence":"Considered; not recalled",head=el("div","trace-memory-head"),open=el("button","ghost-btn",`Open M:${String(candidate.memory_id||"").slice(0,8)} ↗`);
          open.type="button";open.addEventListener("click",()=>selectMemory(candidate.memory_id));head.append(el("strong","",`${humanize(candidate.kind||"memory")} · score ${Number(candidate.score||0).toFixed(3)}`),el("span","trace-memory-status",status));item.append(head,el("p","",truncate(cleanMemory(candidate.content_preview||candidate.memory_id||"Memory"),210)),el("p","trace-reason",used?.reason||candidate.reason||"No selection reason was recorded."),open);
          if(candidate.selected){
            const actions=el("div","trace-actions"),existing=feedback[candidate.memory_id]?.label||"";
            [["helpful","Helpful"],["irrelevant","Irrelevant"],["wrong","Wrong"],["outdated","Outdated"]].forEach(([label,text])=>{const button=el("button",`${["wrong","outdated"].includes(label)?"negative ":""}${existing===label?"active":""}`,existing===label?`${text} ✓`:text);button.type="button";button.disabled=existing===label||payload.review_writes_enabled!==true;button.addEventListener("click",()=>submitTraceMemoryFeedback(taskId,candidate.memory_id,label,existing,actions));actions.append(button);});
            if(existing&&payload.review_writes_enabled===true){const clear=el("button","", "Clear label");clear.type="button";clear.addEventListener("click",()=>submitTraceMemoryFeedback(taskId,candidate.memory_id,"clear",existing,actions));actions.append(clear);}
            if(payload.review_writes_enabled!==true&&!existing)actions.append(el("span","small-meta","Feedback is read-only on this host."));
            item.append(actions);
          }
          list.append(item);
        });
        body.append(list,el("div","trace-boundary","Recalled means Cortex injected the memory into the agent’s context. Attributed means distinctive answer evidence matched it after the turn. Neither proves that memory caused the answer."));
      }catch(error){body.replaceChildren();body.append(emptyState("Could not load this recall trace",error.message));}
    }

    async function submitTraceMemoryFeedback(taskId,memoryId,label,existing,actions){
      const changing=Boolean(existing)&&label!=="clear",clearing=label==="clear";
      if(clearing&&!window.confirm(`Clear the ${existing} label from this memory? Its training signal will be reversed.`))return;
      if(changing&&!window.confirm(`Change this memory from ${existing} to ${label}? Cortex will reverse the old signal before recording the new one.`))return;
      if(!existing&&["wrong","outdated"].includes(label)&&!window.confirm(`Mark this memory ${label} for this recall trace? This records audited negative feedback.`))return;
      actions.querySelectorAll("button").forEach(button=>button.disabled=true);
      try{await authRequest("/api/memory/feedback",{task_id:taskId,memory_id:memoryId,label});showToast(clearing?"Memory feedback cleared.":`${humanize(label)} memory feedback recorded.`);await selectTrace(taskId,{updateUrl:false});}catch(error){showToast(`Could not record memory feedback: ${error.message}`,true);actions.querySelectorAll("button").forEach(button=>button.disabled=false);}
    }

    async function selectMemory(id, { updateUrl = true } = {}) {
      if (updateUrl) {
        const url = new URL(window.location.href);
        url.searchParams.delete("trace");
        url.searchParams.set("memory", id);
        window.history.replaceState({}, "", url);
      }
      app.selectedTrace=null;app.selectedId=id; openDrawer("Memory detail"); const body=document.getElementById("drawer-body");body.replaceChildren();body.append(emptyState("Loading memory…","Fetching provenance and connections."));
      try{const response=await fetch(`/api/memory?id=${encodeURIComponent(id)}`,{cache:"no-store"});if(!response.ok)throw new Error(`HTTP ${response.status}`);const payload=await response.json(),memory=payload.memory,presentation=payload.presentation||{};body.replaceChildren();
        const role=memoryRole(memory),chips=el("div","chip-row");chips.append(el("span",`role-chip role-${role}`,roleLabel(role)),el("span",`status-chip ${memory.state}`,humanize(memory.state)),el("span","chip",humanize(memory.kind)),el("span","chip",humanize(memory.source_category)));body.append(chips);
        const remember=detailSection(role==="reference"?"What this evidence contains":"What the agent remembers");
        remember.append(el("h3","memory-card-title",presentation.display_title||truncate(cleanMemory(memory.content),110)));
        if(presentation.display_summary)remember.append(el("p","memory-card-summary",presentation.display_summary));
        if(role==="claim")remember.append(el("p","small-meta","This is an unconfirmed inference, not an established user fact."));
        body.append(remember);
        const applies=detailSection("When this applies"),contextDl=el("dl","detail-grid"),formatMap=value=>Object.entries(value||{}).map(([key,item])=>`${humanize(key)}=${item}`).join(" · ")||"None",formatList=value=>(Array.isArray(value)?value:[]).join(", ")||"None";
        applies.append(el("p","memory-card-summary",presentation.applies_when||"Applies everywhere"));
        [["Mode",humanize(memory.context_mode||"standalone")],["Scope",formatMap(memory.scope)],["Entities",formatList(memory.entities)],["Preconditions",formatMap(memory.preconditions)],["Systems",formatList(memory.applicable_systems)],["Versions",formatList(memory.applicable_versions)],["Valid window",`${memory.valid_from||"open"} → ${memory.valid_to||"open"}`],["Metadata complete",`${Math.round(Number(memory.metadata_completeness||0)*100)}%`]].forEach(([name,value])=>contextDl.append(el("dt","",name),el("dd","",value)));
        applies.append(contextDl);body.append(applies);
        const neighborhoodSection=detailSection("Where this memory lives"),neighborhoodRows=payload.neighborhoods||[];
        if(neighborhoodRows.length){neighborhoodRows.forEach(item=>{const card=el("div","connection");card.append(el("strong","",item.label),el("p","",item.explanation||"Assigned by the current semantic grouping rule."),el("span","",`${humanize(item.category)} · ${Math.round(Number(item.confidence||0)*100)}% assignment confidence${item.safety_class==="secret_reference_only"?" · secret references only":""}`));neighborhoodSection.append(card);});}else neighborhoodSection.append(el("p","small-meta","No semantic neighborhood has been assigned yet. The memory remains directly searchable."));body.append(neighborhoodSection);
        const retained=detailSection("Why it is retained");
        retained.append(el("p","memory-card-summary",presentation.retention_reason||"Retained by the storage quality checks."));
        const flags=presentation.readability_flags||[];
        if(flags.length){const flagRow=el("div","memory-card-flags");flags.forEach(flag=>flagRow.append(el("span","flag-chip",humanize(flag))));retained.append(flagRow);}
        const scores=el("div","score-grid");[["Confidence",memory.confidence],["Currentness",memory.currentness_confidence],["Importance",memory.importance],["Uniqueness",memory.uniqueness],["Trust",memory.trust],["Volatility",memory.volatility]].forEach(([name,value])=>scores.append(scoreBar(name,value)));retained.append(scores);
        const counters=el("div","counter-grid");[["Retrieved",memory.retrieved_count],["Selected",memory.selected_count],["Used",memory.used_count],["Helpful",Number(memory.helpful_count||0)+Number(memory.validated_count||0)],["Harmful",Number(memory.harmful_count||0)+Number(memory.false_positive_count||0)],["Success",memory.success_count],["Versions",payload.versions.length],["Links",payload.edges.length]].forEach(([name,value])=>{const c=el("div","counter");c.append(el("strong","",formatNum(value)),el("span","",name));counters.append(c);});retained.append(counters);body.append(retained);
        const provenance=detailSection("Source and evidence"),dl=el("dl","detail-grid"),origin=memory.origin_source_category||memory.source_category,approval=memory.approval_state==="operator_approved"?"Allowed into recall by operator review; not independently verified":humanize(memory.approval_state||"unreviewed");[["Original source",`${humanize(origin)} · ${humanize(memory.source_type)}`],["Review status",approval],["Stored category",humanize(memory.source_category)],["Reference",memory.source_ref||"—"],["Observed",formatDate(memory.observed_at)],["Extraction",humanize(memory.extraction_method)],["Claim",memory.subject?`${memory.subject} · ${memory.predicate} · ${memory.object_value}`:"Unstructured"],["Source context",memory.source_context||"None"],["Memory ID",memory.id],["Role decided by",`${humanize(memory.role_method||"legacy_default")}${memory.role_reviewed_at?` · reviewed ${formatDate(memory.role_reviewed_at)}`:""}`],["Protected",memory.protected?"Yes":"No"],["Dirty",memory.dirty?(memory.dirty_reason||"Yes"):"No"]].forEach(([name,value])=>dl.append(el("dt","",name),el("dd","",value)));provenance.append(dl);body.append(provenance);
        const rawSection=detailSection("Raw record");
        const raw=document.createElement("details");raw.className="raw-evidence";if(role!=="reference")raw.open=false;
        const rawSummary=document.createElement("summary");rawSummary.textContent="View raw stored content (unchanged evidence)";
        const rawPre=document.createElement("pre");rawPre.textContent=String(memory.content||"");
        raw.append(rawSummary,rawPre);rawSection.append(raw);body.append(rawSection);
        const history=detailSection("History and connections");
        history.append(el("p","small-meta",`${formatNum(payload.versions.length)} preserved version${payload.versions.length===1?"":"s"} · every change keeps the prior text.`));
        if(payload.edges.length||payload.dependencies.length){payload.edges.slice(0,12).forEach(edge=>{const peerId=edge.src_id===id?edge.dst_id:edge.src_id,item=el("div","connection"),evidenceCount=Number(edge.evidence_records||edge.evidence_count||0),legacy=edge.explainable===false||edge.evidence_type==="legacy_unattributed";item.append(el("strong","",`${humanize(edge.relation)} → ${truncate(cleanMemory(edge.peer_content||peerId),120)}`),el("p","",edge.explanation||"This legacy link predates preserved evidence."),el("span","",legacy?"Legacy link · supporting evidence was not preserved":`${humanize(edge.evidence_type)} · ${formatNum(evidenceCount)} evidence record${evidenceCount===1?"":"s"} · weight ${Number(edge.weight||0).toFixed(2)}`));item.tabIndex=0;item.addEventListener("click",()=>selectMemory(peerId));item.addEventListener("keydown",event=>{if(event.key==="Enter")selectMemory(peerId);});history.append(item);});payload.dependencies.slice(0,8).forEach(dep=>{const item=el("div","connection");item.append(el("strong","",`${humanize(dep.relation)} → ${truncate(cleanMemory(dep.evidence_content||dep.evidence_id),120)}`),el("p","","This memory was derived from the linked evidence memory."),el("span","",`Recorded dependency · weight ${Number(dep.weight||0).toFixed(2)}`));item.tabIndex=0;item.addEventListener("click",()=>selectMemory(dep.evidence_id));item.addEventListener("keydown",event=>{if(event.key==="Enter")selectMemory(dep.evidence_id);});history.append(item);});}
        else history.append(el("p","small-meta","No graph connections yet. Links appear here with their preserved evidence."));
        body.append(history);
      }catch(error){body.replaceChildren();body.append(emptyState("Could not load this memory",error.message));}
    }

    function openDrawer(title="Memory detail"){document.getElementById("drawer-title").textContent=title;document.getElementById("drawer").classList.add("open");document.getElementById("drawer").setAttribute("aria-hidden","false");document.getElementById("drawer-backdrop").classList.add("open");document.body.style.overflow="hidden";}
    function closeDrawer({ updateUrl = true } = {}){if(updateUrl){const url=new URL(window.location.href);url.searchParams.delete("memory");url.searchParams.delete("trace");window.history.replaceState({},"",url);}app.selectedId=null;app.selectedTrace=null;document.getElementById("drawer").classList.remove("open");document.getElementById("drawer").setAttribute("aria-hidden","true");document.getElementById("drawer-backdrop").classList.remove("open");document.body.style.overflow="";}

    function metricCards(id, values, allowText=false){const host=document.getElementById(id);host.replaceChildren();values.forEach(([label,value,note,icon])=>{const card=el("div","panel metric-card"),top=el("div","metric-top");top.append(el("span","metric-label",label),el("span","metric-icon",icon||"•"));card.append(top,el("div","metric-value",typeof value==="number"?formatNum(value):String(value)),el("div","metric-note",note||""));host.append(card);});}
    function detailSection(title){const section=el("section","detail-section");section.append(el("h3","",title));return section;}
    function scoreBar(name,value){const item=el("div","score-item"),label=document.createElement("label"),score=Math.max(0,Math.min(1,Number(value||0))),track=el("div","bar-track"),fill=el("div","bar-fill");label.append(el("span","",name),el("span","",score.toFixed(2)));fill.style.width=`${score*100}%`;track.append(fill);item.append(label,track);return item;}
    function scoreDots(value){const wrap=el("div","score-pair row-confidence"),n=Math.round(Number(value||0)*5);for(let i=1;i<=5;i++)wrap.append(el("i",`score-dot ${i<=n?"on":""}`));return wrap;}
    function emptyState(title,copy){const box=el("div","empty-state"),inner=el("div","");inner.append(el("strong","",title),el("span","",copy));box.append(inner);return box;}
    function showToast(message,isError=false){const toast=document.getElementById("toast");toast.textContent=message;toast.style.borderColor=isError?"rgba(241,120,132,.4)":"";toast.classList.add("show");clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>toast.classList.remove("show"),2800);}
    function scrollExplorer(){document.querySelector(".explorer-panel").scrollIntoView({behavior:"smooth",block:"start"});}

    function technicalIssueCount(a){return Number(a.orphan_fts_rows||0)+Number(a.missing_fts_rows||0)+Number(a.orphan_feature_rows||0)+Number(a.missing_feature_memories||0)+Number(a.feature_stat_mismatches||0)+Number(a.orphan_feature_stats||0)+Number(a.missing_memory_context_terms||0)+Number(a.orphan_memory_context_terms||0)+Number(a.invalid_memory_context_terms||0)+Number(a.invalid_score_rows||0)+Number(a.invalid_context_metadata||0)+Number(a.contextless_memories||0)+Number(a.invalid_memory_write_decisions||0)+Number(a.orphan_memory_write_decisions||0)+Number(a.invalid_memory_context_outcomes||0)+Number(a.orphan_memory_context_outcomes||0)+Number(a.orphan_memory_trace_events||0)+Number(a.invalid_edge_evidence||0)+Number(a.orphan_document_chunks||0)+Number(a.orphan_experiment_assignments||0)+Number(a.orphan_agent_assignments||0)+Number(a.orphan_prospective_items||0);}
    function relationFocus(){return document.getElementById("graph-relation")?.value||"";}
    function edgeDisplayColor(relation,transparent=false){const focus=relationFocus();return focus&&relation!==focus?colorWithAlpha(cssVar("--muted"),transparent?.05:.16):edgeColor(relation);}
    function colorWithAlpha(color,alpha){const hex=String(color).trim();if(/^#[0-9a-f]{3}$/i.test(hex)){const parts=hex.slice(1).split("").map(char=>parseInt(char+char,16));return`rgba(${parts.join(",")},${alpha})`;}if(/^#[0-9a-f]{6}$/i.test(hex)){const parts=[1,3,5].map(index=>parseInt(hex.slice(index,index+2),16));return`rgba(${parts.join(",")},${alpha})`;}const match=hex.match(/rgba?\(([^)]+)\)/i);if(match)return`rgba(${match[1].split(",").slice(0,3).join(",")},${alpha})`;return hex;}
    function graphClusterTargets(nodes,dimensions){const mode=app.graph.clusterMode;if(mode==="none")return new Map();const key=node=>mode==="source"?(node.source_category||node.source_type||"other"):mode==="neighborhood"?(node.primary_neighborhood||node.neighborhoods?.[0]?.label||"Ungrouped"):(node.kind||"other"),groups=unique(nodes.map(key)),targets=new Map(),radius=dimensions===3?260:220;nodes.forEach(node=>{const index=groups.indexOf(key(node)),angle=index*Math.PI*2/Math.max(1,groups.length)-Math.PI/2,x=Math.cos(angle)*radius,y=Math.sin(angle)*radius,z=dimensions===3?((index%2?1:-1)*radius*.55):0;targets.set(node.id,{x,y,z});});return targets;}
    function makeClusterForce(dimensions,strength){let nodes=[];const force=alpha=>{for(const node of nodes){node.vx+=(Number(node.clusterX||0)-Number(node.x||0))*strength*alpha;node.vy+=(Number(node.clusterY||0)-Number(node.y||0))*strength*alpha;if(dimensions===3)node.vz+=(Number(node.clusterZ||0)-Number(node.z||0))*strength*alpha;}};force.initialize=value=>{nodes=value||[];};return force;}
    function revealGraphSearchMatch(mode){const id=app.graph.searchMatchId;if(!id)return;selectGraphNode(id);const position=app.graph.positions.get(id);setTimeout(()=>{try{if(mode==="manual"&&position){app.graph.scale=Math.max(1.35,app.graph.scale);app.graph.panX=500-position.x*app.graph.scale;app.graph.panY=310-position.y*app.graph.scale;applyGraphTransform();}else if(mode==="physics"){const instance=app.graph.instances.physics,node=instance?.graphData().nodes.find(item=>item.id===id);if(node&&Number.isFinite(node.x)){instance.centerAt(node.x,node.y,650);instance.zoom(3.2,700);}}else if(mode==="constellation"){const instance=app.graph.instances.constellation,node=instance?.graphData().nodes.find(item=>item.id===id);if(node&&Number.isFinite(node.x)){const length=Math.hypot(node.x,node.y,node.z)||1,distance=150,ratio=1+distance/length;instance.cameraPosition({x:node.x*ratio,y:node.y*ratio,z:node.z*ratio},node,900);}}else if(mode==="analysis"){const instance=app.graph.instances.analysis,node=instance?.getElementById(id);if(node?.length){instance.animate({center:{eles:node},zoom:2.2},{duration:650});}}}catch{}},mode==="physics"||mode==="constellation"?850:80);}
    function percent(value,total){return total?Math.round(Number(value||0)/Number(total)*100):0;}
    function dateKey(value){const date=value instanceof Date?value:new Date(value);return Number.isNaN(date.getTime())?"unknown":date.toISOString().slice(0,10);}
    function startOfWeek(value){const date=new Date(value),day=(date.getUTCDay()+6)%7;date.setUTCDate(date.getUTCDate()-day);date.setUTCHours(0,0,0,0);return date;}
    function weekKey(value){return dateKey(startOfWeek(value));}
    function compactTime(value){const date=new Date(value);if(Number.isNaN(date.getTime()))return"—";const today=dateKey(new Date())===dateKey(date);return today?date.toLocaleTimeString(undefined,{hour:"numeric",minute:"2-digit"}):date.toLocaleDateString(undefined,{month:"short",day:"numeric"});}
    function signalScore(row){return Number(row.importance||0)*.34+Number(row.confidence||0)*.22+Number(row.currentness_confidence||0)*.16+Number(row.uniqueness||0)*.1+Math.min(1,Math.log2(1+Number(row.used_count||0))/4)*.18;}
    function connectionCount(id){return graphAllEdges().reduce((n,e)=>n+(e.src_id===id||e.dst_id===id?1:0),0);}
    function toolAttempts(tool){return Number(tool.success_count||0)+Number(tool.failure_count||0);}
    function cleanMemory(value){return String(value||"").replace(/^Vault note:\s*[^]*?\sPath:\s*[^]*?\sSection:\s*/i,"").trim()||String(value||"");}
    function sourceName(row){const ref=String(row.source_ref||row.source_type||"Unknown source").replace(/^vault:/,"");return truncate(ref,72);}
    function searchText(row){return`${row.content||""} ${row.source_category||""} ${row.source_ref||""} ${row.subject||""} ${row.kind||""}`.toLowerCase();}
    function eventGlyph(event){return({retrieved:"↗",selected:"◉",injected:"→",used:"✓",helpful:"+",harmful:"!"})[event]||"•";}
    function kindColor(kind){const key={semantic:"--series-semantic",episode:"--series-episode",procedure:"--series-procedure",preference:"--series-preference",identity:"--series-identity",decision:"--series-decision",prospective:"--series-prospective",operational:"--series-operational"}[kind];return key?cssVar(key):cssVar("--muted");}
    function stateColor(state){const key={active:"--green",cold:"--blue",archived:"--muted",quarantine:"--red",tombstoned:"--amber"}[state];return cssVar(key||"--muted");}
    function cssVar(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim()||"currentColor";}
    function edgeColor(relation){return({vault_link:cssVar("--mint"),supersedes:cssVar("--amber"),contradicts:cssVar("--red"),dependency:cssVar("--blue"),related:cssVar("--border-strong"),co_used:cssVar("--pink"),co_observed:cssVar("--muted")})[relation]||cssVar("--border-strong");}
    function nodeRadius(node){return 4.8+Math.min(8,Number(node.importance||0)*5+Math.log2(1+Number(node.used_count||node.used||0))*1.6);}
    function rendererSize(element){const rect=element?.getBoundingClientRect?.()||{width:1000,height:620};return{width:Math.max(320,Math.round(rect.width||1000)),height:Math.max(420,Math.round(rect.height||620))};}
    function escapeHtml(value){return String(value||"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);}
    function humanize(value){return String(value||"unknown").replace(/[_-]+/g," ").replace(/\b\w/g,c=>c.toUpperCase());}
    function unique(values){return[...new Set(values.filter(Boolean))];}
    function countBy(rows,key){return rows.reduce((o,r)=>{const v=r[key]||"unknown";o[v]=(o[v]||0)+1;return o;},{});}
    function sum(rows,key){return rows.reduce((n,r)=>n+Number(r[key]||0),0);}
    function truncate(value,length){value=String(value||"");return value.length>length?`${value.slice(0,length-1)}…`:value;}
    function formatNum(value){const n=Number(value);return Number.isFinite(n)?new Intl.NumberFormat(undefined,{notation:n>=10000?"compact":"standard",maximumFractionDigits:1}).format(n):String(value??"—");}
    function formatBytes(value){const n=Number(value||0);if(n<1024)return`${n} B`;if(n<1048576)return`${(n/1024).toFixed(1)} KB`;return`${(n/1048576).toFixed(1)} MB`;}
    function formatDate(value){const d=new Date(value);return Number.isNaN(d.getTime())?String(value||"—"):d.toLocaleString(undefined,{dateStyle:"medium",timeStyle:"short"});}
    function shortDate(value){const d=new Date(value);return Number.isNaN(d.getTime())?"—":d.toLocaleDateString(undefined,{month:"short",day:"numeric",year:d.getFullYear()!==new Date().getFullYear()?"numeric":undefined});}
    function shortDay(value){const d=new Date(`${value}T12:00:00`);return Number.isNaN(d.getTime())?value:d.toLocaleDateString(undefined,{month:"short",day:"numeric"});}
    function timeAgo(date){const seconds=Math.round((Date.now()-date.getTime())/1000);if(!Number.isFinite(seconds))return"unknown";if(Math.abs(seconds)<60)return"just now";const units=[[31536000,"year"],[2592000,"month"],[86400,"day"],[3600,"hour"],[60,"minute"]];for(const[u,name]of units)if(Math.abs(seconds)>=u){const v=Math.round(seconds/u);return new Intl.RelativeTimeFormat(undefined,{numeric:"auto"}).format(-v,name);}return"just now";}
    function shortId(value){return value?`${String(value).slice(0,8)}…`:"unknown";}
    function hashNumber(value){let h=2166136261;for(const c of String(value)){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}return(h>>>0)/4294967295;}
    function parseJson(value,fallback){try{return JSON.parse(value);}catch{return fallback;}}
    function el(tag,className="",text){const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=String(text);return node;}
    function svgEl(tag,attrs={}){const node=document.createElementNS(svgNS,tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,String(v)));return node;}
    function svgText(x,y,text,className,anchor="start"){const node=svgEl("text",{x,y,class:className,"text-anchor":anchor});node.textContent=String(text);return node;}

    bootAuth();
    // Poll only while the page is visible. A background tab used to request a
    // full snapshot (and re-render it) every minute for as long as it stayed
    // open; coming back to the tab refreshes immediately instead. The
    // `refreshing` class is the same in-flight flag `load()` already toggles, so
    // flicking between tabs cannot stack up overlapping snapshots.
    function pollIfVisible() {
      if (document.visibilityState !== "visible") return;
      if (document.getElementById("refresh").classList.contains("refreshing")) return;
      load(true);
    }
    setInterval(pollIfVisible, 60000);
    document.addEventListener("visibilitychange", pollIfVisible);
