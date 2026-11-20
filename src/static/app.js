(() => {
  const video = document.getElementById("webcam");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const placeholder = document.getElementById("cameraPlaceholder");
  const toggleCamBtn = document.getElementById("toggleCamBtn");
  const imageUploadInput = document.getElementById("imageUploadInput");
  const statusBanner = document.getElementById("statusBanner");

  const modelSelect = document.getElementById("modelSelect");
  const cameraSelect = document.getElementById("cameraSelect");
  const claheToggle = document.getElementById("claheToggle");
  const autoSpaceToggle = document.getElementById("autoSpaceToggle");
  const skeletonToggle = document.getElementById("skeletonToggle");
  const mirrorToggle = document.getElementById("mirrorToggle");
  const latencyHud = document.getElementById("latencyHud");
  const fpsHud = document.getElementById("fpsHud");

  const dwellRingHud = document.getElementById("dwellRingHud");
  const dwellRingProgress = document.getElementById("dwellRingProgress");
  const RING_CIRCUMFERENCE = 2 * Math.PI * 22; // r=22 -> 138.23

  const detectedBox = document.getElementById("detectedBox");
  const detectedLetter = document.getElementById("detectedLetter");
  const top3List = document.getElementById("top3List");

  const textBufferContent = document.getElementById("textBufferContent");
  const textBufferPlaceholder = document.getElementById("textBufferPlaceholder");
  const copyBtn = document.getElementById("copyBtn");
  const speakBtn = document.getElementById("speakBtn");
  const toggleCheatBtn = document.getElementById("toggleCheatBtn");
  const cheatsheetGrid = document.getElementById("cheatsheetGrid");

  const refModal = document.getElementById("refModal");
  const refModalBackdrop = document.getElementById("refModalBackdrop");
  const refModalClose = document.getElementById("refModalClose");
  const refModalBadge = document.getElementById("refModalBadge");
  const refModalTitle = document.getElementById("refModalTitle");
  const refModalImg = document.getElementById("refModalImg");
  const refModalCounter = document.getElementById("refModalCounter");
  const refPrevBtn = document.getElementById("refPrevBtn");
  const refNextBtn = document.getElementById("refNextBtn");
  const cheatTiles = Array.from(document.querySelectorAll(".cheat-tile"));
  let currentRefIdx = 0;

  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = 640;
  captureCanvas.height = 480;
  const captureCtx = captureCanvas.getContext("2d");

  const connections = window.FSL_CONFIG.handConnections;
  const threshold = window.FSL_CONFIG.confidenceThreshold;

  let stream = null;
  let running = false;
  let inFlight = false;
  let frameCount = 0;
  let lastFpsTime = performance.now();
  let currentBufferText = "";

  async function populateCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((d) => d.kind === "videoinput");
      if (videoDevices.length === 0) return;

      const currentVal = cameraSelect.value;
      cameraSelect.innerHTML = "";
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement("option");
        opt.value = dev.deviceId;
        opt.textContent = dev.label || `Camera ${idx + 1}`;
        cameraSelect.appendChild(opt);
      });
      if (currentVal) cameraSelect.value = currentVal;
    } catch (err) {
      console.warn("Could not enumerate cameras:", err);
    }
  }

  function setStatus(text, cls) {
    statusBanner.textContent = text;
    statusBanner.className = `status-pill ${cls}`;
  }

  function setBufferText(text) {
    currentBufferText = text || "";
    textBufferContent.textContent = currentBufferText;
    textBufferPlaceholder.style.display = currentBufferText.length > 0 ? "none" : "inline";
  }

  function drawSkeleton(landmarks) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!skeletonToggle.checked || !landmarks || landmarks.length === 0) return;

    const w = overlay.width;
    const h = overlay.height;

    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#38bdf8";

    for (const [a, b] of connections) {
      const p1 = landmarks[a];
      const p2 = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(p1[0] * w, p1[1] * h);
      ctx.lineTo(p2[0] * w, p2[1] * h);
      ctx.stroke();
    }

    landmarks.forEach((pt, idx) => {
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, idx === 0 ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = idx === 0 ? "#38bdf8" : "#ffffff";
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#38bdf8";
      ctx.stroke();
    });
  }

  function renderTop3(top3) {
    const rows = top3List.querySelectorAll(".top3-row");
    for (let i = 0; i < 3; i++) {
      const item = top3[i];
      const charEl = rows[i].querySelector(".top3-char");
      const fillEl = rows[i].querySelector(".bar-fill");
      const pctEl = rows[i].querySelector(".top3-pct");

      if (item) {
        const pct = Math.round(item.prob * 100);
        charEl.textContent = item.label;
        fillEl.style.width = `${pct}%`;
        pctEl.textContent = `${pct}%`;
      } else {
        charEl.textContent = "-";
        fillEl.style.width = "0%";
        pctEl.textContent = "0%";
      }
    }
  }

  function updateDwellRing(progress) {
    if (progress > 0.01) {
      dwellRingHud.classList.add("visible");
      const offset = RING_CIRCUMFERENCE * (1 - Math.min(1, progress));
      dwellRingProgress.style.strokeDashoffset = `${offset}`;
    } else {
      dwellRingHud.classList.remove("visible");
      dwellRingProgress.style.strokeDashoffset = `${RING_CIRCUMFERENCE}`;
    }
  }

  function highlightCheatTile(letter, aboveThreshold) {
    document.querySelectorAll(".cheat-tile.active").forEach((el) => el.classList.remove("active"));
    if (letter && aboveThreshold) {
      const tile = document.querySelector(`.cheat-tile[data-letter="${letter}"]`);
      if (tile) tile.classList.add("active");
    }
  }

  function flashConfirmed() {
    detectedBox.classList.add("confirmed-flash");
    setTimeout(() => detectedBox.classList.remove("confirmed-flash"), 350);
  }

  function updateMirrorStyle() {
    video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";
  }

  async function processLoop() {
    if (!running) return;
    if (!inFlight && video.readyState >= 2) {
      inFlight = true;
      try {
        captureCtx.save();
        if (mirrorToggle.checked) {
          captureCtx.translate(captureCanvas.width, 0);
          captureCtx.scale(-1, 1);
        }
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        captureCtx.restore();

        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.72);
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image: dataUrl,
            model: modelSelect.value,
            clahe: claheToggle.checked,
            auto_space: autoSpaceToggle.checked,
          }),
        });
        const data = await res.json();

        if (!running) return;

        if (data.hand_detected) {
          setStatus("Hand Tracked", "status-ok");
        } else {
          setStatus("No hand detected — adjust position or lighting", "status-warn");
        }

        drawSkeleton(data.landmarks);
        renderTop3(data.top3 || []);

        const activeChar = data.smoothed_label || data.raw_label || "-";
        const aboveThreshold = (data.raw_confidence || 0) * 100 >= threshold;

        detectedLetter.textContent = activeChar;
        detectedBox.classList.toggle("above-threshold", Boolean(data.hand_detected && aboveThreshold));
        highlightCheatTile(data.smoothed_label || data.raw_label, aboveThreshold);

        if (data.just_confirmed) {
          flashConfirmed();
        }

        const dwellRatio = data.dwell_progress || 0;
        updateDwellRing(dwellRatio);

        if (data.text_buffer !== undefined) {
          setBufferText(data.text_buffer);
        }
        latencyHud.textContent = `Latency: ${data.latency_ms} ms`;

        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
          fpsHud.textContent = `FPS: ${frameCount}`;
          frameCount = 0;
          lastFpsTime = now;
        }
      } catch (err) {
        console.error(err);
      } finally {
        inFlight = false;
      }
    }
    requestAnimationFrame(processLoop);
  }

  async function startCamera() {
    try {
      const videoConstraints = { width: 640, height: 480 };
      if (cameraSelect.value) {
        videoConstraints.deviceId = { exact: cameraSelect.value };
      }
      stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
      await populateCameras();
      video.srcObject = stream;
      updateMirrorStyle();
      placeholder.style.display = "none";
      running = true;
      toggleCamBtn.textContent = "Stop Camera";
      setStatus("Camera Active", "status-ok");
      requestAnimationFrame(processLoop);
    } catch (err) {
      setStatus("Camera permission denied or unavailable", "status-warn");
      alert("Could not access webcam: " + err.message);
    }
  }

  function stopCamera() {
    running = false;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    updateDwellRing(0);
    detectedBox.classList.remove("above-threshold");
    detectedLetter.textContent = "-";
    renderTop3([]);
    highlightCheatTile(null, false);
    latencyHud.textContent = "Latency: -- ms";
    fpsHud.textContent = "FPS: --";
    placeholder.style.display = "flex";
    toggleCamBtn.textContent = "Start Camera";
    setStatus("Camera Offline", "status-idle");
  }

  toggleCamBtn.addEventListener("click", () => {
    if (running) stopCamera();
    else startCamera();
  });

  imageUploadInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    if (running) stopCamera();

    const img = new Image();
    img.onload = async () => {
      placeholder.style.display = "none";
      captureCtx.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
      const scale = Math.min(captureCanvas.width / img.width, captureCanvas.height / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = (captureCanvas.width - dw) / 2;
      const dy = (captureCanvas.height - dh) / 2;
      captureCtx.fillStyle = "#090d12";
      captureCtx.fillRect(0, 0, captureCanvas.width, captureCanvas.height);
      captureCtx.drawImage(img, dx, dy, dw, dh);

      // Draw uploaded image onto overlay, then draw landmarks on top after response
      const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.92);
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: dataUrl,
          model: modelSelect.value,
          clahe: claheToggle.checked,
          auto_space: false,
        }),
      });
      const data = await res.json();

      ctx.clearRect(0, 0, overlay.width, overlay.height);
      ctx.drawImage(captureCanvas, 0, 0, overlay.width, overlay.height);
      if (skeletonToggle.checked && data.landmarks && data.landmarks.length > 0) {
        const w = overlay.width;
        const h = overlay.height;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#38bdf8";
        for (const [a, b] of connections) {
          ctx.beginPath();
          ctx.moveTo(data.landmarks[a][0] * w, data.landmarks[a][1] * h);
          ctx.lineTo(data.landmarks[b][0] * w, data.landmarks[b][1] * h);
          ctx.stroke();
        }
        data.landmarks.forEach((pt, idx) => {
          ctx.beginPath();
          ctx.arc(pt[0] * w, pt[1] * h, idx === 0 ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fillStyle = idx === 0 ? "#38bdf8" : "#ffffff";
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = "#38bdf8";
          ctx.stroke();
        });
      }

      if (data.hand_detected) setStatus("Image Evaluated", "status-ok");
      else setStatus("No hand detected in uploaded image", "status-warn");

      renderTop3(data.top3 || []);
      const activeChar = data.raw_label || "-";
      const aboveThreshold = (data.raw_confidence || 0) * 100 >= threshold;
      detectedLetter.textContent = activeChar;
      detectedBox.classList.toggle("above-threshold", Boolean(data.hand_detected && aboveThreshold));
      highlightCheatTile(data.raw_label, aboveThreshold);
      latencyHud.textContent = `Latency: ${data.latency_ms} ms`;
      fpsHud.textContent = "Mode: Image";
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
    imageUploadInput.value = "";
  });

  cameraSelect.addEventListener("change", () => {
    if (running) {
      stopCamera();
      startCamera();
    }
  });

  mirrorToggle.addEventListener("change", updateMirrorStyle);
  skeletonToggle.addEventListener("change", () => {
    if (!skeletonToggle.checked) ctx.clearRect(0, 0, overlay.width, overlay.height);
  });

  async function bufferAction(action) {
    const res = await fetch(`/api/buffer/${action}`, { method: "POST" });
    const data = await res.json();
    if (data.text_buffer !== undefined) setBufferText(data.text_buffer);
  }

  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => bufferAction(btn.dataset.action));
  });

  copyBtn.addEventListener("click", async () => {
    await navigator.clipboard.writeText(currentBufferText);
    copyBtn.textContent = "Copied";
    setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
  });

  speakBtn.addEventListener("click", () => {
    const txt = currentBufferText.trim();
    if (!txt || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(txt);
    utter.rate = 0.95;
    window.speechSynthesis.speak(utter);
  });

  toggleCheatBtn.addEventListener("click", () => {
    const hidden = cheatsheetGrid.classList.toggle("hidden");
    toggleCheatBtn.textContent = hidden ? "Show Reference" : "Hide Reference";
  });

  function openRefModal(index) {
    currentRefIdx = (index + cheatTiles.length) % cheatTiles.length;
    const tile = cheatTiles[currentRefIdx];
    const letter = tile.dataset.letter;
    const imgEl = tile.querySelector("img");
    refModalBadge.textContent = letter;
    refModalTitle.textContent = `FSL Sign Reference — "${letter}"`;
    refModalImg.src = imgEl.src;
    refModalImg.alt = `FSL Sign ${letter}`;
    refModalCounter.textContent = `${currentRefIdx + 1} / ${cheatTiles.length}`;
    refModal.classList.remove("hidden");
  }

  function closeRefModal() {
    refModal.classList.add("hidden");
  }

  cheatTiles.forEach((tile, idx) => {
    tile.addEventListener("click", () => openRefModal(idx));
  });
  refModalClose.addEventListener("click", closeRefModal);
  refModalBackdrop.addEventListener("click", closeRefModal);
  refPrevBtn.addEventListener("click", () => openRefModal(currentRefIdx - 1));
  refNextBtn.addEventListener("click", () => openRefModal(currentRefIdx + 1));

  window.addEventListener("keydown", (e) => {
    if (!refModal.classList.contains("hidden")) {
      if (e.code === "Escape") closeRefModal();
      else if (e.code === "ArrowLeft") openRefModal(currentRefIdx - 1);
      else if (e.code === "ArrowRight") openRefModal(currentRefIdx + 1);
      return;
    }
    if (e.code === "Space") {
      e.preventDefault();
      bufferAction("space");
    } else if (e.code === "Backspace") {
      e.preventDefault();
      bufferAction("backspace");
    } else if (e.code === "Escape") {
      bufferAction("clear");
    }
  });

  populateCameras();
})();
