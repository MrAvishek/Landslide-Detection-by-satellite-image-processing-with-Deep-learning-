"use strict";

const PREDICTION_ENDPOINT = "/api/v1/predict-upload";
const MAX_FILE_SIZE_MB = 100;

const $ = (id) => document.getElementById(id);

/* =====================================================
   DOM ELEMENT REFERENCES (MATCHED TO LATEST HTML)
===================================================== */

// Navigation & Topbar
const navLinks = document.querySelectorAll(".sidebar-nav .nav-item");
const breadcrumbCurrent = document.querySelector(".breadcrumb strong");
const helpButton = $("help-button");
const closeHelpButton = $("close-help-button");
const helpDialog = $("help-dialog");

// Form & Satellite File Upload
const form = $("prediction-form");
const fileInput = $("image-file");
const chooseFileButton = $("choose-file-button");
const uploadArea = $("upload-area");
const selectedFileRow = $("selected-file-row");
const selectedFileName = $("selected-file-name");
const detailFileSize = $("detail-file-size");
const removeImageButton = $("remove-image-file");
const imageFileError = $("image-file-error");

// Optional Ground-Truth Mask Upload
const maskInput = $("mask-file");
const chooseMaskButton = $("choose-mask-button");
const maskUploadArea = $("mask-upload-area");
const selectedMaskRow = $("selected-mask-row");
const selectedMaskName = $("selected-mask-name");
const selectedMaskSize = $("selected-mask-size");
const removeMaskButton = $("remove-mask-file");
const groundTruthStatus = $("ground-truth-status");
const maskFileError = $("mask-file-error");

// Threshold & Form Actions
const thresholdInput = $("threshold");
const thresholdValue = $("threshold-value");
const runButton = $("run-analysis");
const runButtonText = $("run-button-text");
const resetButton = $("reset-analysis");
const exportButton = $("export-results-button");

// Canvases, Panels & Errors
const maskCanvas = $("mask-canvas");
const groundTruthCanvas = $("ground-truth-canvas");
const evaluationPanel = $("evaluation-panel");
const analysisError = $("analysis-error");

// Toast Notification
const toast = $("toast");
const toastMessage = $("toast-message");
const toastClose = $("toast-close");

// Runtime State
let toastTimer = null;
let latestPrediction = null;
let lastRunDurationMs = null;

/* =====================================================
   UTILITIES & TOAST NOTIFICATIONS
===================================================== */

function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = String(value);
}

function setBarWidth(id, percentage) {
    const bar = $(id);
    if (!bar) return;
    const clamped = Math.max(0, Math.min(100, Number(percentage) || 0));
    bar.style.width = `${clamped.toFixed(2)}%`;
}

function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return "Unknown";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatInteger(value) {
    const num = Number(value);
    return Number.isFinite(num) ? Math.round(num).toLocaleString() : "—";
}

function formatPercentage(value) {
    const num = Number(value);
    return Number.isFinite(num) ? `${num.toFixed(2)}%` : "—";
}

function showToast(message, type = "info") {
    if (!toast || !toastMessage) return;

    // Preserves inner structure (#toast-icon, #toast-message, #toast-close)
    toastMessage.textContent = message;
    toast.hidden = false;
    toast.classList.remove("show", "toast-error", "toast-success", "toast-info");
    toast.classList.add("show", `toast-${type}`);

    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
        toast.classList.remove("show");
        toast.hidden = true;
    }, 4500);
}

toastClose?.addEventListener("click", () => {
    clearTimeout(toastTimer);
    if (toast) {
        toast.classList.remove("show");
        toast.hidden = true;
    }
});

function showFieldError(message, isMask = false) {
    const target = isMask ? maskFileError : imageFileError;
    if (target) {
        target.textContent = message;
        target.hidden = false;
    }
    showToast(message, "error");
}

function clearFieldError(isMask = false) {
    const target = isMask ? maskFileError : imageFileError;
    if (target) {
        target.textContent = "";
        target.hidden = true;
    }
}

function setAnalysisError(message = "") {
    if (!analysisError) return;
    if (!message) {
        analysisError.textContent = "";
        analysisError.hidden = true;
    } else {
        analysisError.textContent = message;
        analysisError.hidden = false;
    }
}

/* =====================================================
   SIDEBAR NAVIGATION & HELP DIALOG
===================================================== */

navLinks.forEach((link) => {
    link.addEventListener("click", () => {
        navLinks.forEach((item) => {
            item.classList.remove("active");
            item.removeAttribute("aria-current");
        });
        link.classList.add("active");
        link.setAttribute("aria-current", "page");

        const label = link.querySelector("span")?.textContent;
        if (breadcrumbCurrent && label) {
            breadcrumbCurrent.textContent = label;
        }
    });
});

helpButton?.addEventListener("click", () => {
    if (typeof helpDialog?.showModal === "function") {
        helpDialog.showModal();
    }
});

closeHelpButton?.addEventListener("click", () => {
    helpDialog?.close();
});

helpDialog?.addEventListener("click", (event) => {
    if (event.target === helpDialog) {
        helpDialog.close();
    }
});

/* =====================================================
   API HEALTH CHECK
===================================================== */

async function checkApiStatus() {
    const topbarStatus = $("api-status");
    const topbarDot = $("api-status-dot");
    const sidebarStatus = $("sidebar-api-label");
    const sidebarDot = $("sidebar-api-dot");

    for (const endpoint of ["/health", "/api/v1/health"]) {
        try {
            const response = await fetch(endpoint, {
                method: "GET",
                headers: { Accept: "application/json" },
                cache: "no-store"
            });

            if (response.ok) {
                if (topbarStatus) topbarStatus.textContent = "API Online";
                if (sidebarStatus) sidebarStatus.textContent = "Connected";
                [topbarDot, sidebarDot].forEach((dot) => {
                    dot?.classList.remove("api-offline", "status-dot-muted");
                    dot?.classList.add("api-online");
                });
                return;
            }
        } catch {
            // Try next health endpoint
        }
    }

    if (topbarStatus) topbarStatus.textContent = "API Offline";
    if (sidebarStatus) sidebarStatus.textContent = "Unavailable";
    [topbarDot, sidebarDot].forEach((dot) => {
        dot?.classList.remove("api-online");
        dot?.classList.add("api-offline");
    });
}

/* =====================================================
   FILE VALIDATION & SELECTION
===================================================== */

function getSelectedFile() {
    return fileInput?.files?.[0] ?? null;
}

function getSelectedMask() {
    return maskInput?.files?.[0] ?? null;
}

function validateHdf5File(file, label = "Satellite") {
    if (!file) return `Select a ${label.toLowerCase()} HDF5 file first.`;

    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!["h5", "hdf5"].includes(extension)) {
        return `Unsupported ${label.toLowerCase()} file. Choose an .h5 or .hdf5 file.`;
    }

    if (file.size === 0) {
        return `The selected ${label.toLowerCase()} file is empty.`;
    }

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        return `${label} file exceeds the ${MAX_FILE_SIZE_MB} MB limit.`;
    }

    return null;
}

function syncEvaluationModeIndicators() {
    const maskFile = getSelectedMask();

    if (maskFile) {
        setText("evaluation-mode-summary", "Reference validation");
        setText("evaluation-mode-hint", maskFile.name);
        setText("workflow-evaluation-tag", "Mask ready");
        setText("configuration-mask-status", maskFile.name);
        setText("result-ground-truth-detail", maskFile.name);
        setText("prediction-only-badge", "Reference mask supplied");

        if (groundTruthStatus) {
            groundTruthStatus.innerHTML =
                `<span class="status-dot api-online"></span>` +
                `<span>Reference mask ready (${maskFile.name}). Overlap and pixel classification metrics will be computed.</span>`;
        }
    } else {
        setText("evaluation-mode-summary", "Prediction only");
        setText("evaluation-mode-hint", "Reference mask not supplied");
        setText("workflow-evaluation-tag", "When mask supplied");
        setText("configuration-mask-status", "Not supplied");
        setText("result-ground-truth-detail", "Not supplied");
        setText("prediction-only-badge", "Prediction only");

        if (groundTruthStatus) {
            groundTruthStatus.innerHTML =
                `<span class="status-dot status-dot-muted"></span>` +
                `<span>No reference mask supplied. Evaluation metrics will be unavailable.</span>`;
        }
    }
}

function handleSatelliteFileChange() {
    clearFieldError(false);
    setAnalysisError("");

    const file = getSelectedFile();

    if (!file) {
        if (selectedFileRow) selectedFileRow.hidden = true;
        if (runButton) runButton.disabled = true;
        uploadArea?.classList.remove("has-file");

        setText("selected-file-name", "No file selected");
        setText("detail-file-size", "File size will appear here");
        setText("file-metric", "No file");
        setText("detail-file-name", "Select an HDF5 file to begin");
        setText("configuration-input-shape", "Not loaded");
        return;
    }

    const error = validateHdf5File(file, "Satellite");
    if (error) {
        fileInput.value = "";
        if (selectedFileRow) selectedFileRow.hidden = true;
        if (runButton) runButton.disabled = true;
        uploadArea?.classList.remove("has-file");
        showFieldError(error, false);
        return;
    }

    if (selectedFileRow) selectedFileRow.hidden = false;
    if (runButton) runButton.disabled = false;
    uploadArea?.classList.add("has-file");

    setText("selected-file-name", file.name);
    setText("detail-file-size", formatFileSize(file.size));
    setText("file-metric", file.name);
    setText("detail-file-name", `${formatFileSize(file.size)} · Ready for analysis`);
    setText("configuration-input-shape", "128 × 128 × 26");

    if ($("file-metric")) $("file-metric").title = file.name;

    // Clear previous run output so old masks aren't mistaken for the new file
    resetResultsDisplay();
    setResultStatePill("Ready for analysis", "idle");
    showToast("Satellite .h5 file selected.", "success");
}

function handleMaskFileChange() {
    clearFieldError(true);
    setAnalysisError("");

    const maskFile = getSelectedMask();

    if (!maskFile) {
        if (selectedMaskRow) selectedMaskRow.hidden = true;
        maskUploadArea?.classList.remove("has-file");
        setText("selected-mask-name", "No mask selected");
        setText("selected-mask-size", "Reference mask ready for evaluation");
        syncEvaluationModeIndicators();
        return;
    }

    const error = validateHdf5File(maskFile, "Ground-truth mask");
    if (error) {
        maskInput.value = "";
        if (selectedMaskRow) selectedMaskRow.hidden = true;
        maskUploadArea?.classList.remove("has-file");
        syncEvaluationModeIndicators();
        showFieldError(error, true);
        return;
    }

    if (selectedMaskRow) selectedMaskRow.hidden = false;
    maskUploadArea?.classList.add("has-file");

    setText("selected-mask-name", maskFile.name);
    setText("selected-mask-size", `${formatFileSize(maskFile.size)} · Reference mask ready`);
    syncEvaluationModeIndicators();
    showToast("Ground-truth .h5 mask selected.", "success");
}

// Event Listeners for File Pickers & Remove Buttons
chooseFileButton?.addEventListener("click", () => fileInput?.click());
fileInput?.addEventListener("change", handleSatelliteFileChange);

removeImageButton?.addEventListener("click", () => {
    if (fileInput) fileInput.value = "";
    handleSatelliteFileChange();
    resetResultsDisplay();
});

chooseMaskButton?.addEventListener("click", () => maskInput?.click());
maskInput?.addEventListener("change", handleMaskFileChange);

removeMaskButton?.addEventListener("click", () => {
    if (maskInput) maskInput.value = "";
    handleMaskFileChange();
    if (evaluationPanel) evaluationPanel.hidden = true;
    resetCanvas(groundTruthCanvas);
    setPlaceholder("ground-truth-canvas", "ground-truth-placeholder", true);
    setText("ground-truth-visualization-note", "Reference data is optional");
    setText("ground-truth-shape", "Shape: —");
});

/* =====================================================
   DRAG AND DROP (SATELLITE & MASK UPLOAD ZONES)
===================================================== */

function attachDragAndDrop(zone, inputEl, isMask = false) {
    if (!zone || !inputEl) return;

    zone.addEventListener("dragover", (event) => {
        event.preventDefault();
        zone.classList.add("drag-over");
    });

    zone.addEventListener("dragleave", (event) => {
        if (!zone.contains(event.relatedTarget)) {
            zone.classList.remove("drag-over");
        }
    });

    zone.addEventListener("drop", (event) => {
        event.preventDefault();
        zone.classList.remove("drag-over");

        const file = event.dataTransfer?.files?.[0];
        if (!file) return;

        const error = validateHdf5File(
            file,
            isMask ? "Ground-truth mask" : "Satellite"
        );
        if (error) {
            showFieldError(error, isMask);
            return;
        }

        try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            inputEl.files = transfer.files;
            inputEl.dispatchEvent(new Event("change", { bubbles: true }));
        } catch (err) {
            console.error(err);
            showFieldError("Could not attach dropped file. Use the Choose button.", isMask);
        }
    });
}

attachDragAndDrop(uploadArea, fileInput, false);
attachDragAndDrop(maskUploadArea, maskInput, true);

/* =====================================================
   THRESHOLD CONTROL
===================================================== */

function handleThresholdInput() {
    if (!thresholdInput) return;

    const value = Number(thresholdInput.value);
    if (!Number.isFinite(value)) return;

    const formatted = value.toFixed(2);
    setText("threshold-value", formatted);
    setText("result-threshold", `Threshold: ${formatted}`);
    setText("configuration-threshold", formatted);
    setText("result-threshold-detail", formatted);

    // Live re-render of prediction & evaluation when sliding threshold
    if (latestPrediction) {
        try {
            displayPrediction(latestPrediction, value, getSelectedFile());
        } catch (err) {
            console.error("Failed to re-render at new threshold:", err);
        }
    }
}

thresholdInput?.addEventListener("input", handleThresholdInput);

/* =====================================================
   CANVAS HELPERS
===================================================== */

function resetCanvas(canvas) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = "none";
}

function setPlaceholder(canvasId, placeholderId, showPlaceholder) {
    const canvas = $(canvasId);
    const placeholder = $(placeholderId);
    if (canvas) canvas.style.display = showPlaceholder ? "none" : "block";
    if (placeholder) placeholder.style.display = showPlaceholder ? "flex" : "none";
}

function normalizeMaskArray(mask, label = "Mask") {
    if (!Array.isArray(mask) || mask.length === 0) {
        throw new Error(`${label} array is missing or empty.`);
    }

    const rows = mask.map((row) => {
        if (!Array.isArray(row) && !ArrayBuffer.isView(row)) {
            throw new Error(`${label} contains an invalid row.`);
        }
        return Array.from(row).map((pixel) => {
            if (Array.isArray(pixel) || ArrayBuffer.isView(pixel)) {
                return Number(pixel[0]);
            }
            return Number(pixel);
        });
    });

    const width = rows[0].length;
    if (!width || rows.some((r) => r.length !== width)) {
        throw new Error(`${label} has inconsistent row dimensions.`);
    }

    return rows;
}

function drawMaskOnCanvas(canvas, mask, options = {}) {
    if (!canvas) throw new Error("Canvas element not found.");

    const {
        threshold = 0.5,
        binary = false,
        positiveRgb = [255, 126, 62],
        backgroundRgb = [19, 29, 46]
    } = options;

    const rows = normalizeMaskArray(mask);
    const height = rows.length;
    const width = rows[0].length;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not initialize 2D canvas context.");

    const imageData = ctx.createImageData(width, height);
    const binaryRows = [];
    let positivePixels = 0;

    for (let y = 0; y < height; y++) {
        const binaryRow = [];
        for (let x = 0; x < width; x++) {
            const val = rows[y][x];
            const isPositive = binary ? val > 0 : val >= threshold;

            binaryRow.push(isPositive ? 1 : 0);
            if (isPositive) positivePixels++;

            const idx = (y * width + x) * 4;
            const color = isPositive ? positiveRgb : backgroundRgb;

            imageData.data[idx] = color[0];
            imageData.data[idx + 1] = color[1];
            imageData.data[idx + 2] = color[2];
            imageData.data[idx + 3] = 255;
        }
        binaryRows.push(binaryRow);
    }

    ctx.putImageData(imageData, 0, 0);
    canvas.style.display = "block";

    return {
        height,
        width,
        positivePixels,
        totalPixels: height * width,
        rows,
        binaryRows
    };
}

/* =====================================================
   EVALUATION METRICS & CONFUSION MATRIX
===================================================== */

function calculateConfusionAndScores(predBinaryRows, truthBinaryRows) {
    const height = predBinaryRows.length;
    const width = predBinaryRows[0].length;

    if (
        truthBinaryRows.length !== height ||
        truthBinaryRows[0].length !== width
    ) {
        return null;
    }

    let tp = 0;
    let fp = 0;
    let tn = 0;
    let fn = 0;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const predPos = predBinaryRows[y][x] === 1;
            const truthPos = truthBinaryRows[y][x] === 1;

            if (predPos && truthPos) tp++;
            else if (predPos && !truthPos) fp++;
            else if (!predPos && truthPos) fn++;
            else tn++;
        }
    }

    const total = tp + fp + tn + fn;
    const dice = 2 * tp + fp + fn > 0 ? (2 * tp) / (2 * tp + fp + fn) : 1.0;
    const iou = tp + fp + fn > 0 ? tp / (tp + fp + fn) : 1.0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0.0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0.0;
    const accuracy = total > 0 ? (tp + tn) / total : 0.0;
    const specificity = tn + fp > 0 ? tn / (tn + fp) : 0.0;

    const gtPositiveShare = total > 0 ? ((tp + fn) / total) * 100 : 0;
    const predPositiveShare = total > 0 ? ((tp + fp) / total) * 100 : 0;

    return {
        tp,
        fp,
        tn,
        fn,
        dice,
        iou,
        precision,
        recall,
        accuracy,
        specificity,
        gtPositiveShare,
        predPositiveShare
    };
}

function updateEvaluationSection(result, predStats, truthStats) {
    const backendMetrics =
        result.evaluation_metrics ??
        result.metrics ??
        result.evaluation ??
        null;

    // Dynamically calculate from binary arrays if both masks are present
    const clientMetrics =
        predStats && truthStats
            ? calculateConfusionAndScores(
                  predStats.binaryRows,
                  truthStats.binaryRows
              )
            : null;

    if (!clientMetrics && (!backendMetrics || typeof backendMetrics !== "object")) {
        if (evaluationPanel) evaluationPanel.hidden = true;
        return;
    }

    if (evaluationPanel) evaluationPanel.hidden = false;

    const resolveScore = (clientValue, keys) => {
        if (clientMetrics && Number.isFinite(clientValue)) return clientValue;
        if (!backendMetrics) return NaN;
        const foundKey = keys.find((k) =>
            Object.prototype.hasOwnProperty.call(backendMetrics, k)
        );
        if (!foundKey) return NaN;
        const raw = Number(backendMetrics[foundKey]);
        return raw > 1 ? raw / 100 : raw;
    };

    const dice = resolveScore(clientMetrics?.dice, ["dice", "dice_score", "Dice", "f1"]);
    const iou = resolveScore(clientMetrics?.iou, ["iou", "IoU", "intersection_over_union"]);
    const precision = resolveScore(clientMetrics?.precision, ["precision"]);
    const recall = resolveScore(clientMetrics?.recall, ["recall", "sensitivity"]);
    const accuracy = resolveScore(clientMetrics?.accuracy, ["accuracy", "pixel_accuracy"]);
    const specificity = resolveScore(clientMetrics?.specificity, ["specificity"]);

    const setScoreCard = (valueId, barId, scoreRatio) => {
        if (Number.isFinite(scoreRatio)) {
            const pct = scoreRatio * 100;
            setText(valueId, `${pct.toFixed(2)}%`);
            setBarWidth(barId, pct);
        } else {
            setText(valueId, "—");
            setBarWidth(barId, 0);
        }
    };

    setScoreCard("dice-score", "dice-score-bar", dice);
    setScoreCard("iou-score", "iou-score-bar", iou);
    setScoreCard("precision-score", "precision-score-bar", precision);
    setScoreCard("recall-score", "recall-score-bar", recall);
    setScoreCard("accuracy-score", "accuracy-score-bar", accuracy);
    setScoreCard("specificity-score", "specificity-score-bar", specificity);

    // Confusion Matrix Counts
    const tp = clientMetrics?.tp ?? backendMetrics?.tp ?? backendMetrics?.true_positives;
    const fp = clientMetrics?.fp ?? backendMetrics?.fp ?? backendMetrics?.false_positives;
    const tn = clientMetrics?.tn ?? backendMetrics?.tn ?? backendMetrics?.true_negatives;
    const fn = clientMetrics?.fn ?? backendMetrics?.fn ?? backendMetrics?.false_negatives;

    setText("true-positive-count", formatInteger(tp));
    setText("false-positive-count", formatInteger(fp));
    setText("true-negative-count", formatInteger(tn));
    setText("false-negative-count", formatInteger(fn));

    // Class Distribution Chart
    const predShare =
        clientMetrics?.predPositiveShare ??
        (predStats ? (predStats.positivePixels / predStats.totalPixels) * 100 : NaN);

    const gtShare =
        clientMetrics?.gtPositiveShare ??
        (truthStats
            ? (truthStats.positivePixels / truthStats.totalPixels) * 100
            : Number(backendMetrics?.ground_truth_percentage ?? NaN));

    setText("ground-truth-percentage", formatPercentage(gtShare));
    setBarWidth("ground-truth-distribution-bar", gtShare);

    setText("prediction-percentage-eval", formatPercentage(predShare));
    setBarWidth("prediction-distribution-bar", predShare);

    const distributionInsight = $("distribution-insight");
    if (distributionInsight && Number.isFinite(predShare) && Number.isFinite(gtShare)) {
        const delta = predShare - gtShare;
        let note = "";
        if (Math.abs(delta) < 0.25) {
            note = "Predicted positive pixel share closely aligns with the ground-truth distribution.";
        } else if (delta > 0) {
            note = `Prediction classifies +${delta.toFixed(2)}% more landslide area than ground truth.`;
        } else {
            note = `Prediction classifies ${delta.toFixed(2)}% less landslide area than ground truth.`;
        }
        distributionInsight.innerHTML = `<span class="insight-dot"></span><span>${note}</span>`;
    }
}

/* =====================================================
   RENDER PREDICTION & INSIGHTS
===================================================== */

function setResultStatePill(label, state = "idle") {
    const pill = $("result-status");
    if (!pill) return;
    pill.className = `result-state-pill state-${state}`;
    pill.innerHTML = `<span class="status-dot"></span><span>${label}</span>`;
}

function displayPrediction(result, threshold, file) {
    const rawMask =
        result.mask ?? result.probability_mask ?? result.prediction;

    if (!rawMask) {
        throw new Error(
            "API response does not contain 'mask' or 'probability_mask'."
        );
    }

    const predStats = drawMaskOnCanvas(maskCanvas, rawMask, {
        threshold,
        binary: false,
        positiveRgb: [255, 126, 62], // High-contrast orange
        backgroundRgb: [19, 29, 46]
    });

    setPlaceholder("mask-canvas", "mask-placeholder", false);

    const positivePixels = predStats.positivePixels;
    const totalPixels = predStats.totalPixels;
    const positivePercentage = totalPixels > 0 ? (positivePixels / totalPixels) * 100 : 0;
    const backgroundPercentage = Math.max(0, 100 - positivePercentage);

    const dimensions = Array.isArray(result.mask_shape)
        ? result.mask_shape
        : [predStats.height, predStats.width];
    const shapeLabel = `${dimensions[0]} × ${dimensions[1]}`;

    // 1. Top 4 Prediction Metric Cards
    setText("positive-metric", formatInteger(positivePixels));
    setText("positive-metric-caption", `Out of ${formatInteger(totalPixels)} grid pixels`);
    setText("percentage-metric", formatPercentage(positivePercentage));
    setBarWidth("predicted-share-bar", positivePercentage);

    if (file) {
        setText("file-metric", file.name);
        setText("detail-file-name", `${formatFileSize(file.size)} · Analyzed`);
    }

    setText("result-shape", shapeLabel);
    setText("result-mask-shape", `${formatInteger(totalPixels)} total pixels`);

    // 2. Prediction Canvas Footer
    setText(
        "prediction-visualization-note",
        `${formatInteger(positivePixels)} positive pixels (${positivePercentage.toFixed(2)}%)`
    );
    setText("result-threshold", `Threshold: ${threshold.toFixed(2)}`);

    // 3. Spatial Signal Overview (Prediction Insights Section)
    setBarWidth("composition-positive", positivePercentage);
    setBarWidth("composition-background", backgroundPercentage);
    setText("composition-positive-label", formatPercentage(positivePercentage));
    setText("composition-background-label", formatPercentage(backgroundPercentage));

    setText("coverage-value", formatPercentage(positivePercentage));
    setText(
        "coverage-description",
        positivePixels === 0
            ? `No pixels exceeded the ${threshold.toFixed(2)} probability cutoff.`
            : `${formatInteger(positivePixels)} of ${formatInteger(totalPixels)} pixels (${positivePercentage.toFixed(2)}%) classified as landslide.`
    );

    setText("configuration-threshold", threshold.toFixed(2));
    setText(
        "configuration-input-shape",
        Array.isArray(result.input_shape)
            ? result.input_shape.join(" × ")
            : `${dimensions[0]} × ${dimensions[1]} × 26`
    );
    if (lastRunDurationMs !== null) {
        setText("inference-duration", `${(lastRunDurationMs / 1000).toFixed(2)} s`);
    }

    // 4. Run Details Panel
    setText("result-file-name", file?.name ?? result.filename ?? "—");
    setText("result-mask-shape-detail", shapeLabel);
    setText("result-threshold-detail", threshold.toFixed(2));
    setText("result-positive", formatInteger(positivePixels));
    setText("result-percentage", formatPercentage(positivePercentage));

    // 5. Ground-Truth Canvas & Reference Evaluation
    const rawTruthMask =
        result.ground_truth_mask ?? result.reference_mask ?? null;
    let truthStats = null;

    if (rawTruthMask !== null) {
        truthStats = drawMaskOnCanvas(groundTruthCanvas, rawTruthMask, {
            binary: true,
            positiveRgb: [56, 189, 248], // Cyan reference highlight
            backgroundRgb: [19, 29, 46]
        });

        setPlaceholder("ground-truth-canvas", "ground-truth-placeholder", false);
        setText(
            "ground-truth-visualization-note",
            `${formatInteger(truthStats.positivePixels)} reference positive pixels`
        );
        setText("ground-truth-shape", `Shape: ${truthStats.height} × ${truthStats.width}`);
    } else {
        resetCanvas(groundTruthCanvas);
        setPlaceholder("ground-truth-canvas", "ground-truth-placeholder", true);
        setText(
            "ground-truth-visualization-note",
            getSelectedMask()
                ? "Reference mask evaluated on server"
                : "Reference data is optional"
        );
        setText("ground-truth-shape", "Shape: —");
    }

    updateEvaluationSection(result, predStats, truthStats);
    setResultStatePill("Analysis complete", "success");

    if (exportButton) exportButton.disabled = false;
}

/* =====================================================
   RESET & EXPORT SUMMARY
===================================================== */

function resetResultsDisplay() {
    latestPrediction = null;
    lastRunDurationMs = null;

    resetCanvas(maskCanvas);
    resetCanvas(groundTruthCanvas);
    setPlaceholder("mask-canvas", "mask-placeholder", true);
    setPlaceholder("ground-truth-canvas", "ground-truth-placeholder", true);

    if (evaluationPanel) evaluationPanel.hidden = true;
    if (exportButton) exportButton.disabled = true;

    setText("positive-metric", "—");
    setText("positive-metric-caption", "Available after prediction");
    setText("percentage-metric", "—");
    setBarWidth("predicted-share-bar", 0);
    setText("result-shape", "—");
    setText("result-mask-shape", "Waiting for model output");

    setText("prediction-visualization-note", "No prediction generated");
    setText("ground-truth-visualization-note", "Reference data is optional");
    setText("ground-truth-shape", "Shape: —");

    setBarWidth("composition-positive", 0);
    setBarWidth("composition-background", 100);
    setText("composition-positive-label", "—");
    setText("composition-background-label", "—");
    setText("coverage-value", "—");
    setText(
        "coverage-description",
        "The share of grid pixels classified as landslide will appear after inference."
    );
    setText("inference-duration", "—");

    setText("result-file-name", "—");
    setText("result-mask-shape-detail", "—");
    setText("result-positive", "—");
    setText("result-percentage", "—");
}

form?.addEventListener("reset", () => {
    setTimeout(() => {
        clearFieldError(false);
        clearFieldError(true);
        setAnalysisError("");
        handleSatelliteFileChange();
        handleMaskFileChange();
        handleThresholdInput();
        resetResultsDisplay();
        setResultStatePill("Awaiting analysis", "idle");
        showToast("Workspace reset.", "info");
    }, 0);
});

exportButton?.addEventListener("click", () => {
    if (!latestPrediction) return;

    const exportData = {
        timestamp: new Date().toISOString(),
        input_file: getSelectedFile()?.name ?? $("result-file-name")?.textContent,
        reference_mask_file: getSelectedMask()?.name ?? null,
        threshold: Number(thresholdInput?.value ?? 0.5),
        inference_duration_ms: lastRunDurationMs,
        prediction_summary: {
            output_shape: $("result-mask-shape-detail")?.textContent,
            positive_pixels: $("result-positive")?.textContent,
            positive_share: $("result-percentage")?.textContent
        },
        evaluation_metrics: evaluationPanel?.hidden
            ? null
            : {
                  dice: $("dice-score")?.textContent,
                  iou: $("iou-score")?.textContent,
                  precision: $("precision-score")?.textContent,
                  recall: $("recall-score")?.textContent,
                  accuracy: $("accuracy-score")?.textContent,
                  specificity: $("specificity-score")?.textContent,
                  confusion_matrix: {
                      tp: $("true-positive-count")?.textContent,
                      fp: $("false-positive-count")?.textContent,
                      tn: $("true-negative-count")?.textContent,
                      fn: $("false-negative-count")?.textContent
                  }
              }
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `terrawatch-run-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);

    showToast("Run summary exported as JSON.", "success");
});

/* =====================================================
   FORM SUBMISSION (PREDICTION REQUEST)
===================================================== */

form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    setAnalysisError("");

    const satelliteFile = getSelectedFile();
    const satelliteError = validateHdf5File(satelliteFile, "Satellite");
    if (satelliteError) {
        showFieldError(satelliteError, false);
        chooseFileButton?.focus();
        return;
    }

    const maskFile = getSelectedMask();
    if (maskFile) {
        const maskError = validateHdf5File(maskFile, "Ground-truth mask");
        if (maskError) {
            showFieldError(maskError, true);
            return;
        }
    }

    const threshold = Number(thresholdInput?.value ?? 0.5);
    if (!Number.isFinite(threshold) || threshold < 0.05 || threshold > 0.95) {
        showToast("Select a valid threshold between 0.05 and 0.95.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", satelliteFile);
    formData.append("threshold", String(threshold));

    // Matches <input id="mask-file" name="mask_file"> in index.html
    if (maskFile) {
        formData.append("mask_file", maskFile);
    }

    if (runButton) runButton.disabled = true;
    if (chooseFileButton) chooseFileButton.disabled = true;
    if (chooseMaskButton) chooseMaskButton.disabled = true;
    if (resetButton) resetButton.disabled = true;

    form.classList.add("is-submitting");
    if (runButtonText) runButtonText.textContent = "Running analysis…";
    setResultStatePill("Running analysis…", "pending");

    const startTime = performance.now();

    try {
        const response = await fetch(PREDICTION_ENDPOINT, {
            method: "POST",
            body: formData,
            headers: { Accept: "application/json" }
        });

        const contentType = response.headers.get("content-type") || "";
        let result;

        if (contentType.includes("application/json")) {
            result = await response.json();
        } else {
            const rawText = await response.text();
            throw new Error(rawText || `Server returned HTTP ${response.status}.`);
        }

        if (!response.ok) {
            const detail = result?.detail;
            throw new Error(
                typeof detail === "string"
                    ? detail
                    : Array.isArray(detail)
                      ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
                      : `Analysis failed (HTTP ${response.status}).`
            );
        }

        lastRunDurationMs = Math.round(performance.now() - startTime);
        latestPrediction = result;

        displayPrediction(result, threshold, satelliteFile);

        showToast(
            maskFile
                ? "Segmentation and reference mask evaluation complete."
                : "Landslide segmentation complete.",
            "success"
        );

        $("results-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        console.error("Analysis request failed:", error);
        setResultStatePill("Analysis failed", "error");

        const message =
            error instanceof Error
                ? error.message
                : "An unexpected error occurred during analysis.";

        setAnalysisError(message);
        showToast(message, "error");
    } finally {
        if (runButton) runButton.disabled = !getSelectedFile();
        if (chooseFileButton) chooseFileButton.disabled = false;
        if (chooseMaskButton) chooseMaskButton.disabled = false;
        if (resetButton) resetButton.disabled = false;

        form.classList.remove("is-submitting");
        if (runButtonText) runButtonText.textContent = "Run analysis";
    }
});

/* =====================================================
   INITIALIZATION
===================================================== */

resetResultsDisplay();
handleThresholdInput();
handleSatelliteFileChange();
handleMaskFileChange();
checkApiStatus();