const systemStatus = document.querySelector("#systemStatus");
const refreshButton = document.querySelector("#refreshButton");
const uploadForm = document.querySelector("#uploadForm");
const uploadButton = document.querySelector("#uploadButton");
const uploadStatus = document.querySelector("#uploadStatus");
const documentsTable = document.querySelector("#documentsTable");
const chatForm = document.querySelector("#chatForm");
const askButton = document.querySelector("#askButton");
const chatStatus = document.querySelector("#chatStatus");
const answerText = document.querySelector("#answerText");
const sourcesList = document.querySelector("#sourcesList");

function setStatus(element, message, type = "") {
  element.textContent = message;
  element.className = `status-line ${type}`.trim();
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : {};
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}

async function refreshHealth() {
  try {
    const health = await apiJson("/health");
    const llama = health.llama_server_reachable ? "LLM online" : "LLM offline";
    systemStatus.textContent = `${health.status.toUpperCase()} | DB ${health.database_connected ? "online" : "offline"} | ${llama}`;
  } catch (error) {
    systemStatus.textContent = `API error | ${error.message}`;
  }
}

async function refreshDocuments() {
  const documents = await apiJson("/documents");
  documentsTable.replaceChildren();

  if (!documents.length) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="5">No documents</td>`;
    documentsTable.append(row);
    return;
  }

  for (const documentItem of documents) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(documentItem.document_name)}</td>
      <td>${escapeHtml(documentItem.language || "-")}</td>
      <td>${escapeHtml(documentItem.status)}</td>
      <td>${documentItem.chunks}</td>
      <td><button class="secondary" type="button" data-delete="${documentItem.document_id}">Delete</button></td>
    `;
    documentsTable.append(row);
  }
}

async function refreshAll() {
  await Promise.all([refreshHealth(), refreshDocuments()]);
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.querySelector("#documentFile");
  if (!fileInput.files.length) return;

  uploadButton.disabled = true;
  setStatus(uploadStatus, "Uploading...");

  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    const uploaded = await apiJson("/documents/upload", {
      method: "POST",
      body: formData,
    });

    setStatus(uploadStatus, "Processing embeddings...");
    const processed = await apiJson(`/documents/${uploaded.document_id}/process`, {
      method: "POST",
    });

    setStatus(
      uploadStatus,
      `Ready | ${processed.document_name} | ${processed.chunks} chunk(s)`,
      "ok",
    );
    fileInput.value = "";
    await refreshDocuments();
  } catch (error) {
    setStatus(uploadStatus, error.message, "error");
  } finally {
    uploadButton.disabled = false;
  }
});

documentsTable.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-delete]");
  if (!button) return;

  button.disabled = true;
  try {
    await apiJson(`/documents/${button.dataset.delete}`, { method: "DELETE" });
    await refreshDocuments();
  } catch (error) {
    setStatus(uploadStatus, error.message, "error");
    button.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  askButton.disabled = true;
  answerText.textContent = "";
  sourcesList.replaceChildren();
  setStatus(chatStatus, "Retrieving...");

  const payload = {
    question: document.querySelector("#question").value,
    top_k: Number(document.querySelector("#topK").value),
    retrieval_mode: document.querySelector("#retrievalMode").value,
    rerank: document.querySelector("#rerank").checked,
    no_generate: !document.querySelector("#generate").checked,
  };

  try {
    const response = await apiJson("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    answerText.textContent = response.answer || response.prompt_preview || "";
    renderSources(response.sources);
    setStatus(
      chatStatus,
      `${response.cache_hit ? "Cache hit" : "Done"} | ${response.latency_ms} ms | ${response.query_id}`,
      "ok",
    );
  } catch (error) {
    setStatus(chatStatus, error.message, "error");
  } finally {
    askButton.disabled = false;
  }
});

refreshButton.addEventListener("click", () => {
  refreshAll().catch((error) => {
    systemStatus.textContent = `Refresh error | ${error.message}`;
  });
});

function renderSources(sources) {
  sourcesList.replaceChildren();
  for (const source of sources) {
    const item = document.createElement("li");
    item.className = "source-item";
    item.innerHTML = `
      <strong>${escapeHtml(source.document_name)}</strong>
      <div class="source-meta">Page ${source.page_number ?? "-"} | ${source.language || "-"} | score ${source.score}</div>
      <div>${escapeHtml(source.preview)}</div>
    `;
    sourcesList.append(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

refreshAll().catch((error) => {
  systemStatus.textContent = `Startup error | ${error.message}`;
});
