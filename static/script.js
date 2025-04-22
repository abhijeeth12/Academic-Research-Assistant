currentState = {
    query: "",
    documents: [],
    selectedDocs: [],
    currentPage: 1,
    itemsPerPage: 10
};
document.addEventListener('DOMContentLoaded', initApp);

function initApp() {
    bindDocumentEvents();
}

function bindDocumentEvents() {
    document.getElementById('documentList').addEventListener('click', (e) => {
        const item = e.target.closest('.document-item');
        if (item) {
            const checkbox = item.querySelector('input[type="checkbox"]');
            checkbox.checked = !checkbox.checked;
            toggleDocumentSelection(item, checkbox.checked);
            updateSelectedCount();
        }
    });
}

function toggleDocumentSelection(item, isSelected) {
    const url = item.querySelector('input').value;
    const index = currentState.selectedDocs.indexOf(url);
    
    if (isSelected) {
        if (index === -1) {
            currentState.selectedDocs.push(url);
            item.classList.add('selected');
        }
    } else {
        currentState.selectedDocs.splice(index, 1);
        item.classList.remove('selected');
    }
}

function updateSelectedCount() {
    const count = currentState.selectedDocs.length;
    document.getElementById('selectedCount').textContent = count;
}

async function searchDocuments() {
    const query = document.getElementById("queryInput").value.trim();
    currentState.query = query;
    if (!query) return showError("Please enter a research query");
    
    const selectedTypes = getSelectedTypes();
    showLoading(true);

    try {
        const response = await fetch("/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, types: selectedTypes })
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        currentState.documents = data.documents;
        currentState.selectedDocs = [];
        displayDocuments(data.documents);
        showPhase('resultsPhase');
    } catch (error) {
        showError("Search failed: " + error.message);
    } finally {
        showLoading(false);
    }
}

function getSelectedTypes() {
    return Array.from(document.querySelectorAll('input[name="docType"]:checked'))
        .map(el => el.value);
}

function displayDocuments(documents) {
    // Add document type icons and better metadata display
    const start = (currentState.currentPage - 1) * currentState.itemsPerPage;
    const end = start + currentState.itemsPerPage;
    const paginatedDocs = currentState.documents.slice(start, end);
    const container = document.getElementById("documentList");
    container.innerHTML = documents.map(doc => `
        <label class="document-item ${currentState.selectedDocs.includes(doc.url) ? 'selected' : ''}">
            <input type="checkbox" value="${doc.url}" class="visually-hidden" ${currentState.selectedDocs.includes(doc.url) ? 'checked' : ''}>
            <div class="document-content">
                <div class="doc-icon">
                    ${getDocumentIcon(doc.type)}
                </div>
                <div class="doc-info">
                    <h4>${doc.title}</h4>
                    <div class="document-meta">
                        <span><i class="fas fa-user"></i> ${doc.author || 'Unknown'}</span>
                        <span><i class="fas fa-calendar"></i> ${doc.year || 'Unknown'}</span>
                        <span class="doc-type">${doc.type}</span>
                    </div>
                </div>
            </div>
        </label>
    `).join('');
    addPaginationControls();
}
function addPaginationControls() {
    const totalPages = Math.ceil(currentState.documents.length / currentState.itemsPerPage);
    if (totalPages <= 1) return;
    
    const pagination = document.createElement('div');
    pagination.className = 'pagination';
    
    // Previous button
    if (currentState.currentPage > 1) {
        pagination.innerHTML += `<button onclick="changePage(${currentState.currentPage - 1})">Previous</button>`;
    }
    
    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
        pagination.innerHTML += `<button class="${i === currentState.currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
    }
    
    // Next button
    if (currentState.currentPage < totalPages) {
        pagination.innerHTML += `<button onclick="changePage(${currentState.currentPage + 1})">Next</button>`;
    }
    
    document.getElementById('documentList').append(pagination);
}
function getDocumentIcon(type) {
    const icons = {
        'PDF': 'file-pdf',
        'PPT': 'file-powerpoint',
        'Word': 'file-word'
    };
    return `<i class="fas fa-${icons[type] || 'file'}"></i>`;
}

async function analyzeSelected() {
    if (currentState.selectedDocs.length === 0) {
        return showError("Please select at least one document");
    }

    showLoading(true);
    
    try {
        const results = await Promise.all(
            currentState.selectedDocs.map(url => analyzeDocument(url))
        );
        
        displayAnalysisResults(results);
        showPhase('analysisPhase');
    } catch (error) {
        showError("Analysis failed: " + error.message);
    } finally {
        showLoading(false);
    }
}

async function analyzeDocument(url) {
    const response = await fetch("/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, query: currentState.query })
    });
    
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    return { url, ...data };
}

function displayAnalysisResults(results) {
    const container = document.getElementById("analysisResults");
    container.innerHTML = results.map(result => `
        <div class="analysis-item">
            <div class="analysis-header">
                <h3>${getDocTitle(result.url)}</h3>
                <div class="relevance-badge">
                    Relevance: ${(result.stats.top_relevance * 100).toFixed(0)}%
                </div>
            </div>
            <div class="summary">${formatSummary(result.summary)}</div>
            <div class="analysis-footer">
                <a href="${result.url}" target="_blank" class="doc-link">
                    <i class="fas fa-external-link-alt"></i> View Source
                </a>
                <div class="analysis-stats">
                    <span>${result.stats.relevant_chunks} relevant sections</span>
                </div>
            </div>
        </div>
    `).join('');
}

function formatSummary(summary) {
    // Convert markdown-like formatting to HTML
    return summary
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function getDocTitle(url) {
    return currentState.documents.find(d => d.url === url).title;
}

function showPhase(phaseId) {
    document.querySelectorAll('.phase').forEach(phase => {
        phase.classList.toggle('active', phase.id === phaseId);
    });
}

function resetSearch() {
    currentState = { query: "", documents: [], selectedDocs: [] };
    document.getElementById("queryInput").value = "";
    showPhase('searchPhase');
    updateSelectedCount();
}

function backToResults() {
    showPhase('resultsPhase');
}

function showLoading(show) {
    document.getElementById("loadingOverlay").style.display = show ? 'flex' : 'none';
}

function showError(message) {
    const error = document.createElement('div');
    error.className = 'error-message';
    error.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        ${message}
    `;
    document.body.appendChild(error);
    setTimeout(() => error.remove(), 5000);
}
