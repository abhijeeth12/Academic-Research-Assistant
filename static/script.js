    let currentState = {
        query: "",
        documents: [],
        selectedDocs: []
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
        const container = document.getElementById("documentList");
        container.innerHTML = documents.map(doc => `
            <label class="document-item ${currentState.selectedDocs.includes(doc.url) ? 'selected' : ''}">
                <input type="checkbox" value="${doc.url}" class="visually-hidden" ${currentState.selectedDocs.includes(doc.url) ? 'checked' : ''}>
                <div class="document-content">
                    <h4>${doc.title}</h4>
                    <div class="document-meta">
                        <span>${doc.author}</span> • 
                        <span>${doc.type}</span>
                    </div>
                </div>
            </label>
        `).join('');
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
                <h3>${getDocTitle(result.url)}</h3>
                <div class="summary">${result.summary}</div>
                <div class="analysis-meta">
                    <a href="${result.url}" target="_blank">View Source Document</a>
                </div>
            </div>
        `).join('');
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
