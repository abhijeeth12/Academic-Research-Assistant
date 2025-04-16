let currentState = {
    query: "",
    documents: []
  };
  
  async function searchDocuments() {
    const query = document.getElementById("queryInput").value.trim();
    if (!query) return alert("Please enter a query");
    
    const selectedTypes = Array.from(document.querySelectorAll('input[name="docType"]:checked'))
                            .map(el => el.value);
    
    showLoading(true);
    currentState.query = query;
    
    try {
        const response = await fetch("/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                query, 
                types: selectedTypes 
            })
        });
        
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        
        currentState.documents = data.documents;
        displayDocuments(data.documents);
        
        document.getElementById("searchPhase").style.display = "none";
        document.getElementById("resultsPhase").style.display = "block";
    } catch (error) {
        alert("Search failed: " + error.message);
    } finally {
        showLoading(false);
    }
  }
  
  function displayDocuments(documents) {
    const container = document.getElementById("documentList");
    container.innerHTML = "";
    
    // Group by type
    const grouped = {};
    documents.forEach(doc => {
        if (!grouped[doc.type]) grouped[doc.type] = [];
        grouped[doc.type].push(doc);
    });
    
    // Render each type
    for (const [type, docs] of Object.entries(grouped)) {
        const section = document.createElement("div");
        section.className = "document-type-section";
        section.innerHTML = `<h3>${type}</h3>`;
        
        docs.forEach(doc => {
            const docEl = document.createElement("div");
            docEl.className = "document-item";
            docEl.innerHTML = `
                <input type="checkbox" id="doc-${doc.url}" data-url="${doc.url}">
                <label for="doc-${doc.url}">
                    <h4>${doc.title}</h4>
                    <p class="author">${doc.author}</p>
                    <a href="${doc.url}" target="_blank">View Document</a>
                </label>
            `;
            section.appendChild(docEl);
        });
        
        container.appendChild(section);
    }
  }
  
  async function analyzeSelected() {
    const selected = Array.from(document.querySelectorAll('#documentList input:checked'))
                       .map(el => el.dataset.url);
    
    if (selected.length === 0) return alert("Please select at least one document");
    
    showLoading(true);
    document.getElementById("resultsPhase").style.display = "none";
    document.getElementById("analysisPhase").style.display = "block";
    
    const resultsContainer = document.getElementById("analysisResults");
    resultsContainer.innerHTML = "";
    
    for (const url of selected) {
        try {
            const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    url, 
                    query: currentState.query 
                })
            });
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            
            const docInfo = currentState.documents.find(d => d.url === url);
            
            const resultEl = document.createElement("div");
            resultEl.className = "analysis-result";
            resultEl.innerHTML = `
                <div class="document-header">
                    <h3>${docInfo.title}</h3>
                    <p class="author">${docInfo.author}</p>
                    <a href="${url}" target="_blank">Source Document</a>
                </div>
                
                <div class="summary">
                    <h4>Comprehensive Summary</h4>
                    <p>${data.summary}</p>
                </div>
                
                <div class="query-context">
                    <p><strong>User Query:</strong> ${currentState.query}</p>
                </div>
            `;
            
            resultsContainer.appendChild(resultEl);
        } catch (error) {
            console.error(`Error analyzing ${url}:`, error);
            const errorEl = document.createElement("div");
            errorEl.className = "analysis-error";
            errorEl.textContent = `Failed to analyze document: ${url}`;
            resultsContainer.appendChild(errorEl);
        }
    }
    
    showLoading(false);
  }
  
  function backToResults() {
    document.getElementById("analysisPhase").style.display = "none";
    document.getElementById("resultsPhase").style.display = "block";
  }
  
  function resetSearch() {
    document.getElementById("resultsPhase").style.display = "none";
    document.getElementById("searchPhase").style.display = "block";
    document.getElementById("queryInput").value = "";
    currentState = { query: "", documents: [] };
  }
  
  function showLoading(show) {
    document.getElementById("loadingOverlay").style.display = show ? "flex" : "none";
  }