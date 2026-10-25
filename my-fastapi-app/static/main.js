let leftWidth = 20;
let middleWidth = 40;
let isResizing = false;
let currentHandle = null;

// Resizable panels
const handle1 = document.getElementById('handle1');
const handle2 = document.getElementById('handle2');
const contentWrapper = document.querySelector('.content-wrapper');

// Track current state
let currentMindmapId = null;
let currentBlockId = null;
let currentGraphId = null;
let allMindmaps = [];
let currentGraphData = null;
let simulation = null;

const modal = document.getElementById('ui-modal');
const modalTitle = document.getElementById('ui-modal-title');
const modalMessage = document.getElementById('ui-modal-message');
const modalInput = document.getElementById('ui-modal-input');
const modalConfirm = document.getElementById('ui-modal-confirm');
const modalCancel = document.getElementById('ui-modal-cancel');
const modalClose = document.querySelector('.ui-modal-close');
const modalBackdrop = document.querySelector('[data-modal-close]');
let modalResolve = null;

function showModal(options) {
    if (!modal) {
        if (options.type === 'confirm') return Promise.resolve(false);
        if (options.type === 'prompt') return Promise.resolve(null);
        return Promise.resolve();
    }

    const type = options.type || 'alert';
    modal.dataset.type = type;
    modalTitle.textContent = options.title || 'Notice';
    modalMessage.textContent = options.message || '';
    modalConfirm.textContent = options.confirmText || 'OK';
    modalCancel.textContent = options.cancelText || 'Cancel';

    const showInput = type === 'prompt';
    modalInput.classList.toggle('is-visible', showInput);
    if (showInput) {
        modalInput.value = options.defaultValue || '';
        modalInput.placeholder = options.placeholder || '';
    }

    modalCancel.classList.toggle('is-hidden', type === 'alert');

    modal.classList.remove('hidden');
    requestAnimationFrame(() => modal.classList.add('is-visible'));
    modal.setAttribute('aria-hidden', 'false');

    setTimeout(() => {
        if (showInput) {
            modalInput.focus();
            modalInput.select();
        } else {
            modalConfirm.focus();
        }
    }, 0);

    return new Promise((resolve) => {
        modalResolve = resolve;
    });
}

function closeModal(result) {
    if (!modal) return;
    modal.classList.remove('is-visible');
    modal.setAttribute('aria-hidden', 'true');
    window.setTimeout(() => {
        modal.classList.add('hidden');
    }, 200);

    if (modalResolve) {
        const resolve = modalResolve;
        modalResolve = null;
        resolve(result);
    }
}

function handleModalCancel() {
    if (!modal) return;
    const type = modal.dataset.type;
    if (type === 'prompt') {
        closeModal(null);
        return;
    }
    if (type === 'confirm') {
        closeModal(false);
        return;
    }
    closeModal();
}

function handleModalConfirm() {
    if (!modal) return;
    const type = modal.dataset.type;
    if (type === 'prompt') {
        closeModal(modalInput.value.trim());
        return;
    }
    if (type === 'confirm') {
        closeModal(true);
        return;
    }
    closeModal();
}

if (modalConfirm) {
    modalConfirm.addEventListener('click', handleModalConfirm);
}

if (modalCancel) {
    modalCancel.addEventListener('click', handleModalCancel);
}

if (modalClose) {
    modalClose.addEventListener('click', handleModalCancel);
}

if (modalBackdrop) {
    modalBackdrop.addEventListener('click', handleModalCancel);
}

document.addEventListener('keydown', (event) => {
    if (!modal || !modal.classList.contains('is-visible')) return;
    if (event.key === 'Escape') {
        event.preventDefault();
        handleModalCancel();
    }
    if (event.key === 'Enter' && modal.dataset.type === 'prompt') {
        event.preventDefault();
        handleModalConfirm();
    }
});

async function showAlert(message, title = 'Notice') {
    await showModal({ type: 'alert', title, message, confirmText: 'OK' });
}

async function showConfirm(message, title = 'Confirm', confirmText = 'Confirm', cancelText = 'Cancel') {
    return await showModal({ type: 'confirm', title, message, confirmText, cancelText });
}

async function showPrompt(
    message,
    title = 'New mindmap',
    placeholder = '',
    confirmText = 'Create',
    cancelText = 'Cancel'
) {
    return await showModal({ type: 'prompt', title, message, placeholder, confirmText, cancelText });
}

// API Functions
async function fetchMindmaps() {
    try {
        const response = await fetch('/api/mindmaps');
        if (!response.ok) throw new Error('Failed to fetch mindmaps');
        const data = await response.json();
        return data.mindmaps;
    } catch (error) {
        console.error('Error fetching mindmaps:', error);
        return [];
    }
}

async function fetchGraphData(graphId) {
    try {
        const response = await fetch(`/api/mindmaps/${graphId}/graph`);
        if (!response.ok) throw new Error('Failed to fetch graph');
        return await response.json();
    } catch (error) {
        console.error('Error fetching graph:', error);
        return null;
    }
}

async function switchMindmap(graphId) {
    try {
        const response = await fetch(`/api/mindmaps/${graphId}/switch`, {
            method: 'POST'
        });
        if (!response.ok) throw new Error('Failed to switch mindmap');
        return await response.json();
    } catch (error) {
        console.error('Error switching mindmap:', error);
        return null;
    }
}

async function switchBlock(blockId) {
    try {
        const response = await fetch(`/api/blocks/${blockId}/switch`, {
            method: 'POST'
        });
        if (!response.ok) throw new Error('Failed to switch block');
        return await response.json();
    } catch (error) {
        console.error('Error switching block:', error);
        return null;
    }
}

async function fetchBlockMessages(blockId) {
    try {
        const response = await fetch(`/api/blocks/${blockId}/messages`);
        if (!response.ok) throw new Error('Failed to fetch messages');
        return await response.json();
    } catch (error) {
        console.error('Error fetching messages:', error);
        return null;
    }
}

async function deleteBlock(blockId) {
    try {
        const response = await fetch(`/api/blocks/${blockId}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error('Failed to delete block');
        return await response.json();
    } catch (error) {
        console.error('Error deleting block:', error);
        return null;
    }
}

async function deleteMindmapRequest(graphId) {
    try {
        const response = await fetch(`/api/mindmaps/${graphId}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error('Failed to delete mindmap');
        return await response.json();
    } catch (error) {
        console.error('Error deleting mindmap:', error);
        return null;
    }
}

function renderMessageContent(content) {
    if (window.marked) {
        return window.marked.parse(content);
    }
    return content;
}

function renderMindmapList(mindmaps, activeGraphId) {
    allMindmaps = mindmaps;
    const mindmapList = document.querySelector('.mindmap-list');
    mindmapList.innerHTML = '';

    mindmaps.forEach((mindmap) => {
        const li = document.createElement('li');
        li.className = 'mindmap-item' + (mindmap.graph_id === activeGraphId ? ' active' : '');
        li.dataset.graphId = mindmap.graph_id;
        li.innerHTML = `
            <span class="mindmap-title">${mindmap.title}</span>
            <button class="mindmap-delete-btn" onclick="deleteMindmap(event, '${mindmap.graph_id}')">x</button>
        `;
        li.onclick = function() { selectMindmap(this, mindmap.graph_id); };
        mindmapList.appendChild(li);
    });
}

function updateRightHeader(title, intent, blockId) {
    const rightHeader = document.querySelector('.right-panel-header');
    const canDelete = blockId && currentGraphData && blockId !== currentGraphData.root_block_id;
    const safeTitle = title || 'AI Chat';
    const safeIntent = intent || 'No intent';

    rightHeader.innerHTML = `
        <div class="right-header-content">
            <div>
                <h3>${safeTitle}</h3>
                <p class="right-header-subtitle">${safeIntent}</p>
            </div>
            <button class="delete-block-btn${canDelete ? '' : ' hidden'}" type="button" onclick="deleteCurrentBlock()" aria-label="Delete block">x</button>
        </div>
    `;
}

function updateRightHeaderFromGraph() {
    if (!currentGraphData || !currentBlockId) return;
    const node = currentGraphData.nodes?.find(n => n.id === currentBlockId);
    if (!node) return;
    updateRightHeader(node.label, node.intent, currentBlockId);
}

function getDescendantCount(blockId) {
    if (!currentGraphData || !currentGraphData.links) return 0;
    const childrenMap = new Map();
    const getId = (value) => (typeof value === 'object' && value ? value.id : value);

    currentGraphData.links.forEach(link => {
        const sourceId = getId(link.source);
        const targetId = getId(link.target);
        if (!childrenMap.has(sourceId)) {
            childrenMap.set(sourceId, []);
        }
        childrenMap.get(sourceId).push(targetId);
    });

    const visited = new Set();
    const stack = [...(childrenMap.get(blockId) || [])];
    while (stack.length) {
        const current = stack.pop();
        if (visited.has(current)) continue;
        visited.add(current);
        const children = childrenMap.get(current) || [];
        stack.push(...children);
    }

    return visited.size;
}

async function deleteCurrentBlock() {
    if (!currentBlockId || !currentGraphData) return;
    if (currentBlockId === currentGraphData.root_block_id) {
        await showAlert('You cannot delete the root block.', 'Cannot delete');
        return;
    }

    const descendantCount = getDescendantCount(currentBlockId);
    const warning = descendantCount > 0
        ? `This will delete this block and ${descendantCount} descendant(s). Continue?`
        : 'Delete this block?';

    const shouldDelete = await showConfirm(warning, 'Delete block', 'Delete');
    if (!shouldDelete) return;

    const result = await deleteBlock(currentBlockId);
    if (!result) return;

    currentGraphId = result.graph_id;
    currentMindmapId = result.graph_id;
    currentBlockId = result.current_block_id;
    currentGraphData = result.graph;

    if (result.graph) {
        drawMindmap(result.graph);
    }

    const mindmaps = await fetchMindmaps();
    renderMindmapList(mindmaps, currentGraphId);
    updateRightHeaderFromGraph();

    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    result.messages.forEach(msg => {
        const roleClass = msg.role === 'assistant' ? 'bot' : msg.role;
        const wrapper = document.createElement('div');
        wrapper.className = `chat-message-wrapper ${roleClass}`;
        const timestamp = new Date(msg.timestamp * 1000).toLocaleTimeString();
        wrapper.innerHTML = `
            <div class="chat-message">
                <div class="chat-bubble">${renderMessageContent(msg.content)}</div>
                <div class="chat-timestamp">${timestamp}</div>
            </div>
        `;
        chatMessages.appendChild(wrapper);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Generic chat API using ConversationManager (mirrors CLI logic)
async function chatWithAssistant(content) {
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content })
        });
        if (!response.ok) throw new Error('Failed to send chat message');
        return await response.json();
    } catch (error) {
        console.error('Error in chatWithAssistant:', error);
        return null;
    }
}

// Create a new mindmap via dedicated endpoint
async function createMindmap(topic) {
    try {
        const response = await fetch('/api/mindmaps/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic })
        });
        if (!response.ok) throw new Error('Failed to create mindmap');
        return await response.json();
    } catch (error) {
        console.error('Error in createMindmap:', error);
        return null;
    }
}

function startResize(e, handle) {
    isResizing = true;
    currentHandle = handle;
    e.preventDefault();
}

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;

    const rect = contentWrapper.getBoundingClientRect();
    const totalWidth = rect.width;
    const mouseX = e.clientX - rect.left;
    const percentX = (mouseX / totalWidth) * 100;

    if (currentHandle === handle1) {
        leftWidth = Math.max(15, Math.min(40, percentX));
    } else if (currentHandle === handle2) {
        middleWidth = Math.max(20, Math.min(70, percentX - leftWidth));
    }

    updateLayout();
});

document.addEventListener('mouseup', () => {
    isResizing = false;
    currentHandle = null;
});

function updateLayout() {
    const leftPanel = document.querySelector('.left-panel');
    const middlePanel = document.querySelector('.middle-panel');
    const rightPanel = document.querySelector('.right-panel');

    leftPanel.style.flex = `0 0 ${leftWidth}%`;
    middlePanel.style.flex = `0 0 ${middleWidth}%`;
    rightPanel.style.flex = `1`;
    
    // Redraw mindmap on resize
    if (simulation) {
        drawMindmap(currentMindmapId);
    }
}

handle1.addEventListener('mousedown', (e) => startResize(e, handle1));
handle2.addEventListener('mousedown', (e) => startResize(e, handle2));

// Mindmap selection
async function selectMindmap(element, graphId) {
    document.querySelectorAll('.mindmap-item').forEach(item => {
        item.classList.remove('active');
    });
    if (element) element.classList.add('active');
    
    currentGraphId = graphId;
    currentMindmapId = graphId;

    await switchMindmap(graphId);
    
    // Fetch graph data from API
    const graphData = await fetchGraphData(graphId);
    if (graphData) {
        currentGraphData = graphData;
        const rootBlock = graphData.nodes.find(n => n.id === graphData.root_block_id);
        if (rootBlock) {
            document.getElementById('mindmapTitle').textContent = rootBlock.label;
            currentBlockId = rootBlock.id;  // Set current block to root
            await switchBlock(rootBlock.id);
        }
        drawMindmap(graphData);
        updateRightHeaderFromGraph();
        
        // Load initial block messages
        if (graphData.root_block_id) {
            await loadBlockMessages(graphData.root_block_id);
        }
    }
}

// Draw mindmap using D3.js force-directed graph with API data
function drawMindmap(graphData) {
    const svg = d3.select('#mindmapSvg');
    svg.selectAll('*').remove(); // Clear previous graph
    
    const container = document.querySelector('#mindmapSvg').parentElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
        console.warn('No graph data to display');
        return;
    }
    
    // Create force simulation
    simulation = d3.forceSimulation(graphData.nodes)
        .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(50));
    
    // Create links with relation-based coloring
    const link = svg.append('g')
        .selectAll('line')
        .data(graphData.links)
        .enter().append('line')
        .attr('stroke', d => d.color || '#999')
        .attr('stroke-width', d => d.strokeWidth || 2)
        .attr('stroke-opacity', 0.7);

    // Hover tooltip: why these two blocks are connected (SVG needs a <title> child)
    link.append('title')
        .text(d => {
            const label = `${d.relation} (${((d.confidence || 0) * 100).toFixed(0)}%)`;
            return d.reasoning ? `${label}\n${d.reasoning}` : label;
        });

    // Create nodes
    const node = svg.append('g')
        .selectAll('g')
        .data(graphData.nodes)
        .enter().append('g')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Add circles for nodes
    node.append('circle')
        .attr('r', d => d.id === graphData.root_block_id ? 30 : 20)
        .attr('fill', d => d.is_current ? '#FF6B6B' : (d.id === graphData.root_block_id ? '#0d6efd' : '#6610f2'))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('click', async (event, d) => {
            // Clicking a node just loads its messages into the chat panel.
            await switchBlock(d.id);
            await loadBlockMessages(d.id);
            
            // Update current block highlighting
            d3.selectAll('circle').attr('fill', node_d => 
                node_d.id === d.id ? '#FF6B6B' : (node_d.id === graphData.root_block_id ? '#0d6efd' : '#6610f2')
            );
            currentBlockId = d.id;
        });
    
    // Add labels
    node.append('text')
        .text(d => d.label)
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('font-size', d => d.id === graphData.root_block_id ? '14px' : '12px')
        .attr('font-weight', d => d.id === graphData.root_block_id ? 'bold' : 'normal')
        .attr('fill', '#333')
        .style('pointer-events', 'none');
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    
    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Initialize mindmap on page load
window.addEventListener('load', async () => {
    // Load mindmaps from API
    const mindmaps = await fetchMindmaps();
    
    if (mindmaps.length === 0) {
        // Show empty state with instruction to create new mindmap
        document.getElementById('mindmapTitle').textContent = 'Graph';
        const svg = d3.select('#mindmapSvg');
        svg.selectAll('*').remove();
        svg.append('text')
            .attr('x', '50%')
            .attr('y', '50%')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.3em')
            .attr('fill', '#999')
            .attr('font-size', '16px')
            .text('No mindmaps. Press the "new mindmap" button to create one.');
        
        // Show placeholder in right panel
        updateRightHeader('Ready to chat', '', null);
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '<div style="color: #999; text-align: center; margin-top: 20px;">Click the Add Mindmap button to start</div>';
        return;
    }
    
    const activeGraphId = mindmaps.find(m => m.is_current)?.graph_id || mindmaps[0]?.graph_id;
    renderMindmapList(mindmaps, activeGraphId);

    if (activeGraphId) {
        const activeItem = Array.from(document.querySelectorAll('.mindmap-item'))
            .find(item => item.dataset.graphId === activeGraphId);
        if (activeItem) {
            selectMindmap(activeItem, activeGraphId);
        }
    }
});

// Load block messages and display in right panel
async function loadBlockMessages(blockId) {
    const blockData = await fetchBlockMessages(blockId);
    if (!blockData) return;
    
    currentBlockId = blockId;
    
    // Update right panel header
    updateRightHeader(blockData.title, blockData.intent, blockId);
    
    // Clear and populate messages
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = '';
    
    blockData.messages.forEach(msg => {
        const roleClass = msg.role === 'assistant' ? 'bot' : msg.role;
        const wrapper = document.createElement('div');
        wrapper.className = `chat-message-wrapper ${roleClass}`;
        const timestamp = new Date(msg.timestamp * 1000).toLocaleTimeString();
        wrapper.innerHTML = `
            <div class="chat-message">
                <div class="chat-bubble">${renderMessageContent(msg.content)}</div>
                <div class="chat-timestamp">${timestamp}</div>
            </div>
        `;
        chatMessages.appendChild(wrapper);
    });
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Chat functionality
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();

    if (!message) return;

    // Treat /new as a shortcut to "start a new conversation" via chat endpoint
    if (message === '/new') {
        input.value = '';
        autoResizeTextarea();
        return addMindmap();
    }

    const chatMessages = document.getElementById('chatMessages');

    // Optimistic user message
    const userWrapper = document.createElement('div');
    userWrapper.className = 'chat-message-wrapper user';
    userWrapper.innerHTML = `<div class="chat-message"><div class="chat-bubble">${message}</div><div class="chat-timestamp">Just now</div></div>`;
    chatMessages.appendChild(userWrapper);

    input.value = '';
    autoResizeTextarea();

    // Show thinking indicator while waiting for the assistant response.
    const thinkingWrapper = document.createElement('div');
    thinkingWrapper.className = 'chat-message-wrapper bot thinking';
    thinkingWrapper.innerHTML = `
        <div class="chat-message">
            <div class="chat-bubble">
                <span class="thinking-dots" aria-label="AI is thinking">
                    <span></span><span></span><span></span>
                </span>
            </div>
            <div class="chat-timestamp">Thinking...</div>
        </div>
    `;
    chatMessages.appendChild(thinkingWrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const result = await chatWithAssistant(message);
    if (!result) {
        thinkingWrapper.querySelector('.chat-bubble').textContent = 'Sorry, something went wrong.';
        return;
    }

    // Update current graph and block from response
    currentGraphId = result.graph_id;
    currentMindmapId = result.graph_id;
    currentBlockId = result.current_block_id;

    // Refresh mindmap list and update active selection
    const mindmaps = await fetchMindmaps();
    renderMindmapList(mindmaps, currentGraphId);

    // Redraw mindmap from returned graph
    if (result.graph) {
        currentGraphData = result.graph;
        drawMindmap(result.graph);
    }

    updateRightHeaderFromGraph();

    // Replace chat panel with full message history for current block
    chatMessages.innerHTML = '';
    result.messages.forEach(msg => {
        const roleClass = msg.role === 'assistant' ? 'bot' : msg.role;
        const wrapper = document.createElement('div');
        wrapper.className = `chat-message-wrapper ${roleClass}`;
        const timestamp = new Date(msg.timestamp * 1000).toLocaleTimeString();
        wrapper.innerHTML = `
            <div class="chat-message">
                <div class="chat-bubble">${renderMessageContent(msg.content)}</div>
                <div class="chat-timestamp">${timestamp}</div>
            </div>
        `;
        chatMessages.appendChild(wrapper);
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Auto-resize textarea as user types
const messageInput = document.getElementById('messageInput');

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
}

messageInput.addEventListener('input', autoResizeTextarea);

// Allow Enter key to send (Ctrl+Enter for new line)
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.ctrlKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Add mindmap function - starts a new conversation via /api/chat
async function addMindmap() {
    const topic = await showPrompt('What would you like to discuss?', 'New mindmap', 'Mindmap topic');
    if (!topic) return;  // User cancelled

    const chatMessages = document.getElementById('chatMessages');

    // Echo the user's topic in the chat panel.
    const userWrapper = document.createElement('div');
    userWrapper.className = 'chat-message-wrapper user';
    userWrapper.innerHTML = `
        <div class="chat-message">
            <div class="chat-bubble">${topic}</div>
            <div class="chat-timestamp">Just now</div>
        </div>
    `;
    chatMessages.appendChild(userWrapper);

    // Show thinking indicator while waiting for the assistant response.
    const thinkingWrapper = document.createElement('div');
    thinkingWrapper.className = 'chat-message-wrapper bot thinking';
    thinkingWrapper.innerHTML = `
        <div class="chat-message">
            <div class="chat-bubble">
                <span class="thinking-dots" aria-label="AI is thinking">
                    <span></span><span></span><span></span>
                </span>
            </div>
            <div class="chat-timestamp">Thinking...</div>
        </div>
    `;
    chatMessages.appendChild(thinkingWrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const result = await createMindmap(topic);
    if (!result) {
        thinkingWrapper.querySelector('.chat-bubble').textContent = 'Sorry, something went wrong.';
        return;
    }

    // Update current graph and block
    currentGraphId = result.graph_id;
    currentMindmapId = result.graph_id;
    currentBlockId = result.current_block_id;

    // Refresh mindmap list from storage
    const mindmaps = await fetchMindmaps();
    renderMindmapList(mindmaps, currentGraphId);

    // Select the new mindmap visually and refresh middle panel
    const newItem = Array.from(document.querySelectorAll('.mindmap-item'))
        .find(item => item.dataset.graphId === currentGraphId);
    if (newItem) {
        await selectMindmap(newItem, currentGraphId);
    } else {
        await selectMindmap(document.querySelector('.mindmap-item'), currentGraphId);
    }
}

// Delete mindmap function (not implemented)
async function deleteMindmap(event, graphId) {
    event.stopPropagation();
    const shouldDelete = await showConfirm(
        'Delete this mindmap? This will remove all its blocks and messages.',
        'Delete mindmap',
        'Delete'
    );
    if (!shouldDelete) {
        return;
    }

    deleteMindmapRequest(graphId).then(async (result) => {
        if (!result) return;

        const mindmaps = await fetchMindmaps();
        if (mindmaps.length === 0) {
            renderMindmapList([], '');
            currentGraphId = null;
            currentMindmapId = null;
            currentBlockId = null;
            currentGraphData = null;
            document.getElementById('mindmapTitle').textContent = 'Graph';
            const svg = d3.select('#mindmapSvg');
            svg.selectAll('*').remove();
            svg.append('text')
                .attr('x', '50%')
                .attr('y', '50%')
                .attr('text-anchor', 'middle')
                .attr('dy', '0.3em')
                .attr('fill', '#999')
                .attr('font-size', '16px')
                .text('No mindmaps. Type "/new" to create one.');
            updateRightHeader('Ready to chat', '', null);
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = '<div style="color: #999; text-align: center; margin-top: 20px;">Create a mindmap with /new to start</div>';
            return;
        }

        const nextGraphId = result.current_graph_id || mindmaps[0].graph_id;
        renderMindmapList(mindmaps, nextGraphId);
        const nextItem = Array.from(document.querySelectorAll('.mindmap-item'))
            .find(item => item.dataset.graphId === nextGraphId);
        if (nextItem) {
            await selectMindmap(nextItem, nextGraphId);
        }
    });
}
