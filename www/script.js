
// --- Global State ---
let ortSession = null;
let sgfContent = null;
let boardSize = 19;
let moves = [];
let komi = 7.5;

// --- Constants ---
const BLACK = 1;
const WHITE = 2;
const EMPTY = 0;

// --- Board Logic ---
class Board {
    constructor(size) {
        this.size = size;
        this.board = new Int8Array(size * size).fill(EMPTY);
        this.koPoint = -1; // -1 means no ko
        this.captures = { [BLACK]: 0, [WHITE]: 0 };
    }

    copy() {
        const newBoard = new Board(this.size);
        newBoard.board = new Int8Array(this.board);
        newBoard.koPoint = this.koPoint;
        newBoard.captures = { ...this.captures };
        return newBoard;
    }

    idx(x, y) {
        return y * this.size + x;
    }

    xy(idx) {
        return { x: idx % this.size, y: Math.floor(idx / this.size) };
    }

    get(x, y) {
        if (x < 0 || x >= this.size || y < 0 || y >= this.size) return -1; // Off-board
        return this.board[this.idx(x, y)];
    }

    play(x, y, color) {
        if (x < 0 || x >= this.size || y < 0 || y >= this.size) return false; // Pass or invalid
        
        const idx = this.idx(x, y);
        if (this.board[idx] !== EMPTY) return false;

        // Place stone
        this.board[idx] = color;
        const opp = color === BLACK ? WHITE : BLACK;
        let captured = [];

        // Check neighbors for captures
        const neighbors = [[x+1, y], [x-1, y], [x, y+1], [x, y-1]];
        for (const [nx, ny] of neighbors) {
            if (this.get(nx, ny) === opp) {
                const group = this.getGroup(nx, ny);
                if (group.liberties === 0) {
                    captured.push(...group.stones);
                }
            }
        }

        // Remove captured stones
        for (const cIdx of captured) {
            this.board[cIdx] = EMPTY;
        }
        this.captures[color] += captured.length;

        // Check for suicide (not allowed in most rules, but KataGo handles it? usually illegal)
        // Simple check: if no liberties and no captures, it's suicide
        const selfGroup = this.getGroup(x, y);
        if (selfGroup.liberties === 0 && captured.length === 0) {
            this.board[idx] = EMPTY; // Revert
            return false; // Suicide
        }

        // Handle Ko
        if (captured.length === 1 && selfGroup.stones.length === 1) {
            // Check if this recreates the previous state (simple ko)
            // For simple ko, we just mark the captured spot as illegal for next move
            this.koPoint = captured[0];
        } else {
            this.koPoint = -1;
        }

        return true;
    }

    getGroup(startX, startY) {
        const color = this.get(startX, startY);
        if (color === EMPTY) return { stones: [], liberties: 0 };

        const stack = [[startX, startY]];
        const visited = new Set();
        const stones = [];
        let liberties = 0;
        const visitedLibs = new Set();

        visited.add(this.idx(startX, startY));

        while (stack.length > 0) {
            const [x, y] = stack.pop();
            stones.push(this.idx(x, y));

            const neighbors = [[x+1, y], [x-1, y], [x, y+1], [x, y-1]];
            for (const [nx, ny] of neighbors) {
                if (nx < 0 || nx >= this.size || ny < 0 || ny >= this.size) continue;
                
                const nIdx = this.idx(nx, ny);
                const nColor = this.board[nIdx];

                if (nColor === EMPTY) {
                    if (!visitedLibs.has(nIdx)) {
                        liberties++;
                        visitedLibs.add(nIdx);
                    }
                } else if (nColor === color) {
                    if (!visited.has(nIdx)) {
                        visited.add(nIdx);
                        stack.push([nx, ny]);
                    }
                }
            }
        }
        return { stones, liberties };
    }
    
    getNumLiberties(x, y) {
        return this.getGroup(x, y).liberties;
    }
}

// --- SGF Parsing ---
function parseSGF(sgf) {
    // Very basic parser
    const moves = [];
    let size = 19;
    let komi = 7.5;

    // Extract Size
    const szMatch = sgf.match(/SZ\[(\d+)\]/);
    if (szMatch) size = parseInt(szMatch[1]);

    // Extract Komi
    const kmMatch = sgf.match(/KM\[([\d.]+)\]/);
    if (kmMatch) komi = parseFloat(kmMatch[1]);

    // Extract Moves
    // Matches ;B[xx] or ;W[xx]
    const regex = /;([BW])\[([a-zA-Z]*)\]/g;
    let match;
    while ((match = regex.exec(sgf)) !== null) {
        const color = match[1] === 'B' ? BLACK : WHITE;
        const coord = match[2];
        let x = -1, y = -1;
        if (coord.length === 2) {
            // SGF coordinates are a-s (0-18)
            // But sometimes they are A-S? Usually lowercase in SGF.
            // Let's handle both.
            const charCode0 = coord.charCodeAt(0);
            const charCode1 = coord.charCodeAt(1);
            x = charCode0 >= 97 ? charCode0 - 97 : charCode0 - 65;
            y = charCode1 >= 97 ? charCode1 - 97 : charCode1 - 65;
        }
        moves.push({ color, x, y });
    }
    return { size, komi, moves };
}

// --- Featurization ---
function featurize(board, history, pla) {
    // history is array of {color, x, y} (moves)
    // We need to reconstruct the board state history for the last 5 moves?
    // Or just the moves themselves?
    // KataGo features 9-13 are "Prev move locations".
    
    const size = board.size;
    const num_planes = 22;
    const input = new Float32Array(1 * num_planes * size * size).fill(0);
    
    const opp = pla === BLACK ? WHITE : BLACK;

    function set(b, c, h, w, val) {
        // NCHW layout: (batch, channel, height, width)
        // Index = b * (C*H*W) + c * (H*W) + h * W + w
        const idx = b * (num_planes * size * size) + c * (size * size) + h * size + w;
        input[idx] = val;
    }

    // 0: Ones
    // 1: Pla stones
    // 2: Opp stones
    // 3: 1 lib
    // 4: 2 libs
    // 5: 3 libs
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            set(0, 0, y, x, 1.0); // Feature 0: Ones

            const color = board.get(x, y);
            if (color === pla) {
                set(0, 1, y, x, 1.0);
            } else if (color === opp) {
                set(0, 2, y, x, 1.0);
            }

            if (color !== EMPTY) {
                const libs = board.getNumLiberties(x, y);
                if (libs === 1) set(0, 3, y, x, 1.0);
                if (libs === 2) set(0, 4, y, x, 1.0);
                if (libs === 3) set(0, 5, y, x, 1.0);
            }
        }
    }

    // 6: Ko
    if (board.koPoint !== -1) {
        const { x, y } = board.xy(board.koPoint);
        set(0, 6, y, x, 1.0);
    }

    // 9-13: History
    // history array contains all moves. We need the last 5.
    // move_idx is current move index (length of history applied)
    // history[last] is prev move (opp) -> feature 9
    // history[last-1] is prev-1 move (pla) -> feature 10
    // ...
    const len = history.length;
    if (len >= 1) {
        const m = history[len - 1];
        if (m.x >= 0) set(0, 9, m.y, m.x, 1.0);
    }
    if (len >= 2) {
        const m = history[len - 2];
        if (m.x >= 0) set(0, 10, m.y, m.x, 1.0);
    }
    if (len >= 3) {
        const m = history[len - 3];
        if (m.x >= 0) set(0, 11, m.y, m.x, 1.0);
    }
    if (len >= 4) {
        const m = history[len - 4];
        if (m.x >= 0) set(0, 12, m.y, m.x, 1.0);
    }
    if (len >= 5) {
        const m = history[len - 5];
        if (m.x >= 0) set(0, 13, m.y, m.x, 1.0);
    }

    // Global input
    // 0: Player color? No, KataGo global input is usually:
    // 0: Prev move was pass?
    // 1: Prev-1 move was pass?
    // ...
    // 5: Komi / 20.0
    // 6: Ko rule?
    // ...
    // Let's check features.py for global input
    // It seems complex. But usually Komi is important.
    // Let's assume a simplified global input or check features.py again.
    
    // From features.py:
    // global_input_data[idx,0] = 1.0 if prev1 was pass
    // global_input_data[idx,1] = 1.0 if prev2 was pass
    // ...
    // global_input_data[idx,5] = rules["whiteKomi"] / 20.0
    // ...
    
    const global_input = new Float32Array(1 * 19).fill(0);
    
    if (len >= 1 && history[len-1].x < 0) global_input[0] = 1.0;
    if (len >= 2 && history[len-2].x < 0) global_input[1] = 1.0;
    if (len >= 3 && history[len-3].x < 0) global_input[2] = 1.0;
    if (len >= 4 && history[len-4].x < 0) global_input[3] = 1.0;
    if (len >= 5 && history[len-5].x < 0) global_input[4] = 1.0;
    
    // Komi (assuming Pla is Black, Komi is for White)
    // If Pla is White, Komi is favorable?
    // KataGo usually takes Komi from White's perspective in global features?
    // features.py: global_input_data[idx,5] = rules["whiteKomi"] / 20.0
    // It seems it's always White Komi.
    global_input[5] = komi / 20.0;
    
    // Other global features (ko rules etc) - leave as 0 for now (implies simple defaults)

    return { bin_input: input, global_input };
}

// --- UI Handling ---
const modelDrop = document.getElementById('model-drop');
const modelInput = document.getElementById('model-input');
const modelStatus = document.getElementById('model-status');

const sgfDrop = document.getElementById('sgf-drop');
const sgfInput = document.getElementById('sgf-input');
const sgfStatus = document.getElementById('sgf-status');

const moveNumberInput = document.getElementById('move-number');
const runBtn = document.getElementById('run-btn');
const outputDiv = document.getElementById('output');
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');

// Drag & Drop Helpers
function setupDropZone(dropZone, input, callback) {
    dropZone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => handleFile(e.target.files[0], callback));
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0], callback);
    });
}

function handleFile(file, callback) {
    if (!file) return;
    callback(file);
}

// Model Loading
setupDropZone(modelDrop, modelInput, async (file) => {
    modelStatus.textContent = `Loading ${file.name}...`;
    try {
        const buffer = await file.arrayBuffer();
        ortSession = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] });
        modelStatus.textContent = `Loaded: ${file.name}`;
        checkReady();
    } catch (e) {
        modelStatus.textContent = `Error: ${e.message}`;
        console.error(e);
    }
});

// SGF Loading
setupDropZone(sgfDrop, sgfInput, async (file) => {
    sgfStatus.textContent = `Loading ${file.name}...`;
    try {
        const text = await file.text();
        const parsed = parseSGF(text);
        boardSize = parsed.size;
        komi = parsed.komi;
        moves = parsed.moves;
        sgfContent = parsed;
        
        sgfStatus.textContent = `Loaded: ${file.name} (${moves.length} moves, Size ${boardSize})`;
        moveNumberInput.max = moves.length;
        moveNumberInput.value = Math.min(moves.length, 10);
        
        drawBoard();
        checkReady();
    } catch (e) {
        sgfStatus.textContent = `Error: ${e.message}`;
        console.error(e);
    }
});

function checkReady() {
    if (ortSession && sgfContent) {
        runBtn.disabled = false;
    }
}

// Drawing
function drawBoard() {
    const size = boardSize;
    const width = canvas.width;
    const height = canvas.height;
    
    // Add margin for coordinates
    const margin = 25;
    const boardWidth = width - 2 * margin;
    // const boardHeight = height - 2 * margin;
    
    const cell = boardWidth / (size - 1);
    
    ctx.clearRect(0, 0, width, height);
    
    // Coordinates
    ctx.fillStyle = '#000';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    const letters = "ABCDEFGHJKLMNOPQRST";
    
    for (let i = 0; i < size; i++) {
        const x = margin + i * cell;
        const y = margin + i * cell;
        
        // Horizontal letters (Top and Bottom)
        if (i < letters.length) {
            ctx.fillText(letters[i], x, margin / 2);
            ctx.fillText(letters[i], x, height - margin / 2);
        }
        
        // Vertical numbers (Left and Right)
        // 1 is at bottom, 19 at top
        const num = size - i;
        ctx.fillText(num, margin / 2, y);
        ctx.fillText(num, width - margin / 2, y);
    }
    
    // Grid
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    
    ctx.beginPath();
    for (let i = 0; i < size; i++) {
        // Vertical lines
        ctx.moveTo(margin + i * cell, margin);
        ctx.lineTo(margin + i * cell, height - margin);
        
        // Horizontal lines
        ctx.moveTo(margin, margin + i * cell);
        ctx.lineTo(width - margin, margin + i * cell);
    }
    ctx.stroke();
    
    // Star points (for 19x19)
    if (size === 19) {
        const stars = [3, 9, 15];
        ctx.fillStyle = '#000';
        for (const x of stars) {
            for (const y of stars) {
                ctx.beginPath();
                ctx.arc(margin + x * cell, margin + y * cell, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
        }
    }

    // Replay moves
    const targetMove = parseInt(moveNumberInput.value);
    const currentBoard = new Board(size);
    
    for (let i = 0; i < targetMove; i++) {
        const m = moves[i];
        if (m.x >= 0) {
            currentBoard.play(m.x, m.y, m.color);
        }
    }
    
    // Draw Stones
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const color = currentBoard.get(x, y);
            if (color !== EMPTY) {
                const cx = margin + x * cell;
                const cy = margin + y * cell;

                ctx.beginPath();
                ctx.arc(cx, cy, cell * 0.45, 0, 2 * Math.PI);
                ctx.fillStyle = color === BLACK ? '#000' : '#fff';
                ctx.fill();
                if (color === WHITE) {
                    ctx.strokeStyle = '#000';
                    ctx.stroke();
                }
            }
        }
    }
    
    // Mark last move
    if (targetMove > 0) {
        const last = moves[targetMove - 1];
        if (last.x >= 0) {
            const cx = margin + last.x * cell;
            const cy = margin + last.y * cell;

            ctx.beginPath();
            ctx.arc(cx, cy, cell * 0.2, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();
        }
    }
    
    return currentBoard;
}

moveNumberInput.addEventListener('input', drawBoard);

// Run Inference
runBtn.addEventListener('click', async () => {
    if (!ortSession || !sgfContent) return;
    
    outputDiv.textContent = "Running inference...";
    
    try {
        const targetMove = parseInt(moveNumberInput.value);
        const currentBoard = new Board(boardSize);
        const history = [];
        
        for (let i = 0; i < targetMove; i++) {
            const m = moves[i];
            if (m.x >= 0) {
                currentBoard.play(m.x, m.y, m.color);
            }
            history.push(m);
        }
        
        // Determine current player
        // If moves list is empty, Black.
        // If last move was Black, White.
        let nextPla = BLACK;
        if (history.length > 0) {
            nextPla = history[history.length - 1].color === BLACK ? WHITE : BLACK;
        }
        
        const { bin_input, global_input } = featurize(currentBoard, history, nextPla);
        
        // Create Tensors
        const binTensor = new ort.Tensor('float32', bin_input, [1, 22, boardSize, boardSize]);
        const globalTensor = new ort.Tensor('float32', global_input, [1, 19]);
        
        // Run
        const feeds = {
            "bin_input": binTensor,
            "global_input": globalTensor
        };
        
        const results = await ortSession.run(feeds);
        
        // Process Results
        const policy = results.policy.data; // Float32Array
        const value = results.value.data;   // Float32Array
        const miscvalue = results.miscvalue.data; // Float32Array
        
        // Winrate
        // value is [win, loss, no_result] logits
        // Softmax
        const expValue = [Math.exp(value[0]), Math.exp(value[1]), Math.exp(value[2])];
        const sumValue = expValue[0] + expValue[1] + expValue[2];
        const winrate = expValue[0] / sumValue;
        
        // Score Lead
        // miscvalue index 2 is lead
        // Multiplier is usually 20.0 for KataGo?
        // In model_pytorch.py: self.lead_multiplier = 20.0
        const lead = miscvalue[2] * 20.0;
        
        // Top Moves
        // policy shape: [batch, num_policy_outputs, moves]
        // We want index 0 (policy)
        // moves = size*size + 1 (pass)
        const numMoves = boardSize * boardSize + 1;
        const policyLogits = policy.subarray(0, numMoves);
        
        // Softmax policy
        let maxLogit = -Infinity;
        for(let i=0; i<numMoves; i++) if(policyLogits[i] > maxLogit) maxLogit = policyLogits[i];
        
        const policyProbs = new Float32Array(numMoves);
        let sumProbs = 0;
        for(let i=0; i<numMoves; i++) {
            policyProbs[i] = Math.exp(policyLogits[i] - maxLogit);
            sumProbs += policyProbs[i];
        }
        for(let i=0; i<numMoves; i++) policyProbs[i] /= sumProbs;
        
        // Sort
        const indices = Array.from({length: numMoves}, (_, i) => i);
        indices.sort((a, b) => policyProbs[b] - policyProbs[a]);
        
        // Display
        let out = `Player: ${nextPla === BLACK ? 'Black' : 'White'}\n`;
        out += `Winrate: ${(winrate * 100).toFixed(2)}%\n`;
        out += `Score Lead: ${lead.toFixed(2)}\n\n`;
        out += `Top Moves:\n`;
        
        const letters = "ABCDEFGHJKLMNOPQRST";
        
        for (let i = 0; i < 10; i++) {
            const idx = indices[i];
            const prob = policyProbs[idx];
            let moveStr = "";
            
            if (idx === boardSize * boardSize) {
                moveStr = "PASS";
            } else {
                const y = Math.floor(idx / boardSize);
                const x = idx % boardSize;
                // GTP coordinates
                moveStr = `${letters[x]}${boardSize - y}`;
            }
            
            out += `${i+1}. ${moveStr.padEnd(5)} ${(prob * 100).toFixed(2)}%\n`;
        }
        
        outputDiv.textContent = out;
        
    } catch (e) {
        outputDiv.textContent = `Error: ${e.message}`;
        console.error(e);
    }
});
