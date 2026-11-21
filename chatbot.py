import os
import time
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langfuse import Langfuse
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import uvicorn

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"), override=True)

# ── Setup ────────────────────────────────────────────────────────────
embedder      = SentenceTransformer("all-MiniLM-L6-v2")
groq_client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
langfuse      = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)
chroma_client = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", os.path.join(BASE, "chromadb"))
)
collection    = chroma_client.get_or_create_collection("mac_studio")
app           = FastAPI()

# ── Faithfulness judge ────────────────────────────────────────────────
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    def load_model(self):          return self._client
    def get_model_name(self) -> str: return "llama-3.3-70b-versatile"
    def generate(self, prompt: str, schema=None) -> str:
        r = self._client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1024,
        )
        return r.choices[0].message.content
    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.5, model=GroqJudge(), include_reason=True, async_mode=False
)

# ── RAG Core ─────────────────────────────────────────────────────────
def rag_query(question: str) -> dict:
    start_total = time.time()

    with langfuse.start_as_current_observation(name="rag_query", input=question):

        # Step 1: Embed question
        with langfuse.start_as_current_observation(name="embedding", as_type="embedding"):
            t0 = time.time()
            query_vector = embedder.encode(question).tolist()
            embed_ms = round((time.time() - t0) * 1000)
            langfuse.update_current_span(
                input={"question": question},
                output={
                    "latency_ms":   embed_ms,
                    "vector_dim":   len(query_vector),
                    "vector_preview": query_vector[:8],
                },
                metadata={"model": "all-MiniLM-L6-v2"}
            )

        # Step 2: Retrieve chunks
        with langfuse.start_as_current_observation(name="retrieval", as_type="retriever"):
            t1 = time.time()
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(3, collection.count())
            )
            retrieve_ms = round((time.time() - t1) * 1000)
            chunks    = results["documents"][0]
            distances = results["distances"][0]
            ids       = results["ids"][0]
            langfuse.update_current_span(
                input={"query": question},
                output={
                    "chunks_retrieved": len(chunks),
                    "latency_ms":       retrieve_ms,
                    "results": [
                        {
                            "rank":     i + 1,
                            "id":       ids[i],
                            "distance": round(distances[i], 4),
                            "content":  chunks[i],
                        }
                        for i in range(len(chunks))
                    ],
                },
                metadata={"collection": "mac_studio", "n_results": len(chunks)}
            )

        # Step 3: Build prompt
        context = "\n\n---\n\n".join(chunks)
        prompt  = f"""You are a helpful assistant for Mac Studio 2025.
Answer ONLY using the context below.
If the answer is not in the context, say:
"I don't have that information in the Mac Studio documentation."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

        # Step 4: Generate answer
        with langfuse.start_as_current_observation(name="generation", as_type="generation"):
            t2 = time.time()
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            gen_ms   = round((time.time() - t2) * 1000)
            answer   = response.choices[0].message.content.strip()
            tokens   = response.usage.total_tokens
            langfuse.update_current_generation(
                input=prompt,
                output=answer,
                model="llama-3.3-70b-versatile",
                model_parameters={"temperature": 0.1, "max_tokens": 500},
                usage_details={
                    "input":  response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total":  response.usage.total_tokens,
                },
                metadata={"latency_ms": gen_ms}
            )

        total_ms = round((time.time() - start_total) * 1000)

        # Step 5: Log to Langfuse
        langfuse.set_current_trace_io(input=question, output=answer)
        langfuse.update_current_span(metadata={
            "embed_ms":    embed_ms,
            "retrieve_ms": retrieve_ms,
            "gen_ms":      gen_ms,
            "total_ms":    total_ms,
            "tokens":      tokens,
            "chunks_used": len(chunks)
        })

    return {
        "answer":      answer,
        "chunks":      chunks,
        "distances":   distances,
        "embed_ms":    embed_ms,
        "retrieve_ms": retrieve_ms,
        "gen_ms":      gen_ms,
        "total_ms":    total_ms,
        "tokens":      tokens
    }

# ── Web UI ───────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mac Studio RAG Chatbot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0a; color: #e0e0e0; height: 100vh;
    display: flex; flex-direction: column;
  }

  /* Header */
  .header {
    background: #1a1a1a; border-bottom: 1px solid #333;
    padding: 16px 24px; display: flex;
    align-items: center; gap: 12px;
  }
  .header-icon {
    width: 36px; height: 36px; background: #0071e3;
    border-radius: 8px; display: flex;
    align-items: center; justify-content: center;
    font-size: 18px;
  }
  .header h1 { font-size: 16px; font-weight: 600; color: #fff; }
  .header p  { font-size: 11px; color: #666; margin-top: 2px; }
  .status {
    margin-left: auto; display: flex;
    align-items: center; gap: 6px;
    font-size: 11px; color: #4ade80;
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  /* Main layout */
  .main { display: flex; flex: 1; overflow: hidden; }

  /* Chat area */
  .chat-area {
    flex: 1; display: flex;
    flex-direction: column; overflow: hidden;
  }
  .messages {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .message { display: flex; gap: 12px; max-width: 85%; }
  .message.user { margin-left: auto; flex-direction: row-reverse; }

  .avatar {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center;
    justify-content: center; font-size: 14px;
    flex-shrink: 0;
  }
  .avatar.bot  { background: #0071e3; }
  .avatar.user { background: #2a2a2a; }

  .bubble {
    padding: 12px 16px; border-radius: 12px;
    font-size: 14px; line-height: 1.6; max-width: 100%;
  }
  .bubble.bot  {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 12px 12px 12px 2px;
  }
  .bubble.user {
    background: #0071e3; color: #fff;
    border-radius: 12px 12px 2px 12px;
  }

  /* Metrics bar */
  .metrics {
    display: flex; gap: 12px; margin-top: 8px;
    flex-wrap: wrap;
  }
  .metric {
    font-size: 10px; color: #666;
    background: #111; border: 1px solid #222;
    padding: 2px 8px; border-radius: 4px;
  }
  .metric span { color: #4ade80; font-weight: 600; }

  /* Retrieved chunks panel */
  .chunks-panel {
    display: flex; gap: 6px;
    margin-top: 8px; flex-wrap: wrap;
  }
  .chunk-badge {
    font-size: 10px; background: #111;
    border: 1px solid #0071e3;
    color: #60a5fa; padding: 2px 8px;
    border-radius: 4px; cursor: pointer;
    transition: background 0.2s;
  }
  .chunk-badge:hover { background: #0a1628; }

  /* Chunk detail */
  .chunk-detail {
    margin-top: 8px; padding: 10px 12px;
    background: #0d1117; border: 1px solid #1e3a5f;
    border-radius: 8px; font-size: 11px;
    color: #94a3b8; line-height: 1.6;
    display: none; max-height: 150px;
    overflow-y: auto;
  }
  .chunk-detail.open { display: block; }

  /* Typing indicator */
  .typing {
    display: flex; gap: 4px;
    align-items: center; padding: 4px 0;
  }
  .typing span {
    width: 6px; height: 6px;
    background: #666; border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-8px); }
  }

  /* Input area */
  .input-area {
    padding: 16px 24px; background: #111;
    border-top: 1px solid #222;
  }
  .input-row {
    display: flex; gap: 10px;
    background: #1a1a1a; border: 1px solid #333;
    border-radius: 12px; padding: 8px 8px 8px 16px;
    align-items: flex-end;
  }
  textarea {
    flex: 1; background: transparent;
    border: none; outline: none; color: #e0e0e0;
    font-size: 14px; resize: none;
    line-height: 1.5; max-height: 120px;
    font-family: inherit;
  }
  button {
    width: 36px; height: 36px;
    background: #0071e3; border: none;
    border-radius: 8px; cursor: pointer;
    display: flex; align-items: center;
    justify-content: center; font-size: 16px;
    transition: background 0.2s; flex-shrink: 0;
  }
  button:hover  { background: #0077ed; }
  button:active { background: #005bb5; }

  /* Sidebar */
  .sidebar {
    width: 280px; background: #111;
    border-left: 1px solid #1e1e1e;
    padding: 16px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 16px;
  }
  .sidebar h3 {
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.1em; color: #555;
    margin-bottom: 8px;
  }

  /* Test cases */
  .test-case {
    padding: 8px 10px; background: #1a1a1a;
    border: 1px solid #2a2a2a; border-radius: 6px;
    font-size: 12px; color: #aaa;
    cursor: pointer; transition: all 0.15s;
    margin-bottom: 6px;
  }
  .test-case:hover {
    border-color: #0071e3; color: #fff;
    background: #0a1628;
  }
  .test-case .tc-id {
    font-size: 9px; color: #555;
    font-family: monospace; margin-bottom: 2px;
  }
  .test-case .tc-cat {
    font-size: 9px; padding: 1px 5px;
    border-radius: 3px; float: right;
    margin-top: -14px;
  }
  .cat-hardware  { background: #0a1e40; color: #60a5fa; }
  .cat-storage   { background: #1a0a40; color: #a78bfa; }
  .cat-software  { background: #0a1e10; color: #4ade80; }
  .cat-oos       { background: #2a0a0a; color: #f87171; }
  .cat-perf      { background: #1e1a0a; color: #fbbf24; }

  /* Stats */
  .stat-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .stat {
    background: #1a1a1a; border: 1px solid #222;
    border-radius: 6px; padding: 8px;
    text-align: center;
  }
  .stat-val {
    font-size: 18px; font-weight: 700;
    color: #fff; display: block;
  }
  .stat-lbl {
    font-size: 9px; color: #555;
    text-transform: uppercase; letter-spacing: 0.08em;
  }

  /* Welcome */
  .welcome {
    text-align: center; padding: 40px 20px;
    color: #444;
  }
  .welcome .icon { font-size: 48px; margin-bottom: 16px; }
  .welcome h2   { color: #666; font-size: 16px; margin-bottom: 8px; }
  .welcome p    { font-size: 13px; line-height: 1.6; }

  /* DeepEval trace panel */
  .deepeval-panel {
    margin-top: 10px; border-radius: 8px;
    border: 1px solid #1e3a5f; overflow: hidden;
  }
  .deepeval-header {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; background: #0a1628;
    cursor: pointer; user-select: none;
    font-size: 11px; font-weight: 600; color: #60a5fa;
  }
  .deepeval-header:hover { background: #0d1e3a; }
  .deepeval-score-badge {
    margin-left: auto; padding: 2px 8px;
    border-radius: 4px; font-size: 11px; font-weight: 700;
  }
  .deepeval-body {
    display: none; padding: 12px;
    background: #050d1a; font-size: 11px;
  }
  .deepeval-body.open { display: block; }
  .deepeval-section { margin-bottom: 12px; }
  .deepeval-section-title {
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.08em; color: #555;
    margin-bottom: 6px;
  }
  .deepeval-reason {
    color: #94a3b8; line-height: 1.6;
    padding: 6px 10px; background: #0a1628;
    border-radius: 4px; border-left: 2px solid #1e3a5f;
  }
  .claim-row {
    display: flex; gap: 8px; align-items: flex-start;
    padding: 5px 8px; border-radius: 4px;
    margin-bottom: 4px; background: #0a1628;
  }
  .claim-icon { flex-shrink: 0; font-size: 12px; }
  .claim-text { color: #cbd5e1; line-height: 1.5; flex: 1; }
  .claim-sub  { color: #64748b; font-size: 10px; margin-top: 2px; }
  .truth-row  {
    padding: 4px 8px; color: #94a3b8;
    border-left: 2px solid #334155;
    margin-bottom: 3px; line-height: 1.5;
  }

  ::-webkit-scrollbar       { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
</head>
<body>

<div class="header">
  <div class="header-icon">🖥️</div>
  <div>
    <h1>Mac Studio 2025 — RAG Chatbot</h1>
    <p>ChromaDB · all-MiniLM-L6-v2 · Llama 3.3 70B · Langfuse</p>
  </div>
  <div class="status"><div class="dot"></div>Live</div>
</div>

<div class="main">

  <!-- Chat -->
  <div class="chat-area">
    <div class="messages" id="messages">
      <div class="welcome">
        <div class="icon">🤖</div>
        <h2>Ask me anything about Mac Studio 2025</h2>
        <p>Try the test cases on the right →<br>
        or type your own question below</p>
      </div>
    </div>

    <div class="input-area">
      <div class="input-row">
        <textarea id="input" rows="1"
          placeholder="Ask about Mac Studio 2025..."
          onkeydown="handleKey(event)"
          oninput="autoResize(this)"></textarea>
        <button onclick="sendMessage()" title="Send">➤</button>
      </div>
    </div>
  </div>

  <!-- Sidebar -->
  <div class="sidebar">

    <!-- Stats -->
    <div>
      <h3>Session Stats</h3>
      <div class="stat-grid">
        <div class="stat">
          <span class="stat-val" id="stat-queries">0</span>
          <span class="stat-lbl">Queries</span>
        </div>
        <div class="stat">
          <span class="stat-val" id="stat-tokens">0</span>
          <span class="stat-lbl">Tokens</span>
        </div>
        <div class="stat">
          <span class="stat-val" id="stat-avgms">-</span>
          <span class="stat-lbl">Avg ms</span>
        </div>
        <div class="stat">
          <span class="stat-val" id="stat-chunks">3</span>
          <span class="stat-lbl">Chunks</span>
        </div>
      </div>
    </div>

    <!-- Test Cases -->
    <div>
      <h3>Test Cases — Golden Dataset</h3>

      <div class="test-case" onclick="runTest(this)"
        data-q="What chip does Mac Studio use?"
        data-expected="M4 Max">
        <div class="tc-id">TC001</div>
        <span class="tc-cat cat-hardware">hardware</span>
        What chip does Mac Studio use?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="How many CPU cores does M4 Max have?"
        data-expected="16-core">
        <div class="tc-id">TC002</div>
        <span class="tc-cat cat-hardware">hardware</span>
        How many CPU cores does M4 Max have?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="What storage options are available?"
        data-expected="512GB">
        <div class="tc-id">TC003</div>
        <span class="tc-cat cat-storage">storage</span>
        What storage options are available?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="How many displays can M3 Ultra support?"
        data-expected="eight">
        <div class="tc-id">TC004</div>
        <span class="tc-cat cat-hardware">hardware</span>
        How many displays can M3 Ultra support?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="What is Apple Intelligence?"
        data-expected="Apple Intelligence">
        <div class="tc-id">TC005</div>
        <span class="tc-cat cat-software">software</span>
        What is Apple Intelligence?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="How much faster is GPU performance?"
        data-expected="2.6x">
        <div class="tc-id">TC006</div>
        <span class="tc-cat cat-perf">performance</span>
        How much faster is GPU performance?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="What is the price of Mac Studio?"
        data-expected="NOT_IN_DOCUMENT">
        <div class="tc-id">TC007 ⚠️</div>
        <span class="tc-cat cat-oos">out-of-scope</span>
        What is the price of Mac Studio?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="What is the design size of Mac Studio?"
        data-expected="7.7-inch">
        <div class="tc-id">TC008</div>
        <span class="tc-cat cat-hardware">hardware</span>
        What is the design size of Mac Studio?
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="Tell me about Mac Studio connectivity ports"
        data-expected="HDMI">
        <div class="tc-id">TC009</div>
        <span class="tc-cat cat-hardware">hardware</span>
        Tell me about connectivity ports
      </div>

      <div class="test-case" onclick="runTest(this)"
        data-q="Does Mac Studio support iPhone?"
        data-expected="NOT_IN_DOCUMENT">
        <div class="tc-id">TC010 ⚠️</div>
        <span class="tc-cat cat-oos">hallucination trap</span>
        Does Mac Studio support iPhone?
      </div>
    </div>

  </div>
</div>

<script>
let totalQueries = 0;
let totalTokens  = 0;
let totalMs      = 0;

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function runTest(el) {
  const q = el.getAttribute('data-q');
  const expected = el.getAttribute('data-expected');
  document.getElementById('input').value = q;
  sendMessage(expected);
}

function addMessage(role, content, meta = null) {
  const msgs = document.getElementById('messages');

  // Remove welcome screen
  const welcome = msgs.querySelector('.welcome');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === 'bot' ? '🤖' : '👤';

  const bubbleWrap = document.createElement('div');
  bubbleWrap.style.maxWidth = '100%';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.textContent = content;
  bubbleWrap.appendChild(bubble);

  // Add metrics if bot message
  if (meta && role === 'bot') {
    const metrics = document.createElement('div');
    metrics.className = 'metrics';
    const fScore  = meta.faithfulness_score;
    const fPassed = meta.faithfulness_passed;
    const fColor  = fPassed ? '#4ade80' : '#f87171';
    const fIcon   = fPassed ? '✅' : '❌';
    metrics.innerHTML = `
      <div class="metric">embed <span>${meta.embed_ms}ms</span></div>
      <div class="metric">retrieve <span>${meta.retrieve_ms}ms</span></div>
      <div class="metric">generate <span>${meta.gen_ms}ms</span></div>
      <div class="metric">total <span>${meta.total_ms}ms</span></div>
      <div class="metric">tokens <span>${meta.tokens}</span></div>
      <div class="metric" style="border-color:${fColor}">
        ${fIcon} faithful <span style="color:${fColor}">${fScore}</span>
      </div>
    `;
    bubbleWrap.appendChild(metrics);

    // ── DeepEval full trace panel ────────────────────────────────────
    if (fScore !== undefined) {
      const panel = document.createElement('div');
      panel.className = 'deepeval-panel';

      const verdicts   = meta.faithfulness_verdicts || [];
      const truths     = meta.faithfulness_truths   || [];
      const passCount  = verdicts.filter(v => v.verdict === 'yes').length;
      const totalClaims = verdicts.length;

      const header = document.createElement('div');
      header.className = 'deepeval-header';
      header.innerHTML = `
        <span>⚖️ DeepEval — Faithfulness Trace</span>
        <span style="color:#64748b;font-weight:400">
          ${passCount}/${totalClaims} claims grounded
        </span>
        <span class="deepeval-score-badge"
          style="background:${fPassed?'#052e16':'#2d0a0a'};color:${fColor}">
          ${fIcon} ${fScore}
        </span>
      `;

      const body = document.createElement('div');
      body.className = 'deepeval-body';

      // Overall reason
      body.innerHTML += `
        <div class="deepeval-section">
          <div class="deepeval-section-title">Overall Verdict</div>
          <div class="deepeval-reason">${meta.faithfulness_reason || '—'}</div>
        </div>
      `;

      // Claims + verdicts
      if (verdicts.length > 0) {
        let claimsHTML = `<div class="deepeval-section">
          <div class="deepeval-section-title">Claims Extracted from Answer (${verdicts.length})</div>`;
        verdicts.forEach(v => {
          const icon = v.verdict === 'yes' ? '✅' : '❌';
          const bg   = v.verdict === 'yes' ? '' : 'background:#1a0808';
          claimsHTML += `
            <div class="claim-row" style="${bg}">
              <span class="claim-icon">${icon}</span>
              <div>
                <div class="claim-text">${v.claim}</div>
                ${v.reason ? `<div class="claim-sub">↳ ${v.reason}</div>` : ''}
              </div>
            </div>`;
        });
        claimsHTML += `</div>`;
        body.innerHTML += claimsHTML;
      }

      // Truths from context
      if (truths.length > 0) {
        let truthsHTML = `<div class="deepeval-section">
          <div class="deepeval-section-title">Truths Extracted from Context (${truths.length})</div>`;
        truths.forEach(t => {
          truthsHTML += `<div class="truth-row">• ${t}</div>`;
        });
        truthsHTML += `</div>`;
        body.innerHTML += truthsHTML;
      }

      header.onclick = () => body.classList.toggle('open');
      panel.appendChild(header);
      panel.appendChild(body);
      bubbleWrap.appendChild(panel);
    }

    // Add chunk badges
    if (meta.chunks && meta.chunks.length > 0) {
      const chunksPanel = document.createElement('div');
      chunksPanel.className = 'chunks-panel';

      meta.chunks.forEach((chunk, i) => {
        const badge = document.createElement('div');
        badge.className = 'chunk-badge';
        badge.textContent = `Chunk ${i} (dist: ${meta.distances[i].toFixed(3)})`;

        const detail = document.createElement('div');
        detail.className = 'chunk-detail';
        detail.textContent = chunk.substring(0, 300) + '...';

        badge.onclick = () => detail.classList.toggle('open');
        chunksPanel.appendChild(badge);
        chunksPanel.appendChild(detail);
      });

      bubbleWrap.appendChild(chunksPanel);
    }

    // Test result if expected provided
    if (meta.expected) {
      const testResult = document.createElement('div');
      const passed = meta.expected === 'NOT_IN_DOCUMENT'
        ? !meta.answer.toLowerCase().includes('price') ||
          meta.answer.toLowerCase().includes("don't have")
        : meta.answer.toLowerCase().includes(meta.expected.toLowerCase());

      testResult.style.cssText = `
        margin-top: 8px; padding: 4px 10px;
        border-radius: 4px; font-size: 11px; font-weight: 600;
        background: ${passed ? '#0a1e10' : '#1e0a0a'};
        border: 1px solid ${passed ? '#4ade80' : '#f87171'};
        color: ${passed ? '#4ade80' : '#f87171'};
      `;
      testResult.textContent = passed
        ? `✅ PASS — "${meta.expected}" found in answer`
        : `❌ FAIL — "${meta.expected}" not found in answer`;
      bubbleWrap.appendChild(testResult);
    }
  }

  div.appendChild(avatar);
  div.appendChild(bubbleWrap);
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;

  return div;
}

function addTyping() {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message bot';
  div.id = 'typing';
  div.innerHTML = `
    <div class="avatar bot">🤖</div>
    <div class="bubble bot">
      <div class="typing">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

async function sendMessage(expected = null) {
  const input = document.getElementById('input');
  const q = input.value.trim();
  if (!q) return;

  input.value = '';
  input.style.height = 'auto';

  addMessage('user', q);
  addTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();

    document.getElementById('typing')?.remove();
    addMessage('bot', data.answer, { ...data, expected });

    // Update stats
    totalQueries++;
    totalTokens += data.tokens;
    totalMs     += data.total_ms;
    document.getElementById('stat-queries').textContent = totalQueries;
    document.getElementById('stat-tokens').textContent  = totalTokens;
    document.getElementById('stat-avgms').textContent   =
      Math.round(totalMs / totalQueries);

  } catch(e) {
    document.getElementById('typing')?.remove();
    addMessage('bot', 'Error: ' + e.message);
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

@app.post("/chat")
async def chat(request: Request):
    body   = await request.json()
    result = rag_query(body["question"])

    # ── faithfulness scoring ──────────────────────────────────────────
    tc = LLMTestCase(
        input=body["question"],
        actual_output=result["answer"],
        retrieval_context=result["chunks"],
    )
    faithfulness_metric.measure(tc)
    score   = round(faithfulness_metric.score, 3)
    passed  = faithfulness_metric.score >= faithfulness_metric.threshold
    reason  = faithfulness_metric.reason or ""
    claims  = faithfulness_metric.claims  or []
    truths  = faithfulness_metric.truths  or []
    verdicts = [
        {"claim": claims[i] if i < len(claims) else "",
         "verdict": v.verdict,
         "reason":  v.reason or ""}
        for i, v in enumerate(faithfulness_metric.verdicts or [])
    ]

    status_icon = "✅" if passed else "❌"
    print(f"\n{'─'*60}")
    print(f"  FAITHFULNESS  {status_icon}  score={score}  ({'PASS' if passed else 'FAIL'})")
    print(f"  Q: {body['question']}")
    print(f"  Reason: {reason}")
    print(f"  Claims ({len(claims)}):")
    for v in verdicts:
        icon = "✅" if v['verdict'] == 'yes' else "❌"
        print(f"    {icon} {v['claim']}")
        if v['reason']:
            print(f"       → {v['reason']}")
    print(f"  Truths from context ({len(truths)}):")
    for t in truths:
        print(f"    • {t}")
    print(f"{'─'*60}")

    result["faithfulness_score"]   = score
    result["faithfulness_passed"]  = passed
    result["faithfulness_reason"]  = reason
    result["faithfulness_claims"]  = claims
    result["faithfulness_truths"]  = truths
    result["faithfulness_verdicts"] = verdicts
    return result

if __name__ == "__main__":
    print("Starting Mac Studio RAG Chatbot...")
    print("Open browser: http://157.180.74.119:8053")
    uvicorn.run(app, host="0.0.0.0", port=8053)
