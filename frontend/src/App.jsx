import { useState, useRef, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { Send, FileText, Upload, AlertCircle, Trash2 } from 'lucide-react'

const API = 'http://localhost:8000'
const SESSION_ID = 'session-' + Math.random().toString(36).slice(2, 9)

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [indexReady, setIndexReady] = useState(false)
  const [ingestedFiles, setIngestedFiles] = useState([])
  const [pendingFiles, setPendingFiles] = useState([])
  const chatEndRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    axios.get(`${API}/health`).then(r => setIndexReady(r.data.index_ready)).catch(() => {})
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const onDrop = useCallback(accepted => {
    setPendingFiles(prev => [...prev, ...accepted])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
  })

  const handleIngest = async () => {
    if (!pendingFiles.length) return
    setIngesting(true)
    try {
      const form = new FormData()
      pendingFiles.forEach(f => form.append('files', f))
      const res = await axios.post(`${API}/ingest`, form)
      const { files, chunks_indexed } = res.data
      setIngestedFiles(prev => [
        ...prev,
        ...files.map(name => ({ name, chunks: Math.round(chunks_indexed / files.length) }))
      ])
      setIndexReady(true)
      setPendingFiles([])
    } catch (e) {
      alert(e.response?.data?.detail || 'Ingestion failed')
    } finally {
      setIngesting(false)
    }
  }

  const handleClearSession = async () => {
    await axios.delete(`${API}/session/${SESSION_ID}`).catch(() => {})
    setMessages([])
  }

  const handleSend = async () => {
    const q = input.trim()
    if (!q || loading) return
    if (!indexReady) { alert('Upload and index documents first.'); return }

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setInput('')
    setLoading(true)

    // Add placeholder assistant message we'll stream into
    const assistantIdx = messages.length + 1
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      citations: [],
      confident: true,
      streaming: true,
    }])

    try {
      const response = await fetch(`${API}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, session_id: SESSION_ID }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        let eventType = null
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && eventType) {
            const data = JSON.parse(line.slice(6))

            if (eventType === 'citations') {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1
                  ? { ...m, citations: data.citations, confident: data.confident, condensed_query: data.condensed_query }
                  : m
              ))
            } else if (eventType === 'token') {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1
                  ? { ...m, content: m.content + data.token }
                  : m
              ))
            } else if (eventType === 'done') {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1 ? { ...m, streaming: false } : m
              ))
            } else if (eventType === 'error') {
              setMessages(prev => prev.map((m, i) =>
                i === prev.length - 1
                  ? { ...m, content: 'Error: ' + data.message, streaming: false }
                  : m
              ))
            }
            eventType = null
          }
        }
      }
    } catch (e) {
      setMessages(prev => prev.map((m, i) =>
        i === prev.length - 1
          ? { ...m, content: 'Connection error: ' + e.message, streaming: false }
          : m
      ))
    } finally {
      setLoading(false)
      setMessages(prev => prev.map((m, i) =>
        i === prev.length - 1 ? { ...m, streaming: false } : m
      ))
    }
  }

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="app">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="logo">
          ContextIQ
          <span>Enterprise Document Assistant</span>
        </div>

        <div className="index-status">
          <div className={`status-dot ${indexReady ? 'ready' : 'not-ready'}`} />
          {indexReady ? `Index ready · ${ingestedFiles.length} file(s)` : 'No index — upload docs'}
        </div>

        <div {...getRootProps()} className={`drop-zone ${isDragActive ? 'active' : ''}`}>
          <input {...getInputProps()} />
          <Upload size={20} style={{ display: 'block', margin: '0 auto 8px' }} />
          {isDragActive ? 'Drop here…' : 'Drop PDFs, DOCX, CSV, XLSX\nor click to browse'}
        </div>

        {pendingFiles.length > 0 && (
          <div className="file-list">
            {pendingFiles.map((f, i) => (
              <div key={i} className="file-item">
                <FileText size={13} />
                <span className="name">{f.name}</span>
              </div>
            ))}
            <button className="btn btn-primary" onClick={handleIngest} disabled={ingesting}>
              {ingesting ? 'Indexing…' : `Index ${pendingFiles.length} file(s)`}
            </button>
          </div>
        )}

        {ingestedFiles.length > 0 && (
          <div className="file-list">
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Indexed</div>
            {ingestedFiles.map((f, i) => (
              <div key={i} className="file-item">
                <FileText size={13} />
                <span className="name">{f.name}</span>
                <span className="chunks">{f.chunks}c</span>
              </div>
            ))}
          </div>
        )}

        {messages.length > 0 && (
          <button className="btn btn-ghost" onClick={handleClearSession} style={{ marginTop: 'auto' }}>
            <Trash2 size={13} /> Clear chat
          </button>
        )}
      </div>

      {/* Main */}
      <div className="main">
        <div className="chat-area">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Ask anything about your documents</h2>
              <p>Upload PDFs, DOCX, CSV or Excel files — ContextIQ indexes them<br />and answers with exact source citations.</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="bubble">
                {msg.content}
                {msg.streaming && <span className="cursor">▋</span>}
              </div>

              {msg.role === 'assistant' && msg.condensed_query && msg.condensed_query !== messages[i - 1]?.content && (
                <div className="condensed-query">
                  ↳ searched: <em>{msg.condensed_query}</em>
                </div>
              )}

              {msg.role === 'assistant' && msg.citations?.length > 0 && (
                <div className="citations">
                  {msg.citations.map((c, j) => (
                    <div key={j} className="citation-tag">
                      <span className="doc">[{j + 1}]</span>
                      {c.source}{c.page ? ` · p${c.page}` : ' · table'}
                    </div>
                  ))}
                </div>
              )}

              {msg.role === 'assistant' && !msg.confident && !msg.streaming && (
                <div className="insufficient">
                  <AlertCircle size={12} /> Low confidence — answer may be incomplete
                </div>
              )}
            </div>
          ))}

          <div ref={chatEndRef} />
        </div>

        <div className="input-bar">
          <textarea
            rows={1}
            placeholder="Ask a question about your documents… (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button className="send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
