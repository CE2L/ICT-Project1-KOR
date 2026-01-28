import React, { useEffect, useMemo, useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [transcripts, setTranscripts] = useState(["", "", ""]);
  const [reference, setReference] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [jobPosition, setJobPosition] = useState("프론트엔드 개발자");
  const [provider, setProvider] = useState("openai");
  const [question, setQuestion] = useState("");
  const [levels, setLevels] = useState([]);

  const textareaRefs = useRef([]);
  const referenceRef = useRef(null);

  const defaultApi = useMemo(() => {
    const envUrl = import.meta.env.VITE_API_URL;
    if (envUrl && String(envUrl).trim()) return String(envUrl).trim();
    return "http://4.230.16.126:8012";
  }, []);

  const [apiUrl, setApiUrl] = useState(defaultApi);

  useEffect(() => {
    document.title = "AI 교차 면접 분석기";
  }, []);

  const baseUrl = useMemo(() => {
    let v = String(apiUrl || "").trim();
    if (v.endsWith("/")) v = v.slice(0, -1);
    return v;
  }, [apiUrl]);

  const autoResizeTextarea = (element) => {
    if (!element) return;
    element.style.height = "auto";
    const newHeight = Math.min(Math.max(element.scrollHeight, 100), 400);
    element.style.height = newHeight + "px";
  };

  useEffect(() => {
    textareaRefs.current.forEach(autoResizeTextarea);
    autoResizeTextarea(referenceRef.current);
  }, [transcripts, reference]);

  const handleAutoGenerate = async () => {
    setLoading(true);
    setResult(null);
    setQuestion("");
    setLevels([]);
    try {
      const response = await axios.post(`${baseUrl}/interviews/generations?provider=${provider}`, {
        job_position: jobPosition,
        num_candidates: 3
      });

      if (response.data?.question) {
        setQuestion(response.data.question);
      }
      if (response.data?.transcripts) {
        setTranscripts(response.data.transcripts);
      }
      if (response.data?.reference) {
        setReference(response.data.reference);
      }
      if (response.data?.levels) {
        setLevels(response.data.levels);
      }

      setResult(response.data);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "자동 생성 중 오류가 발생했습니다.";
      alert(typeof errorMsg === "object" ? JSON.stringify(errorMsg) : errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    const validTranscripts = transcripts.filter(t => t && t.trim());
    if (validTranscripts.length === 0) {
      alert("최소 하나 이상의 면접 답변을 입력해주세요.");
      return;
    }
    if (!reference || !reference.trim()) {
      alert("전문가 기준 답변을 입력해주세요.");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${baseUrl}/interviews/analyses?provider=${provider}`, {
        transcripts: validTranscripts,
        reference: reference.trim()
      });
      setResult(response.data);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "분석 중 오류가 발생했습니다.";
      alert(typeof errorMsg === "object" ? JSON.stringify(errorMsg) : errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const updateTranscript = (index, value) => {
    const newTranscripts = [...transcripts];
    newTranscripts[index] = value;
    setTranscripts(newTranscripts);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>AI 교차 면접 분석기</h1>
        <p>LLM 기반 실시간 면접 분석 및 성과 평가 시스템</p>
      </header>

      <section className="auto-section">
        <h2>API 기본 주소</h2>
        <input
          type="text"
          value={apiUrl}
          onChange={e => setApiUrl(e.target.value)}
          placeholder="http://4.230.16.126:8012"
          className="job-input"
        />
      </section>

      <section className="auto-section">
        <h2>AI 자동 생성 모드</h2>
        <p>직무를 입력하면 AI가 면접 질문과 답변을 자동 생성하고 분석합니다</p>

        <div className="auto-controls">
          <input
            type="text"
            value={jobPosition}
            onChange={e => setJobPosition(e.target.value)}
            placeholder="직무를 입력하세요 (예: 백엔드 개발자, 데이터 엔지니어)"
            className="job-input"
          />
          <select
            value={provider}
            onChange={e => setProvider(e.target.value)}
            className="provider-select"
          >
            <option value="openai">OpenAI (gpt-4o-mini)</option>
            <option value="friendli">Friendli AI (llama-3.1-8b)</option>
            <option value="gemini">Google Gemini (2.0-flash)</option>
          </select>
          <button onClick={handleAutoGenerate} disabled={loading} className="btn-auto">
            {loading ? "생성 중..." : "자동 생성 및 분석"}
          </button>
        </div>
      </section>

      <div className="divider">또는 수동 입력</div>

      <section className="input-section">
        <div className="input-group">
          <h2>면접 질문</h2>
          {(question || result?.question) && (
            <div className="question-display">
              <p>{question || result.question}</p>
            </div>
          )}
          <h2>면접 답변</h2>
          {transcripts.map((transcript, index) => (
            <div key={index} className="input-item">
              <label>
                면접자 {index + 1}
                {levels[index] && <span className="level-badge">{levels[index]}</span>}
              </label>
              <textarea
                ref={el => textareaRefs.current[index] = el}
                value={transcript}
                onChange={e => {
                  updateTranscript(index, e.target.value);
                  autoResizeTextarea(e.target);
                }}
                placeholder={`면접자 ${index + 1}의 답변을 입력하세요...`}
                rows={4}
              />
            </div>
          ))}
        </div>

        <div className="input-group">
          <h2>전문가 기준 답변</h2>
          <textarea
            ref={referenceRef}
            value={reference}
            onChange={e => {
              setReference(e.target.value);
              autoResizeTextarea(e.target);
            }}
            placeholder="전문가가 기대하는 이상적인 답변을 입력하세요..."
            rows={4}
          />
        </div>

        <button onClick={handleAnalyze} disabled={loading} className="btn-analyze">
          {loading ? "분석 중..." : "분석 실행"}
        </button>
      </section>

      {result && result.transcripts && result.transcripts.length > 0 && (
        <section className="generated-section">
          <h2>AI 생성 면접 내용</h2>
          {result.transcripts.map((transcript, index) => (
            <div key={index} className="generated-item">
              <h3>
                면접자 {index + 1}
                {result.levels && result.levels[index] && (
                  <span className="level-badge">{result.levels[index]}</span>
                )}
              </h3>
              <p>{transcript || "(내용 없음)"}</p>
            </div>
          ))}
          {result.reference && (
            <div className="generated-item reference">
              <h3>전문가 기준 답변 (중급 수준)</h3>
              <p>{result.reference}</p>
            </div>
          )}
        </section>
      )}

      {result && (
        <section className="result-section">
          {result.ai_provider && (
            <div className="ai-provider-badge">
              <span className="provider-text">AI 제공자: {result.ai_provider}</span>
            </div>
          )}
          <h2>분석 결과</h2>
          <div className="score-cards">
            <div className="score-card" style={{ backgroundColor: "#7C3AED" }}>
              <h3>코사인 유사도</h3>
              <div className="score">{(result.cosine_score * 100).toFixed(1)}%</div>
            </div>
            <div className="score-card" style={{ backgroundColor: "#EF4444" }}>
              <h3>ROUGE 점수</h3>
              <div className="score">{(result.rouge_score * 100).toFixed(1)}%</div>
            </div>
            <div className="score-card" style={{ backgroundColor: "#10B981" }}>
              <h3>종합 점수</h3>
              <div className="score">{(result.score * 100).toFixed(1)}%</div>
              <div className="grade">{result.grade}</div>
            </div>
          </div>
        </section>
      )}

      {result && result.hire_decision && (
        <section className="hire-section">
          <h2>채용 판단</h2>
          <div className="hire-decision">
            <div className="selected-candidate">
              <h3>선정된 면접자: #{result.hire_decision.selected_candidate}</h3>
              <div className="hire-badge">채용</div>
            </div>
            <div className="hire-reason">
              <h4>선정 이유</h4>
              <p>{result.hire_decision.reason}</p>
            </div>
          </div>

          <div className="candidate-scores">
            <h3>면접자별 점수</h3>
            <div className="scores-grid">
              {result.hire_decision.scores.map(score => (
                <div
                  key={score.candidate_number}
                  className={`candidate-score-card ${score.candidate_number === result.hire_decision.selected_candidate ? "selected" : ""}`}
                >
                  <h4>
                    면접자 {score.candidate_number}
                    {result.levels && result.levels[score.candidate_number - 1] && (
                      <span className="level-badge">{result.levels[score.candidate_number - 1]}</span>
                    )}
                  </h4>
                  <div className="score-details">
                    <div className="score-item">
                      <span>코사인 유사도:</span>
                      <span>{(score.cosine_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="score-item">
                      <span>ROUGE 점수:</span>
                      <span>{(score.rouge_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="score-item overall">
                      <span>종합:</span>
                      <span>{(score.overall_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="grade-badge">{score.grade}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {result && (
        <section className="report-section">
          <h2>교차 분석 리포트</h2>
          <div className="report">
            <div className="report-content">
              {String(result.report || "").split("\n").map((line, idx) => (
                <p key={idx}>{line}</p>
              ))}
            </div>
          </div>
        </section>
      )}

      <footer>
        <p>LLM 성능 평가 시스템</p>
      </footer>
    </div>
  );
}

export default App;