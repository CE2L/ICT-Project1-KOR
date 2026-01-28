from typing import List, Optional
from pydantic import BaseModel, Field

class InterviewAnalysisRequest(BaseModel):
    transcripts: List[str] = Field(..., min_length=1)
    reference: str = Field(..., min_length=1)

class InterviewGenerationRequest(BaseModel):
    job_position: str = Field(default="프론트엔드 개발자")
    num_candidates: int = Field(default=3, ge=1, le=10)

class CandidateScore(BaseModel):
    candidate_number: int
    cosine_score: float
    rouge_score: float
    overall_score: float
    grade: str

class HireDecision(BaseModel):
    selected_candidate: int
    reason: str
    scores: List[CandidateScore]

class InterviewResponse(BaseModel):
    question: Optional[str] = None  # 질문 필드 명시
    report: str
    score: float
    cosine_score: float
    rouge_score: float
    grade: str
    iterations: List[dict]
    transcripts: Optional[List[str]] = None
    reference: Optional[str] = None
    hire_decision: Optional[HireDecision] = None
    ai_provider: Optional[str] = None