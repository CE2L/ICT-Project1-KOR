import os
import re
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from models import CandidateScore, HireDecision

class BaseAIService(ABC):
    @abstractmethod
    def fetch_chat_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

class OpenAIService(BaseAIService):
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY") or "DUMMY_FOR_TESTS"
        self.client = OpenAI(api_key=api_key)

    def fetch_chat_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "당신은 IT 기술 면접 전문가입니다. 질문은 변별력 있게 깊이 있는 내용을 다루며, 답변은 실무적인 통찰력이 드러나도록 한국어로 상세히 작성하세요."},
                      {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def get_provider_name(self) -> str:
        return "OpenAI (gpt-4o-mini)"

class FriendliService(BaseAIService):
    def __init__(self):
        api_key = os.environ.get("FRIENDLI_API_KEY")
        self.has_key = bool(api_key)
        if self.has_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://inference.friendli.ai/v1"
            )

    def fetch_chat_completion(self, prompt: str) -> str:
        if not self.has_key:
            raise ValueError("Friendli API 키가 설정되지 않았습니다.")
        response = self.client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt + "\n모든 답변은 한국어로 아주 상세하게 작성해줘."}]
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> List[float]:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return [0.0] * 1536
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def get_provider_name(self) -> str:
        return "Friendli AI (meta-llama-3.1-8b-instruct)"

class GeminiService(BaseAIService):
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        self.has_key = bool(api_key)
        if self.has_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
            self.genai = genai

    def fetch_chat_completion(self, prompt: str) -> str:
        if not self.has_key:
            raise ValueError("Gemini API 키가 설정되지 않았습니다.")
        response = self.model.generate_content(prompt + "\n모든 답변은 한국어로 전문적이고 길게 작성해줘.")
        return response.text

    def get_embedding(self, text: str) -> List[float]:
        if self.has_key:
            try:
                result = self.genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                return result['embedding']
            except:
                pass
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            return [0.0] * 1536
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def get_provider_name(self) -> str:
        return "Google Gemini (gemini-2.0-flash-lite)"

class InterviewService:
    def __init__(self, ai_service: BaseAIService):
        self.ai = ai_service

    def generate_content(self, job_position: str):
        prompt = f"""
        다음 직무에 대해 변별력이 높은 고난도 면접 데이터를 한국어로 생성하세요: {job_position}.
        각 답변은 최소 5문장 이상으로 구성하며, 전문 용어를 적절히 사용하여 구체적으로 작성하세요.

        [QUESTION] 
        (해당 직무의 시니어 급에서 다룰 법한 심화 기술 면접 질문 하나만 작성)

        [CANDIDATE 1] 
        (최고 수준의 답변. 기술적 원리, 장단점 비교, 성능 최적화 관점에서의 고려 사항이 포함된 논리적이고 매우 상세한 한국어 답변)

        [CANDIDATE 2] 
        (중급 수준의 답변. 핵심 개념은 명확히 파악하고 있으나, 대규모 아키텍처나 성능 튜닝에 대한 깊은 통찰력은 다소 부족한 한국어 답변)

        [CANDIDATE 3] 
        (초급 수준의 답변. 기본적인 정의 위주로 설명하며 실무적 응용 능력이 다소 떨어지는 한국어 답변)

        [REFERENCE] 
        (최고 수준의 전문가가 제시하는 모범 답안. 최신 트렌드와 베스트 프렉티스를 포함하여 가장 완성도 높은 본문만 작성하세요.)
        """
        raw_text = self.ai.fetch_chat_completion(prompt)

        q_match = re.search(r"\[QUESTION\](.*?)(\[CANDIDATE 1\]|$)", raw_text, re.S)
        c1 = re.search(r"\[CANDIDATE 1\](.*?)(\[CANDIDATE 2\]|$)", raw_text, re.S)
        c2 = re.search(r"\[CANDIDATE 2\](.*?)(\[CANDIDATE 3\]|$)", raw_text, re.S)
        c3 = re.search(r"\[CANDIDATE 3\](.*?)(\[REFERENCE\]|$)", raw_text, re.S)
        ref = re.search(r"\[REFERENCE\](.*?)$", raw_text, re.S)

        question = q_match.group(1).strip() if q_match else f"{job_position} 심화 기술 면접 질문입니다."
        transcripts = [
            c1.group(1).strip() if c1 else f"{job_position} 전문가 답변 데이터 생성 중...",
            c2.group(1).strip() if c2 else f"{job_position} 실무자 답변 데이터 생성 중...",
            c3.group(1).strip() if c3 else f"{job_position} 주니어 답변 데이터 생성 중..."
        ]
        reference = ref.group(1).strip() if ref else f"{job_position} 모범 답안 생성 중..."

        return {
            "question": question,
            "transcripts": transcripts,
            "reference": reference
        }

class EvaluationService:
    def __init__(self, ai_service: BaseAIService):
        self.ai = ai_service

    def cosine_sim(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        res = dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0.0
        return 1.0 if abs(res - 1.0) < 1e-12 else res

    def calculate_rouge(self, candidate: str, reference: str) -> float:
        cand = re.findall(r"\w+", candidate.lower())
        ref = re.findall(r"\w+", reference.lower())
        if not cand or not ref:
            return 0.0
        overlap = len(set(cand) & set(ref))
        return overlap / len(set(ref))

    def calculate_grade(self, score: float) -> str:
        if score >= 0.9: return "A"
        if score >= 0.8: return "B"
        if score >= 0.7: return "C"
        return "D"

    def generate_cross_analysis(self, transcripts: List[str], reference: str, position: str) -> str:
        prompt = f"""
        당신은 기술 심사위원입니다. {position} 직무 지원자 {len(transcripts)}명의 답변을 정밀 분석하세요.

        전문가 기준 답변: {reference}

        지원자 답변 목록:
        {chr(10).join([f"면접자 {i+1}: {t}" for i, t in enumerate(transcripts)])}

        기술적 깊이, 아키텍처 이해도, 성능 최적화 관점을 포함하여 한국어로 리포트를 작성하세요:
        1. 공통 강점 및 기술적 특징
        2. 주요 누락 사항 및 공통 약점
        3. 변별력을 가르는 핵심적 차이점
        4. 전문가 기준 대비 정렬도 및 신뢰성 분석
        5. 실무 투입 시나리오별 채용 전략 권고
        """
        return self.ai.fetch_chat_completion(prompt)

    def get_hire_decision(self, transcripts: List[str], reference: str) -> HireDecision:
        sims = []
        ref_vec = self.ai.get_embedding(reference)
        for t in transcripts:
            cand_vec = self.ai.get_embedding(t)
            sims.append(self.cosine_sim(cand_vec, ref_vec))

        best_index = sims.index(max(sims))
        candidate_scores = []
        for i, (t, sim) in enumerate(zip(transcripts, sims), start=1):
            rouge = self.calculate_rouge(t, reference)
            overall = (sim + rouge) / 2
            candidate_scores.append(CandidateScore(
                candidate_number=i, cosine_score=sim, rouge_score=rouge,
                overall_score=overall, grade=self.calculate_grade(overall)
            ))

        prompt = f"""
        당신은 CTO입니다. 다음 지원자들의 답변을 기술 면접 기준과 대조하여 최종 채용 의견을 작성하세요.
        기준 답변: {reference}
        지원자별 답변:
        {chr(10).join([f"면접자 {i+1}: {t}" for i, t in enumerate(transcripts)])}

        면접자 {best_index + 1}의 답변이 왜 압도적인지, 특히 어떤 부분에서 실무적 통찰력이 돋보이는지 한국어로 상세히 기술하세요.
        """
        explanation = self.ai.fetch_chat_completion(prompt)

        return HireDecision(
            selected_candidate=best_index + 1,
            scores=candidate_scores,
            reason=explanation
        )

    def process_analysis(self, transcripts, reference, position="Unknown", question=None):
        decision = self.get_hire_decision(transcripts, reference)
        cross_analysis = self.generate_cross_analysis(transcripts, reference, position)

        return {
            "question": question,
            "report": cross_analysis,
            "score": sum(s.overall_score for s in decision.scores) / len(decision.scores),
            "cosine_score": decision.scores[decision.selected_candidate - 1].cosine_score,
            "rouge_score": decision.scores[decision.selected_candidate - 1].rouge_score,
            "grade": decision.scores[decision.selected_candidate - 1].grade,
            "iterations": [],
            "transcripts": transcripts,
            "reference": reference,
            "hire_decision": decision,
            "ai_provider": self.ai.get_provider_name()
        }