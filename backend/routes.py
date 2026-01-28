from fastapi import APIRouter, HTTPException, status, Query
from openai import OpenAIError

from models import (
    InterviewAnalysisRequest,
    InterviewGenerationRequest,
    InterviewResponse,
)
from services import (
    EvaluationService,
    InterviewService,
    OpenAIService,
    FriendliService,
    GeminiService
)

router = APIRouter(prefix="/interviews", tags=["면접"])

AI_PROVIDERS = {
    "openai": OpenAIService(),
    "friendli": FriendliService(),
    "gemini": GeminiService()
}


def get_services(provider_name: str):
    # 기본 제공자를 openai로 설정하거나 사용자의 선택에 따름
    provider = AI_PROVIDERS.get(provider_name.lower(), AI_PROVIDERS["openai"])
    return (
        InterviewService(ai_service=provider),
        EvaluationService(ai_service=provider)
    )


@router.post(
    "/generations",
    response_model=InterviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="모의 면접 데이터 생성 및 평가 실행",
    description="""
지정한 직무에 대해 데모용 면접 데이터를 완성 형태로 생성하고 즉시 평가까지 수행합니다.

이 엔드포인트가 하는 일:
- 지정한 직무에 대해 여러 명의 지원자 답변(기본: 3개)을 LLM으로 생성
- 이상적인 기준 답변(전문가 레퍼런스 답변) 1개 생성
- 지원자 간 교차 분석 리포트 생성(공통점, 차이점, 추천 사항)
- 기준 답변 대비 정량 지표 계산:
  - 임베딩 기반 코사인 유사도
  - ROUGE 유사 점수
- 종합 점수와 등급 산출
- 최고 점수 지원자 1명을 선택하고 간단한 채용 근거 생성

사용 예:
- 수동 입력 없이 리뷰어에게 보여주는 라이브 데모
- 전체 파이프라인이 끝까지 정상 동작하는지 빠른 회귀 테스트
- UI 스크린샷 및 README 문서용 샘플 데이터 생성

참고:
- 외부 AI 서비스를 호출하므로 로컬 전용 엔드포인트보다 느릴 수 있습니다.
- 선택한 제공자의 API 키가 없거나 유효하지 않으면 요청이 실패합니다.
""",
    responses={
        201: {
            "description": "면접 데이터 생성 및 분석이 성공적으로 완료되었습니다.",
        },
        422: {
            "description": "검증 오류 또는 AI 생성 단계에서 예상치 못한 응답 형태가 반환되었습니다.",
        },
        502: {
            "description": "콘텐츠 생성 또는 분석 중 상위 AI 제공자 오류가 발생했습니다.",
        },
    },
)
def create_interview_data(
    req: InterviewGenerationRequest,
    provider: str = Query("openai", description="사용할 AI 엔진: openai, friendli, gemini")
):
    int_service, eval_service = get_services(provider)
    try:
        # 1. AI를 통해 질문, 답변셋, 기준 답변 생성
        generated_data = int_service.generate_content(req.job_position)

        # 2. 생성된 데이터를 평가 서비스로 전달하여 분석 리포트 및 점수 생성
        # 여기서 question 데이터를 명시적으로 넘겨주어야 프론트엔드에서 확인 가능합니다.
        return eval_service.process_analysis(
            transcripts=generated_data["transcripts"],
            reference=generated_data["reference"],
            position=req.job_position,
            question=generated_data.get("question"),
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"예상치 못한 AI 응답 형식입니다(키 누락): {str(e)}",
        )
    except (OpenAIError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{provider}에서 생성/분석 중 오류가 발생했습니다: {str(e)}",
        )


@router.post(
    "/analyses",
    response_model=InterviewResponse,
    summary="사용자 입력 면접 답변을 기준 답변과 비교 분석",
    description="""
사용자가 제공한 면접 답변(지원자 답변)을 전문가 기준 답변과 비교하여 평가합니다.

이 엔드포인트가 하는 일:
- 1개 이상의 지원자 답변과 필수 기준 답변을 입력으로 받음
- 구조화된 교차 분석 리포트 생성:
  - 지원자 전반의 공통 경향
  - 주요 차이점 및 강점/약점
  - 실행 가능한 개선/추천 사항
- 생성된 리포트와 기준 답변 간 정량 지표 계산:
  - 코사인 유사도(임베딩)
  - ROUGE 유사 점수
- 종합 점수 및 등급 산출
- 지원자별 개별 점수 산출 후 최고 지원자 선택
- 선택된 지원자에 대한 간단한 채용 근거 생성

사용 예:
- 실제 면접 준비: 여러 답변을 목표 기준 답변과 비교
- LLM이 만든 리포트가 정답(ground-truth)과 얼마나 일치하는지 측정
- 숫자 점수 + 정성 출력으로 설명 가능한 평가 데모

참고:
- 최소 1개 이상의 답변이 필요합니다.
- 리포트 생성 및 채용 근거 생성에 LLM을 사용하므로 지연/비용이 발생합니다.
""",
    responses={
        200: {
            "description": "분석이 성공적으로 완료되었습니다.",
        },
        400: {
            "description": "잘못된 요청(예: 면접 답변 누락).",
        },
        502: {
            "description": "리포트 또는 채용 근거 생성 중 상위 AI 제공자 오류가 발생했습니다.",
        },
    },
)
def analyze_interviews(
    req: InterviewAnalysisRequest,
    provider: str = Query("openai", description="사용할 AI 엔진: openai, friendli, gemini")
):
    if not req.transcripts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="최소 하나 이상의 면접 답변이 필요합니다.",
        )

    _, eval_service = get_services(provider)
    try:
        return eval_service.process_analysis(
            transcripts=req.transcripts,
            reference=req.reference,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except (OpenAIError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{provider}에서 분석 중 오류가 발생했습니다: {str(e)}",
        )