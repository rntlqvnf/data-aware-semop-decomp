# core/schemas.py

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# =================================================================
# Action Catalog Schemas
# catalog/ 디렉토리의 YAML 파일을 파싱하여 이 구조로 변환합니다.
# =================================================================

class Metrics(BaseModel):
    """Action 구현체의 예상 성능 지표"""
    latency_ms: float
    cost_usd_per_image: float
    # Field를 사용하여 0.0과 1.0 사이의 값만 받도록 강제
    accuracy: float = Field(..., ge=0.0, le=1.0)
    vram_mb: int

class Implementation(BaseModel):
    """Action을 수행하는 구체적인 방법(모델)"""
    model_name: str
    source: str # e.g., 'local_pytorch', 'azure_openai'
    metrics: Metrics

class ActionInputOutput(BaseModel):
    """Action의 입력 또는 출력 명세"""
    name: str
    type: str
    optional: bool = False

class Action(BaseModel):
    """카탈로그에 정의된 단일 Action의 전체 명세"""
    name: str
    description: str
    modality: str
    inputs: List[ActionInputOutput]
    outputs: List[ActionInputOutput]
    implementations: List[Implementation]


# =================================================================
# Plan & Execution Schemas
# Planner가 생성하고 Executor가 실행하는 데이터 구조입니다.
# =================================================================

class Strategy(BaseModel):
    """LLM이 생성한 추상적인 계획(Action의 순서)"""
    name: str
    description: str
    action_names: List[str]

class PlanStep(BaseModel):
    """실행 계획(Plan)을 구성하는 단일 단계"""
    step_id: str
    action_name: str
    implementation_name: str
    # 입력으로 사용할 변수 이름들을 딕셔너리로 명시
    inputs: Dict[str, Any] # e.g., {'image_object': 'var_image_decoded'}
    # 이 단계의 출력을 저장할 새로운 변수 이름들을 명시
    outputs: Dict[str, str] # e.g., {'clarity_score': 'var_clarity_of_image'}

class Plan(BaseModel):
    """하나의 완성된 실행 계획"""
    name: str # e.g., 'Plan-A-Fast-Local'
    strategy_name: str # 어떤 상위 전략에서 생성되었는지
    steps: List[PlanStep]
    total_estimated_metrics: Optional[Metrics] = None # Planner가 계산한 전체 예상 성능

class ExecutionStatus(str, Enum):
    """실행 결과 상태"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class ExecutionTrace(BaseModel):
    """단일 데이터 아이템의 실행 결과 기록 (KnowledgeBase 학습에 사용)"""
    data_id: str
    plan_used: str
    status: ExecutionStatus
    input_features: Dict[str, Any] # 라우팅에 사용된 데이터 특징
    actual_metrics: Metrics # 실제 측정된 성능
    output: Optional[Any] = None
    error_message: Optional[str] = None


# =================================================================
# User Input Schema
# =================================================================

class Constraints(BaseModel):
    """사용자가 main.py에 전달하는 제약 조건"""
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_accuracy: Optional[float] = None