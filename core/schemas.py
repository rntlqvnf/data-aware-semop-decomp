# core/schemas.py

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

# =================================================================
# Action Catalog Schemas
# =================================================================

class Metrics(BaseModel):
    """Action 구현체의 예상 성능 지표"""
    latency_ms: float
    cost_usd_per_image: float
    accuracy: float = Field(..., ge=0.0, le=1.0)
    vram_mb: int

class Implementation(BaseModel):
    """Action을 수행하는 구체적인 방법(모델)"""
    implementation_name: str
    source: str
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
# =================================================================

class PlanStep(BaseModel):
    """실행 계획(Plan)을 구성하는 단일 단계"""
    step_id: str
    action_name: str
    implementation_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, str]

class Plan(BaseModel):
    """하나의 완성된 실행 계획"""
    name: str
    strategy_name: str
    steps: List[PlanStep]
    total_estimated_metrics: Optional[Metrics] = None

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
    input_features: Dict[str, Any]
    actual_metrics: Metrics
    output: Optional[Any] = None
    error_message: Optional[str] = None

# =================================================================
# Strategy & User Input Schemas
# =================================================================

class Strategy(BaseModel):
    """LLM이 생성한 추상적인 계획(Action의 순서)"""
    name: str
    description: str
    # 💡 핵심 수정 부분: 'action_names'를 'plan'으로 변경하고,
    # 그 타입을 LLM이 생성하는 유연한 구조에 맞게 List[Dict]로 정의합니다.
    plan: List[Dict[str, Any]]
    
    # 'return'은 Python의 예약어이므로, alias를 사용하여 필드 이름을 매핑합니다.
    return_val: str = Field(..., alias='return')


class Constraints(BaseModel):
    """사용자가 main.py에 전달하는 제약 조건"""
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_accuracy: Optional[float] = None
