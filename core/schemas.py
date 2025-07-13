# core/schemas.py

from enum import Enum
from typing import List, Dict, Any, Optional, Literal, Union
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

class PhysicalInput(BaseModel):
    source_step: str         # e.g., "decode_image"
    output_index: int = 0    # e.g., 0 (default)

class PlanStep(BaseModel):
    """실행 계획(Plan)을 구성하는 단일 단계"""
    step_id: str
    action_name: str
    implementation_name: str
    inputs: Dict[str, Union[PhysicalInput, Any]]  # Any = literal
    outputs: Dict[str, str]  # output_name → variable_name

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

class InputVar(BaseModel):
    type: Literal["variable", "literal"]
    
    # For variable type
    source_step: Optional[str] = None  # step_id, or "input" if user-provided
    output_index: Optional[int] = 0    # defaults to first output

    # For literal type
    value: Optional[Any] = None

class LogicalStep(BaseModel):
    id: str
    op: str
    in_: Dict[str, InputVar] = Field(..., alias="in")
    out: List[str]

class Strategy(BaseModel):
    name: str
    description: str
    plan: List[LogicalStep]
    return_val: List[str] = Field(..., alias="return")

class Constraints(BaseModel):
    """Constraints passed by the user"""
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_accuracy: Optional[float] = None