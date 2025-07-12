# knowledge/knowledge_base.py

import logging
from typing import Dict, List, Optional, Any
import pandas as pd

# Pydantic 스키마를 임포트하여 타입 힌트로 사용합니다.
from core.schemas import Plan, ExecutionTrace

class KnowledgeBase:
    """
    시스템의 모든 지식(계획, 통계 모델, 라우팅 규칙)을 저장, 관리, 제공하는 중앙 컴포넌트.
    """

    def __init__(self):
        """
        KnowledgeBase를 초기화합니다.
        내부적으로 Plan 저장소와 통계 모델을 가집니다.
        """
        # 이제 Plan 객체를 저장하는 딕셔너리임을 명확히 합니다.
        self.plans: Dict[str, Plan] = {}
        self.statistical_model: Any = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("KnowledgeBase가 생성되었습니다.")

    def register_plans(self, plans: List[Plan]) -> None:
        """
        Planner가 생성한 초기 후보 계획들을 등록합니다.

        Args:
            plans (List[Plan]): Planner가 생성한 Plan 객체의 리스트.
        """
        for plan in plans:
            self.plans[plan.name] = plan
        self.logger.info(f"{len(self.plans)}개의 후보 계획이 KnowledgeBase에 등록되었습니다.")
        # pass # 실제 구현 로직

    def build_initial_model(self, profiling_results: pd.DataFrame) -> None:
        """
        샘플링 및 프로파일링 결과를 바탕으로 초기 통계 모델을 구축합니다.
        'P(성능 | 데이터 특징, 계획)' 모델의 첫 버전을 만듭니다.

        Args:
            profiling_results (pd.DataFrame): 각 Plan을 샘플 데이터에 실행한 결과.
        """
        self.logger.info("샘플링 데이터 기반으로 초기 통계 모델을 구축합니다...")
        # TODO: profiling_results 데이터프레임을 분석하여 통계 모델 구축
        self.statistical_model = {} # 임시 모델
        self.logger.info("초기 통계 모델 구축 완료.")
        # pass # 실제 구현 로직

    def get_best_plan_for(self, features: Dict[str, Any]) -> Optional[Plan]:
        """
        [Router용 API] 주어진 데이터 특징에 가장 적합한 Plan을 추천합니다.

        Args:
            features (Dict[str, Any]): 데이터에서 추출된 특징 딕셔너리.

        Returns:
            Optional[Plan]: 추천되는 Plan 객체. 적합한 Plan이 없으면 None.
        """
        self.logger.debug(f"특징 {features}에 대한 최적 계획을 조회합니다.")
        # TODO: self.statistical_model을 참조하여 최적의 Plan을 찾는 로직 구현.

        # 임시 로직 예시
        # best_plan_name = self.statistical_model.predict(features)
        # return self.plans.get(best_plan_name)

        # 스켈레톤이므로 첫 번째 플랜을 임시로 반환
        if self.plans:
            return next(iter(self.plans.values()))
        return None

    def update_from_trace(self, trace: ExecutionTrace) -> None:
        """
        [Executor/Monitor용 API] 단일 데이터 처리 후의 실행 기록(Trace)을 받아 통계 모델을 업데이트합니다.

        Args:
            trace (ExecutionTrace): 단일 데이터 아이템의 실행 결과 정보 객체.
        """
        self.logger.debug(f"Trace 정보를 바탕으로 모델을 업데이트합니다: data_id={trace.data_id}, plan={trace.plan_used}, status={trace.status}")
        # TODO: 새로운 데이터 포인트를 기존 통계 모델에 반영하여 점진적으로 학습/개선하는 로직 구현.
        # pass # 실제 구현 로직

    def evolve_strategy(self, new_plan: Optional[Plan] = None, new_routing_rule: Optional[Dict] = None) -> None:
        """
        [StrategyEvolver용 API] 진화된 전략(신규 Plan 또는 신규 라우팅 규칙)을 시스템에 반영합니다.

        Args:
            new_plan (Optional[Plan]): 새로 생성된 Plan 객체.
            new_routing_rule (Optional[Dict]): 새로 발견된 특징에 대한 라우팅 규칙.
        """
        if new_plan:
            self.plans[new_plan.name] = new_plan
            self.logger.info(f"전략 진화: 신규 Plan '{new_plan.name}'이 KnowledgeBase에 추가되었습니다.")
            # TODO: 새 Plan에 대한 통계 모델 초기화 로직

        if new_routing_rule:
            self.logger.info(f"전략 진화: 새로운 라우팅 규칙이 반영되었습니다: {new_routing_rule}")
            # TODO: self.statistical_model에 새로운 규칙을 직접 반영/수정하는 로직

        # pass # 실제 구현 로직