# execution/router.py

import logging
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

from knowledge.knowledge_base import KnowledgeBase
from core.schemas import ExecutionStatus

# Router가 사용할 다른 컴포넌트들 (가상)
# from features.extractor import FeatureExtractor
# from execution.executor import Executor


class Router:
    """
    데이터 스트림을 처리하며, 각 데이터 아이템의 특징을 분석하고
    KnowledgeBase를 참조하여 최적의 Plan으로 작업을 분배(라우팅)합니다.
    """

    def __init__(self, config: Dict[str, Any], knowledge_base: KnowledgeBase):
        """
        Router를 초기화합니다.

        Args:
            config (Dict): Router 관련 설정.
            knowledge_base (KnowledgeBase): 의사결정에 사용할 KnowledgeBase 인스턴스.
        """
        self.config = config
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Router는 특징 추출기와 실행기를 내부적으로 사용합니다.
        # self.feature_extractor = FeatureExtractor()
        # self.executor = Executor(knowledge_base=self.knowledge_base)
        
        self.logger.info("Router가 생성되었습니다.")

    def _discover_data_items(self, data_path: Path) -> List[Path]:
        """주어진 경로에서 처리할 데이터 파일 목록을 찾습니다."""
        self.logger.info(f"데이터 탐색 중: {data_path}")
        # 예시: 해당 디렉토리의 모든 jpg, png 파일을 대상으로 함
        items = list(data_path.glob('**/*.jpg')) + list(data_path.glob('**/*.png'))
        self.logger.info(f"총 {len(items)}개의 처리 대상 데이터를 찾았습니다.")
        return items

    def process_data_stream(self, data_path: Path) -> Dict[str, Any]:
        """
        [main.py용 API] 전체 데이터셋 처리 및 적응형 실행 과정을 지휘합니다.

        1. 데이터 목록 탐색
        2. 각 데이터 아이템에 대해 루프 실행:
           a. 특징 추출 (Feature Extraction)
           b. 최적 계획 조회 (Query KnowledgeBase)
           c. 계획 실행 위임 (Delegate to Executor)
           d. 결과 통계 집계
        3. 최종 통계 반환

        Args:
            data_path (Path): 처리할 전체 데이터셋의 경로.

        Returns:
            Dict[str, Any]: 최종 실행 결과 통계.
        """
        data_items = self._discover_data_items(data_path)

        # 최종 통계를 위한 변수 초기화
        success_count = 0
        failure_count = 0
        total_cost = 0.0
        total_latency = 0

        # tqdm을 사용하여 진행률 표시
        for item_path in tqdm(data_items, desc="[Router] Processing Data Stream"):
            try:
                # 1. 특징 추출
                # features = self.feature_extractor.extract(item_path)
                features = {'clarity': 0.85} # 임시 특징

                # 2. KnowledgeBase에 최적 계획 질의
                best_plan = self.knowledge_base.get_best_plan_for(features)

                if not best_plan:
                    self.logger.warning(f"데이터 {item_path.name}에 대해 적합한 Plan을 찾지 못했습니다. 건너뜁니다.")
                    failure_count += 1
                    continue
                
                self.logger.debug(f"데이터 {item_path.name}에 Plan '{best_plan.name}' 할당.")

                # 3. Executor에게 실제 실행 위임
                # Executor는 실행 후 Trace 객체를 반환하며, 내부적으로 Monitor/Evolver를 호출할 수 있음.
                # trace = self.executor.run(item_path, best_plan, features)
                
                # 임시 Trace
                from core.schemas import ExecutionStatus, Metrics
                trace = type('Trace', (object,), {
                    'status': ExecutionStatus.SUCCESS, 
                    'actual_metrics': Metrics(latency_ms=50, cost_usd_per_image=0.001, accuracy=1.0, vram_mb=0)
                })()


                # 4. 결과 통계 집계
                if trace.status == ExecutionStatus.SUCCESS:
                    success_count += 1
                else:
                    failure_count += 1
                total_cost += trace.actual_metrics.cost_usd_per_image
                total_latency += trace.actual_metrics.latency_ms

            except Exception as e:
                self.logger.error(f"데이터 {item_path.name} 처리 중 오류 발생: {e}", exc_info=True)
                failure_count += 1
        
        # 최종 통계 계산 및 반환
        total_items = len(data_items)
        final_stats = {
            "total_items_processed": total_items,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": (success_count / total_items) * 100 if total_items > 0 else 0,
            "total_cost_usd": total_cost,
            "average_latency_ms": total_latency / total_items if total_items > 0 else 0
        }
        return final_stats