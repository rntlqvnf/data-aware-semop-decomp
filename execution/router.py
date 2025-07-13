# execution/router.py

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm
import random # 더미 데이터 생성을 위해 추가

from knowledge.knowledge_base import KnowledgeBase
from core.schemas import Plan, ExecutionStatus, Metrics

# Router가 사용할 다른 컴포넌트들 (가상)
# from features.extractor import FeatureExtractor
from execution.executor import Executor

class Router:
    """
    데이터 스트림을 처리하며, 각 데이터 아이템의 특징을 분석하고
    KnowledgeBase를 참조하여 최적의 Plan으로 작업을 분배(라우팅)합니다.
    """

    def __init__(self, config: Dict[str, Any], knowledge_base: KnowledgeBase):
        """
        Router를 초기화합니다.

        Args:
            config (Dict): 전체 설정 파일 내용.
            knowledge_base (KnowledgeBase): 의사결정에 사용할 KnowledgeBase 인스턴스.
        """
        self.config = config.get('router', {})
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.routing_mode = self.config.get("routing_mode", "auto")
        
        # self.feature_extractor = FeatureExtractor()
        self.executor = Executor(config, knowledge_base)
        
        self.logger.info(f"Router가 생성되었습니다. (라우팅 모드: {self.routing_mode.upper()})")

    def _get_plan_from_human(self, features:  Dict[str, Any], item_path: Path) -> Optional[Plan]:
        """[Human Mode] 사용자에게 직접 Plan을 선택받습니다. 종료 옵션을 포함합니다."""
        available_plans = list(self.knowledge_base.plans.values())
        if not available_plans:
            return None

        print("\n" + "="*50)
        print(f"처리 대상: {item_path.name}")
        print(f"데이터 특징: {features}")
        print("어떤 Plan으로 실행하시겠습니까? 번호를 입력하세요. (종료하려면 'q' 입력)")
        
        for i, plan in enumerate(available_plans):
            print(f"  [{i+1}] {plan.name} (전략: {plan.strategy_name})")
        
        while True:
            try:
                choice = input("선택 (숫자 또는 'q'): ").lower().strip()
                
                # 💡 핵심 수정 부분: 종료 명령어 확인
                if choice in ['q', 'quit', 'exit']:
                    return None # None을 반환하여 종료 신호를 보냅니다.

                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_plans):
                    return available_plans[choice_idx]
                else:
                    print("잘못된 번호입니다. 다시 입력해주세요.")
            except (ValueError, IndexError):
                print("유효한 숫자를 입력해주세요.")

    def _get_plan_automatically(self, features: Dict[str, Any], item_path: Path) -> Optional[Plan]:
        """[Auto Mode] KnowledgeBase를 통해 최적의 Plan을 자동으로 선택합니다."""
        return self.knowledge_base.get_best_plan_for(features)

    def _discover_data_items(self, data_path: Path) -> List[Path]:
        """주어진 경로에서 처리할 데이터 파일 목록을 찾습니다."""
        self.logger.info(f"데이터 탐색 중: {data_path}")
        items = list(data_path.glob('**/*.jpg')) + list(data_path.glob('**/*.png'))
        self.logger.info(f"총 {len(items)}개의 처리 대상 데이터를 찾았습니다.")
        return items

    def process_data_stream(self, data_path: Path) -> Dict[str, Any]:
        """
        [main.py용 API] 전체 데이터셋 처리 및 적응형 실행 과정을 지휘합니다.
        """
        data_items = self._discover_data_items(data_path)
        stats = {"total_items_processed": 0, "success_count": 0, "failure_count": 0, "total_cost_usd": 0.0, "total_latency_ms": 0}

        data_iterator = data_items if self.routing_mode == 'human' else tqdm(data_items, desc="[Router] Processing Data Stream")

        for item_path in data_iterator:
            try:
                # features = self.feature_extractor.extract(item_path)
                features = {'clarity': 0.85} 
                if self.routing_mode == 'human':
                    best_plan = self._get_plan_from_human(features, item_path)
                    if best_plan is None:
                        self.logger.info("사용자에 의해 실행이 중단되었습니다.")
                        break # 메인 루프를 탈출합니다.
                else:
                    best_plan = self._get_plan_automatically(features, item_path)

                if not best_plan:
                    self.logger.warning(f"데이터 {item_path.name}에 대해 적합한 Plan을 찾지 못했습니다. 건너뜁니다.")
                    stats["failure_count"] += 1
                    stats["total_items_processed"] += 1
                    continue
                
                self.logger.info(f"데이터 '{item_path.name}'에 Plan '{best_plan.name}' 할당 및 실행.")
                stats["total_items_processed"] += 1

                trace = self.executor.run(item_path, best_plan, features)
                trace = type('Trace', (object,), {
                    'status': random.choice(list(ExecutionStatus)), 
                    'actual_metrics': Metrics(latency_ms=random.uniform(50, 1000), cost_usd_per_image=random.uniform(0.0, 0.005), accuracy=random.uniform(0.8, 1.0), vram_mb=0)
                })()

                if trace.status == ExecutionStatus.SUCCESS:
                    stats["success_count"] += 1
                else:
                    stats["failure_count"] += 1
                stats["total_cost_usd"] += trace.actual_metrics.cost_usd_per_image
                stats["total_latency_ms"] += trace.actual_metrics.latency_ms

            except Exception as e:
                self.logger.error(f"데이터 {item_path.name} 처리 중 오류 발생: {e}", exc_info=True)
                stats["failure_count"] += 1
        
        # 최종 통계 계산 및 반환
        total = stats["total_items_processed"]
        final_stats = {
            "total_items_processed": total,
            "success_count": stats["success_count"],
            "failure_count": stats["failure_count"],
            "success_rate": (stats["success_count"] / total) * 100 if total > 0 else 0,
            "total_cost_usd": stats["total_cost_usd"],
            "average_latency_ms": stats["total_latency_ms"] / total if total > 0 else 0
        }
        return final_stats
