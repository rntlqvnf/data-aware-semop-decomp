import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# =================================================================
# 프로젝트 모듈 임포트 (가상)
# 각 파일이 실제로 구현되면 이 클래스들을 사용하게 됩니다.
# =================================================================
# from core.schemas import Constraints # Pydantic 모델로 제약조건 관리
from planning.planner import Planner
from knowledge.knowledge_base import KnowledgeBase
from execution.router import Router

# =================================================================
# 로깅 설정
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DWSemDcmp] - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드하고 파싱합니다.
    """
    logging.info(f"설정 파일을 로드합니다: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"설정 파일 파싱 오류: {e}")
        exit(1)


def main():
    """
    DWSemDcmp 프레임워크의 메인 드라이버.
    계획, 최적화, 실행의 전 과정을 지휘합니다.
    """
    # --- 0. 인수 파싱 및 초기 설정 ---
    parser = argparse.ArgumentParser(description="DWSemDcmp: Data-Aware Semantic Operator Decomposition Framework")
    parser.add_argument("--operator", type=str, required=True, help="실행할 시맨틱 연산자 (e.g., 'Find high-quality images of wooden tables')")
    parser.add_argument("--operator_type", type=str, required=True, choices=['Filter', 'Map'], help="연산자 종류")
    parser.add_argument("--modality", type=str, required=True, help="데이터 모달리티 (e.g., 'image')")
    parser.add_argument("--constraints", type=str, default='{}', help="JSON 형식의 제약 조건 (e.g., '{\"max_cost_usd\": 0.5}')")
    parser.add_argument("--data_path", type=Path, required=True, help="처리할 데이터셋의 경로")
    parser.add_argument("--config_path", type=Path, default=Path("configs/config.yaml"), help="설정 파일 경로")
    args = parser.parse_args()

    logging.info("DWSemDcmp 프레임워크를 시작합니다...")
    logging.info(f"요청된 연산: {args.operator}")

    config = load_config(args.config_path)
    try:
        constraints = json.loads(args.constraints)
        # constraints = Constraints(**json.loads(args.constraints)) # Pydantic 사용 시
    except json.JSONDecodeError as e:
        logging.error(f"제약 조건 JSON 파싱 오류: {e}")
        return

    # --- 1. 핵심 컴포넌트 초기화 ---
    logging.info("핵심 컴포넌트를 초기화합니다...")
    knowledge_base = KnowledgeBase()
    planner = Planner(config=config, knowledge_base=knowledge_base)
    router = Router(config=config.get('router', {}), knowledge_base=knowledge_base)
    logging.info("모든 컴포넌트가 성공적으로 초기화되었습니다.")

    # --- 2. [1단계] 계획 및 초기 최적화 ---
    logging.info("=" * 60)
    logging.info("PHASE 1: 계획 및 샘플링 기반 최적화를 시작합니다.")
    logging.info("=" * 60)

    # 이 한 줄의 API 호출이 LLM 분해, 후보 계획 생성, 샘플링/프로파일링,
    # KnowledgeBase 초기화 등 복잡한 1단계 과정을 모두 포함합니다.
    success = planner.plan_and_optimize(
        operator_prompt=args.operator,
        constraints=constraints,
        data_path=args.data_path,
        modality=args.modality,
        sample_size=config.get('sampling', {}).get('size', 100)
    )

    if not success:
        logging.error("초기 계획 수립에 실패했습니다. 프로세스를 종료합니다.")
        return

    logging.info("PHASE 1 완료: KnowledgeBase에 초기 라우팅 전략이 수립되었습니다.")

    # --- 3. [2단계] 적응형 실행 및 진화 ---
    logging.info("=" * 60)
    logging.info("PHASE 2: 전체 데이터셋에 대한 적응형 실행을 시작합니다.")
    logging.info("=" * 60)

    # 이 한 줄의 API 호출이 전체 데이터 처리, 동적 라우팅, 모니터링,
    # 그리고 필요시 Evolution Loop를 발동시키는 모든 과정을 포함합니다.
    final_stats = router.process_data_stream(data_path=args.data_path)

    logging.info("PHASE 2 완료: 전체 데이터셋 처리가 끝났습니다.")

    # --- 4. 최종 결과 요약 ---
    logging.info("=" * 60)
    logging.info("DWSemDcmp 프로세스가 모두 종료되었습니다.")
    if final_stats:
        logging.info("최종 실행 통계:")
        for key, value in final_stats.items():
            logging.info(f"  - {key}: {value}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()