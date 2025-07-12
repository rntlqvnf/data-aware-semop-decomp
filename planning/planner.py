# planning/planner.py

import logging
from typing import Dict, List, Any
from pathlib import Path
import yaml
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# 프로젝트의 다른 모듈에서 필요한 클래스들을 임포트합니다.
from knowledge.knowledge_base import KnowledgeBase
from core.schemas import Action, Plan, Strategy, Constraints 

# 실제 구현 시 주석을 해제할 가상 클래스들입니다.
# from models.llm_client import LLMClient
# from execution.executor import Executor

class Planner:
    """
    사용자의 요청을 받아, LLM을 활용해 여러 후보 계획을 생성하고,
    샘플링을 통해 시스템의 초기 전략을 수립하는 총괄 설계자.
    """

    def __init__(self, config: Dict[str, Any], knowledge_base: KnowledgeBase):
        """
        Planner를 초기화합니다.

        Args:
            config (Dict): Planner 관련 설정 (e.g., 사용할 LLM 모델, 카탈로그 경로 등)
            knowledge_base (KnowledgeBase): 결과를 저장할 KnowledgeBase 인스턴스.
        """
        self.config = config
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LLM 클라이언트 및 프로파일링용 실행기 초기화 (가상)
        # self.llm_client = LLMClient(config.get('llm', {}))
        # self.executor = Executor()
        
        # LLM 프롬프트 템플릿을 위한 Jinja2 환경 설정
        # 'prompts' 디렉토리에서 템플릿 파일을 찾습니다.
        try:
            self.prompt_env = Environment(loader=FileSystemLoader('prompts/'))
            self.logger.info("Jinja2 프롬프트 템플릿 환경이 설정되었습니다.")
        except Exception as e:
            self.logger.error(f"Jinja2 환경 설정 실패: {e}")
            self.prompt_env = None

        self.logger.info("Planner가 생성되었습니다.")

    def plan_and_optimize(
        self,
        operator_prompt: str,
        constraints: Constraints,
        data_path: Path,
        modality: str,
        sample_size: int
    ) -> bool:
        """
        [main.py용 API] 계획 및 최적화의 전체 1단계 과정을 지휘합니다.
        """
        self.logger.info("계획 및 최적화 프로세스를 시작합니다...")
        try:
            # 1. Action Catalog 로드
            available_actions = self._load_action_catalogs(modality)
            if not available_actions:
                self.logger.error("사용 가능한 Action이 없어 계획을 중단합니다.")
                return False

            # 2. LLM을 이용해 Strategy 분해
            strategies = self._decompose_with_llm(operator_prompt, modality, list(available_actions.values()))
            if not strategies:
                self.logger.error("LLM으로부터 유효한 Strategy를 생성하지 못해 계획을 중단합니다.")
                return False

            # 3. 각 Strategy로부터 후보 Plan 생성
            candidate_plans = self._generate_candidate_plans(strategies, constraints, available_actions)
            if not candidate_plans:
                self.logger.error("유효한 후보 Plan을 생성하지 못해 계획을 중단합니다.")
                return False

            # 4. 생성된 Plan을 KnowledgeBase에 등록
            self.knowledge_base.register_plans(candidate_plans)

            # 5. 샘플 데이터로 각 Plan을 프로파일링
            profiling_results = self._run_profiling(candidate_plans, data_path, sample_size)

            # 6. 프로파일링 결과로 KnowledgeBase의 초기 통계 모델 구축
            self.knowledge_base.build_initial_model(profiling_results)

            return True

        except Exception as e:
            self.logger.error(f"계획 및 최적화 과정에서 심각한 오류 발생: {e}", exc_info=True)
            return False

    def _load_action_catalogs(self, modality: str) -> Dict[str, Action]:
        """지정된 모달리티에 맞는 Action Catalog들을 로드하고 파싱합니다."""
        self.logger.info(f"'{modality}' 및 'common' Action Catalog을 로드합니다.")
        actions = {}
        # TODO: 'catalog/{modality}_actions.yaml' 및 'catalog/common_actions.yaml' 파일을 로드하여
        # YAML 내용을 Action Pydantic 모델로 변환하는 로직을 구현해야 합니다.
        # 예시:
        # with open(f'catalog/{modality}_actions.yaml', 'r') as f:
        #     data = yaml.safe_load(f)
        #     for action_data in data:
        #         actions[action_data['name']] = Action(**action_data)
        return actions

    def _build_llm_prompt(self, operator_prompt: str, modality: str, available_actions: List[Action]) -> str:
        """Jinja2 템플릿을 사용하여 LLM에게 전달할 최종 프롬프트를 생성합니다."""
        if not self.prompt_env:
            raise RuntimeError("Jinja2 템플릿 환경이 초기화되지 않았습니다.")
            
        self.logger.info("LLM 프롬프트 생성을 시작합니다...")
        
        # Action 객체 리스트를 LLM이 이해하기 쉬운 YAML 문자열로 변환합니다.
        # 이 때, 불필요한 'implementations' 필드는 제외하여 프롬프트를 간결하게 만듭니다.
        actions_dict_list = [action.model_dump(exclude={'implementations'}) for action in available_actions]
        available_actions_yaml = yaml.dump(actions_dict_list, sort_keys=False, indent=2)

        template = self.prompt_env.get_template('decomposition_template.jinja2')
        
        prompt = template.render(
            operator_prompt=operator_prompt,
            modality=modality,
            num_strategies=self.config.get('num_strategies_to_generate', 3),
            available_actions_yaml=available_actions_yaml
        )
        self.logger.debug(f"생성된 LLM 프롬프트:\n{prompt[:500]}...") # 로그가 너무 길어지지 않도록 일부만 출력
        return prompt

    def _decompose_with_llm(self, operator_prompt: str, modality: str, available_actions: List[Action]) -> List[Strategy]:
        """LLM을 사용하여 사용자 요청을 여러 Strategy로 분해합니다."""
        self.logger.info(f"LLM을 사용하여 '{operator_prompt}'를 Strategy로 분해합니다...")
        
        final_prompt = self._build_llm_prompt(operator_prompt, modality, available_actions)
        
        # TODO: self.llm_client를 사용하여 final_prompt를 LLM에 보내고 응답을 받는 로직 구현
        # llm_response_yaml = self.llm_client.generate(final_prompt)
        llm_response_yaml = "strategies: []" # 임시 응답

        try:
            parsed_data = yaml.safe_load(llm_response_yaml)
            strategies = [Strategy(**s) for s in parsed_data.get('strategies', [])]
            self.logger.info(f"LLM으로부터 {len(strategies)}개의 Strategy를 성공적으로 파싱했습니다.")
            return strategies
        except (yaml.YAMLError, TypeError) as e:
            self.logger.error(f"LLM 응답 파싱 실패: {e}\n응답 내용: {llm_response_yaml}")
            return []

    def _generate_candidate_plans(self, strategies: List[Strategy], constraints: Constraints, available_actions: Dict[str, Action]) -> List[Plan]:
        """각 Strategy와 제약조건을 바탕으로 구체적인 Plan들을 생성합니다."""
        self.logger.info(f"{len(strategies)}개의 Strategy로부터 후보 Plan들을 생성합니다...")
        candidate_plans = []
        # TODO: 각 Strategy의 Action 단계마다, Action Catalog를 참조하여
        # 제약조건(비용, 시간 등)을 만족하는 여러 implementation 조합을 찾아내어
        # Plan 객체들을 생성하는 로직을 구현해야 합니다.
        # (e.g., 비용 우선 Plan, 정확도 우선 Plan 등)
        return candidate_plans

    def _run_profiling(self, plans: List[Plan], data_path: Path, sample_size: int) -> pd.DataFrame:
        """샘플 데이터에 대해 각 Plan을 실행하여 성능을 측정(프로파일링)합니다."""
        self.logger.info(f"{sample_size}개의 샘플 데이터로 {len(plans)}개 Plan의 성능 프로파일링을 시작합니다.")
        profiling_results = []
        # TODO: data_path에서 sample_size만큼 데이터를 샘플링하고,
        # 각 샘플 데이터에 대해 모든 후보 Plan을 self.executor를 통해 실행한 뒤,
        # 그 결과를 ExecutionTrace 형태로 수집하여 pandas DataFrame으로 만드는 로직을 구현해야 합니다.
        return pd.DataFrame(profiling_results)