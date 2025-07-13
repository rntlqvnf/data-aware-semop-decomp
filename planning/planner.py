# planning/planner.py

import logging
import json
import os
from typing import Dict, List, Any
from pathlib import Path
import yaml
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
import random
import textwrap 
import re

from knowledge.knowledge_base import KnowledgeBase
from core.schemas import Action, Plan, PlanStep, Strategy, Constraints, Metrics, ExecutionStatus, InputVar
from models.llm_client import LLMClient

class Planner:
    """
    사용자의 요청을 받아, LLM을 활용해 여러 후보 계획을 생성하고,
    샘플링을 통해 시스템의 초기 전략을 수립하는 총괄 설계자.
    """

    def __init__(self, config: Dict[str, Any], knowledge_base: KnowledgeBase):
        self.config = config
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm_client = LLMClient(self.config.get('llm', {}))
        
        # 💡 캐싱 설정을 초기화합니다.
        self.caching_config = self.config.get('planner', {}).get('caching', {})
        self.caching_enabled = self.caching_config.get('enabled', False)
        self.cache_file_path = self.caching_config.get('cache_file')

        # 💡 시각화 설정 로드
        self.visualization_config = self.config.get('planner', {}).get('visualization', {})
        self.visualization_enabled = self.visualization_config.get('enabled', False)

        try:
            self.prompt_env = Environment(loader=FileSystemLoader('prompts/'))
        except Exception as e:
            self.logger.error(f"Jinja2 환경 설정 실패: {e}")
            self.prompt_env = None

        self.logger.info("Planner가 생성되었습니다.")
        if self.caching_enabled and self.cache_file_path:
            self.logger.info(f"LLM 응답 캐싱이 활성화되었습니다. 캐시 파일: {self.cache_file_path}")
        if self.visualization_enabled:
            self.logger.info("LLM 실행 전략 시각화가 활성화되었습니다.")

    def _load_cache(self) -> Dict[str, Any]:
        """캐시 파일이 존재하면 로드하고, 없으면 빈 딕셔너리를 반환합니다."""
        if not self.cache_file_path:
            return {}
        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_to_cache(self, cache: Dict[str, Any], key: str, value: str):
        """주어진 키-값 쌍을 캐시에 저장합니다."""
        if not self.cache_file_path:
            return
        
        cache[key] = value
        
        # 캐시 파일이 저장될 디렉토리가 없으면 생성합니다.
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        
        with open(self.cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        self.logger.info(f"'{key[:30]}...'에 대한 LLM 응답을 캐시에 저장했습니다.")

    def _visualize_strategies(self, strategies: List[Strategy]):
        """생성된 Strategy들을 콘솔에 보기 좋게 시각화하여 출력합니다."""
        if not self.visualization_enabled:
            return

        self.logger.info("=" * 80)
        self.logger.info("                🤖 LLM이 생성한 실행 전략 시각화 🤖")
        self.logger.info("=" * 80)

        box_width = 78
        content_width = box_width - 4  # "│ "와 " │" 사이의 내용 너비

        for i, strategy in enumerate(strategies):
            def print_bordered(text, indent=0):
                lines = textwrap.wrap(text, width=content_width - indent)
                if not lines:
                    print(f"│ {' ' * content_width} │")
                    return
                for line in lines:
                    print(f"│ {' ' * indent}{line.ljust(content_width - indent)} │")

            def print_bordered_inputs(inputs: Dict[str, InputVar], indent=6):
                for var_name, var_info in inputs.items():
                    print_bordered(f"{var_name}:", indent=indent + 1)
                    if var_info.type == 'literal':
                        val = repr(var_info.value)
                        print_bordered(f"└─ type: literal", indent=indent + 3)
                        print_bordered(f"   value: {val}", indent=indent + 3)
                    elif var_info.type == 'variable':
                        source = var_info.source_step or "?"
                        idx = var_info.output_index if var_info.output_index is not None else 0
                        print_bordered(f"└─ type: variable", indent=indent + 3)
                        print_bordered(f"   source: {source} output[{idx}]", indent=indent + 3)
                    else:
                        print_bordered(f"└─ type: unknown", indent=indent + 3)

            print(f"\n┌{'─' * (box_width - 2)}┐")
            print_bordered(f"[ Strategy {i+1}: {strategy.name} ]")
            print_bordered(f"Description: {strategy.description}")
            print(f"├{'─' * (box_width - 2)}┤")

            # Plan의 각 LogicalStep 출력
            for step_num, step in enumerate(strategy.plan):
                print_bordered(f"[ Step {step_num + 1}: {step.op} (id: {step.id}) ]", indent=2)

                # Input 값 출력
                print_bordered("- Inputs:", indent=4)
                print_bordered_inputs(step.in_, indent=6)

                # Output 값 출력
                print_bordered(f"- Outputs: {step.out}", indent=4)

                if step_num < len(strategy.plan) - 1:
                    print_bordered("▼", indent=int(content_width / 2))

            print(f"├{'─' * (box_width - 2)}┤")
            print_bordered(f"[ Return ]", indent=2)
            print_bordered(f"- Value: {strategy.return_val}", indent=4)
            print(f"└{'─' * (box_width - 2)}┘")

        self.logger.info("=" * 80)

    def _visualize_candidate_plans(self, plans: List[Plan]):
        """생성된 Candidate Plan들을 콘솔에 보기 좋게 시각화하여 출력합니다."""
        if not self.visualization_enabled:
            return

        self.logger.info("=" * 80)
        self.logger.info("              📋 후보 Plan 시각화 📋")
        self.logger.info("=" * 80)

        box_width = 78
        content_width = box_width - 4  # "│ "와 " │" 사이의 내용 너비

        for plan in plans:
            def print_bordered(text: str, indent: int = 0):
                lines = textwrap.wrap(text, width=content_width - indent)
                if not lines:
                    print(f"│ {' ' * content_width} │")
                    return
                for line in lines:
                    print(f"│ {' ' * indent}{line.ljust(content_width - indent)} │")

            print(f"\n┌{'─' * (box_width - 2)}┐")
            print_bordered(f"[ Plan: {plan.name} ]")
            print_bordered(f"Derived from Strategy: {plan.strategy_name}")
            print(f"├{'─' * (box_width - 2)}┤")

            for idx, step in enumerate(plan.steps):
                print_bordered(f"[ Step {idx+1}: {step.step_id} ]", indent=2)
                print_bordered(f"- Action: {step.action_name}", indent=4)
                print_bordered(f"- Impl:   {step.implementation_name}", indent=4)

                # 입력 시각화
                print_bordered("- Inputs:", indent=4)
                for var_name, var_info in step.inputs.items():
                    if isinstance(var_info, dict):
                        source = var_info.get("type", "unknown")
                        if source == "literal":
                            val_str = f"{var_name} ← (literal) {var_info.get('value')}"
                        elif source == "variable":
                            val_str = (f"{var_name} ← (from step '{var_info.get('source_step')}' "
                                    f"output[{var_info.get('output_index', 0)}])")
                        else:
                            val_str = f"{var_name} ← (unknown source)"
                    else:
                        val_str = f"{var_name} ← {var_info}"  # fallback
                    print_bordered(val_str, indent=6)

                # 출력 시각화
                print_bordered("- Outputs:", indent=4)
                for out_key, out_val in step.outputs.items():
                    print_bordered(f"{out_key} → {out_val}", indent=6)

                if idx < len(plan.steps) - 1:
                    print_bordered("▼", indent=int(content_width / 2))

            print(f"└{'─' * (box_width - 2)}┘")

        self.logger.info("=" * 80)

    def _call_llm_and_update_cache(self, operator_prompt: str, modality: str, available_actions: List[Action], cache: Dict) -> str:
        """실제 LLM을 호출하고, 캐싱이 활성화된 경우 결과를 저장합니다."""
        system_prompt = self.config.get('planner', {}).get('llm_system_prompt', "You are a helpful AI assistant.")
        user_prompt = self._build_user_prompt(operator_prompt, modality, available_actions)
        llm_response_yaml = self.llm_client.generate(system_prompt, user_prompt)
        
        if llm_response_yaml and self.caching_enabled:
            self._save_to_cache(cache, operator_prompt, llm_response_yaml)
            
        return llm_response_yaml

    def _decompose_with_llm(self, operator_prompt: str, modality: str, available_actions: List[Action]) -> List[Strategy]:
        """LLM을 사용하여 사용자 요청을 여러 Strategy로 분해합니다. 캐싱 및 시각화를 지원합니다."""
        
        # ... (캐싱 로직은 이전과 동일) ...
        if self.caching_enabled:
            cache = self._load_cache()
            if operator_prompt in cache:
                self.logger.info(f"캐시에서 '{operator_prompt[:50]}...'에 대한 응답을 찾았습니다. 캐시된 결과를 사용합니다.")
                llm_response_yaml = cache[operator_prompt]
            else:
                self.logger.info(f"캐시에 해당 내용이 없습니다. LLM을 호출합니다: '{operator_prompt[:50]}...'")
                llm_response_yaml = self._call_llm_and_update_cache(operator_prompt, modality, available_actions, cache)
        else:
            llm_response_yaml = self._call_llm_and_update_cache(operator_prompt, modality, available_actions, {})

        if not llm_response_yaml:
            self.logger.error("LLM으로부터 빈 응답을 받았습니다.")
            return []

        try:
            if "```yaml" in llm_response_yaml:
                llm_response_yaml = llm_response_yaml.split("```yaml")[1].split("```")[0]

            parsed_data = yaml.safe_load(llm_response_yaml)
            strategies = [Strategy(**s) for s in parsed_data.get('strategies', [])]
            self.logger.info(f"LLM으로부터 {len(strategies)}개의 Strategy를 성공적으로 파싱했습니다.")
            
            # 💡 파싱 성공 후, 시각화 함수 호출
            self._visualize_strategies(strategies)
            
            return strategies
        except Exception as e:
            self.logger.error(f"LLM 응답 파싱 실패: {e}\n응답 내용: {llm_response_yaml}")
            return []

    def plan_and_optimize(self, operator_prompt: str, constraints: Constraints, data_path: Path, modality: str, sample_size: int) -> bool:
        self.logger.info("계획 및 최적화 프로세스를 시작합니다...")
        try:
            available_actions = self._load_action_catalogs(modality)
            if not available_actions: return False
            strategies = self._decompose_with_llm(operator_prompt, modality, list(available_actions.values()))
            if not strategies: return False
            candidate_plans = self._generate_candidate_plans(strategies, constraints, available_actions)
            if not candidate_plans: return False
            self.knowledge_base.register_plans(candidate_plans)
            profiling_results = self._run_profiling(candidate_plans, data_path, sample_size)
            self.knowledge_base.build_initial_model(profiling_results)
            self.logger.info("계획 및 최적화 프로세스가 성공적으로 완료되었습니다.")
            return True
        except Exception as e:
            self.logger.error(f"계획 및 최적화 과정에서 심각한 오류 발생: {e}", exc_info=True)
            return False

    def _load_action_catalogs(self, modality: str) -> Dict[str, Action]:
        self.logger.info(f"'{modality}' 및 'common' Action Catalog을 로드합니다.")
        actions: Dict[str, Action] = {}
        catalog_dir = Path("catalog/")
        catalog_files_to_load = [catalog_dir / "common_actions.yaml", catalog_dir / f"{modality}_actions.yaml"]
        for file_path in catalog_files_to_load:
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if not data: continue
                    for action_data in data:
                        actions[action_data['name']] = Action(**action_data)
            except (FileNotFoundError, yaml.YAMLError, ValidationError) as e:
                self.logger.warning(f"카탈로그 파일 처리 중 오류 발생 ({file_path}): {e}")
        self.logger.info(f"총 {len(actions)}개의 Action을 성공적으로 로드했습니다.")
        return actions

    def _build_user_prompt(self, operator_prompt: str, modality: str, available_actions: List[Action]) -> str:
        if not self.prompt_env:
            raise RuntimeError("Jinja2 템플릿 환경이 초기화되지 않았습니다.")
        actions_as_dict = [action.model_dump(exclude={'implementations'}) for action in available_actions]
        available_actions_yaml = yaml.dump(actions_as_dict, sort_keys=False, indent=2)
        template = self.prompt_env.get_template('decomposition_template.jinja2')
        user_prompt = template.render(
            operator_prompt=operator_prompt,
            modality=modality,
            num_strategies=self.config.get('planner', {}).get('decomposition_params', {}).get('num_strategies_to_generate', 3),
            available_actions_yaml=available_actions_yaml
        )
        return user_prompt

    def _generate_candidate_plans(self, strategies: List[Strategy], constraints: Constraints, available_actions: Dict[str, Action]) -> List[Plan]:
        """각 Strategy로부터 Plan 객체를 생성합니다."""
        self.logger.info(f"{len(strategies)}개의 Strategy로부터 후보 Plan들을 생성합니다...")
        candidate_plans = []

        for strategy in strategies:
            plan_steps = []

            for step in strategy.plan:
                step_id = step.id
                action_name = step.op
                action = available_actions.get(action_name)

                if not action:
                    self.logger.warning(f"Strategy '{strategy.name}'의 Action '{action_name}'을 카탈로그에서 찾을 수 없습니다.")
                    continue

                # 입력 매핑: 각 입력 변수를 처리
                inputs_dict = {}
                for input_name, input_val in step.in_.items():
                    if input_val.type == "literal":
                        inputs_dict[input_name] = input_val.value
                    elif input_val.type == "variable":
                        inputs_dict[input_name] = {
                            "source_step": input_val.source_step,
                            "output_index": input_val.output_index or 0
                        }
                    else:
                        self.logger.warning(f"알 수 없는 입력 타입: {input_val.type} in step {step_id}")

                # 출력 매핑: Logical → Physical 이름 그대로 전달
                outputs_dict = {name: name for name in step.out}

                # 구현체 결정 (임시로 첫 번째 구현체 사용)
                implementation_name = (
                    action.implementations[0].implementation_name
                    if action.implementations else "dummy_implementation"
                )

                plan_steps.append(
                    PlanStep(
                        step_id=step_id,
                        action_name=action_name,
                        implementation_name=implementation_name,
                        inputs=inputs_dict,
                        outputs=outputs_dict
                    )
                )

            # Plan 완성
            plan = Plan(
                name=f"Plan-for-{strategy.name}",
                strategy_name=strategy.name,
                steps=plan_steps
            )
            candidate_plans.append(plan)
            self.logger.info(f"생성된 Plan: {plan.name}")

        self._visualize_candidate_plans(candidate_plans)
        return candidate_plans

    def _run_profiling(self, plans: List[Plan], data_path: Path, sample_size: int) -> pd.DataFrame:
        """각 Plan에 대해 더미 프로파일링 데이터를 생성합니다."""
        self.logger.info(f"{sample_size}개의 샘플 데이터로 {len(plans)}개 Plan의 성능 프로파일링을 시작합니다.")
        profiling_data = []

        for i in range(sample_size):
            for plan in plans:
                # 무작위 더미 성능 지표 생성
                dummy_metrics = Metrics(
                    latency_ms=random.uniform(20.0, 2000.0),
                    cost_usd_per_image=random.uniform(0.0, 0.01),
                    accuracy=random.uniform(0.75, 0.99),
                    vram_mb=random.randint(500, 8000)
                )
                # 무작위 더미 데이터 특징 생성
                dummy_features = {
                    'clarity': random.random(),
                    'has_text': random.choice([True, False])
                }
                
                profiling_data.append({
                    "data_id": f"sample_{i+1}.jpg",
                    "plan_name": plan.name,
                    "status": random.choice(list(ExecutionStatus)).value,
                    "actual_metrics": dummy_metrics.model_dump(),
                    "input_features": dummy_features
                })
        
        self.logger.info("더미 프로파일링 데이터 생성을 완료했습니다.")
        return pd.DataFrame(profiling_data)
