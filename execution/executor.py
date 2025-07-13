# execution/executor.py

import logging
import time
from typing import Dict, Any
from pathlib import Path

from knowledge.knowledge_base import KnowledgeBase
from core.schemas import Plan, ExecutionTrace, ExecutionStatus, Metrics, PhysicalInput

from implementations import image as image_impl
from implementations import common as common_impl

class Executor:
    """
    단일 Plan과 데이터 아이템을 받아, 계획된 단계를 순차적으로 실행하고
    그 결과를 ExecutionTrace로 반환합니다.
    """

    def __init__(self, config: Dict[str, Any], knowledge_base: KnowledgeBase):
        """
        Executor를 초기화합니다.

        Args:
            config (Dict): Executor 관련 설정.
            knowledge_base (KnowledgeBase): 실행 결과를 업데이트할 KnowledgeBase 인스턴스.
        """
        self.config = config
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Executor가 생성되었습니다.")

    def run(self, data_item_path: Path, plan: Plan, features: Dict[str, Any]) -> ExecutionTrace:
        """Executes a Plan on a given data item and returns the execution trace."""
        self.logger.info(f"'{plan.name}' 실행 시작. 대상: '{data_item_path.name}'")
        start_time = time.time()

        try:
            # Initialize context
            with open(data_item_path, 'rb') as f:
                image_bytes = f.read()

            context: Dict[str, Any] = {}

            for step in plan.steps:
                self.logger.debug(f"Step '{step.step_id}' 실행 중: {step.action_name}")
                step_inputs = {}

                # Resolve inputs
                for input_name, input_info in step.inputs.items():
                    if isinstance(input_info, PhysicalInput):
                        source_step = input_info.source_step
                        output_index = input_info.output_index

                        if source_step == "input":
                            # 🔧 최초 입력에서 가져오는 변수 처리
                            if input_name == "binary_image":
                                step_inputs[input_name] = image_bytes
                            elif input_name in features:
                                step_inputs[input_name] = features[input_name]
                            elif input_name == "filename":
                                step_inputs[input_name] = data_item_path.name
                            else:
                                raise ValueError(f"'input'으로부터 유도된 '{input_name}' 처리 미지원")
                        else:
                            # 일반 step 결과에서 가져오기
                            if source_step not in context:
                                raise ValueError(f"'{source_step}'의 출력이 context에 없음.")
                            
                            source_values = context[source_step]
                            if isinstance(source_values, list):
                                if output_index >= len(source_values):
                                    raise IndexError(f"'{output_index}'가 '{source_step}'의 출력 범위를 벗어남.")
                                step_inputs[input_name] = source_values[output_index]
                            elif isinstance(source_values, dict):
                                keys = list(source_values.keys())
                                if output_index >= len(keys):
                                    raise IndexError(f"'{output_index}'가 '{source_step}'의 출력 범위를 벗어남.")
                                step_inputs[input_name] = source_values[keys[output_index]]
                            else:
                                if output_index != 0:
                                    raise ValueError(f"'{source_step}'의 단일 출력은 output_index=0만 허용됨.")
                                step_inputs[input_name] = source_values
                    else:
                        step_inputs[input_name] = input_info

                # Special case for control-flow action
                if step.action_name == "FilterByBoolean":
                    condition = step_inputs["condition"]
                    if not condition:
                        self.logger.info(f"조건 불충족. FilterByBoolean({condition}) → 실행 중단.")
                        break
                    continue

                # Dispatch and execute
                if step.action_name == "ExecutePythonCode": 
                    # Special case: ExecutePythonCode
                    code_string = step_inputs.get("code_string")
                    context_inputs = {k: v for k, v in step_inputs.items() if k != "code_string"}
                    expected_outputs = list(step.outputs.values())

                    result = self._dispatch_action(
                        action_name=step.action_name,
                        implementation_name=step.implementation_name,
                        code_string=code_string,
                        context=context_inputs,
                        expected_outputs=expected_outputs
                    )
                else:
                    # 일반 액션 실행
                    result = self._dispatch_action(
                        action_name=step.action_name,
                        implementation_name=step.implementation_name,
                        **step_inputs
                    )

                # Store outputs
                output_var_map = step.outputs
                if not isinstance(result, dict):
                    # If single output but dict not returned
                    result = {"_default_output": result}

                mapped_outputs = {}
                for i, (out_key, var_name) in enumerate(output_var_map.items()):
                    if out_key in result:
                        mapped_outputs[var_name] = result[out_key]
                    else:
                        # fallback to positional
                        result_values = list(result.values())
                        if i < len(result_values):
                            mapped_outputs[var_name] = result_values[i]
                        else:
                            raise ValueError(f"출력 매핑 오류: '{step.step_id}'의 출력 {out_key} 없음")

                context[step.step_id] = list(mapped_outputs.values()) if len(mapped_outputs) > 1 else [list(mapped_outputs.values())[0]]
                self.logger.debug(f"Step '{step.step_id}' 결과: {context[step.step_id]}")

            end_time = time.time()
            final_output = context.get(plan.steps[-1].step_id, None)

            actual_metrics = Metrics(
                latency_ms=(end_time - start_time) * 1000,
                cost_usd_per_image=0.001,
                accuracy=0.95,
                vram_mb=0
            )

            trace = ExecutionTrace(
                data_id=data_item_path.name,
                plan_used=plan.name,
                status=ExecutionStatus.SUCCESS,
                input_features=features,
                actual_metrics=actual_metrics,
                output=final_output
            )

        except Exception as e:
            self.logger.error(f"Plan '{plan.name}' 실행 중 오류 발생: {e}", exc_info=True)
            end_time = time.time()
            trace = ExecutionTrace(
                data_id=data_item_path.name,
                plan_used=plan.name,
                status=ExecutionStatus.FAILURE,
                input_features=features,
                actual_metrics=Metrics(
                    latency_ms=(end_time - start_time) * 1000,
                    cost_usd_per_image=0.0,
                    accuracy=0.0,
                    vram_mb=0
                ),
                error_message=str(e)
            )

        self.knowledge_base.update_from_trace(trace)
        return trace

    def _dispatch_action(self, action_name: str, implementation_name: str, **kwargs) -> Any:
        """
        Dynamically resolve and execute the appropriate function for an action.
        """
        function_name = f"{action_name.replace('-', '_')}_{implementation_name.replace('-', '_').replace('.', '_')}"
        self.logger.debug(f"Dispatching to function: {function_name}")

        modules_to_search = [common_impl, image_impl]

        for module in modules_to_search:
            if hasattr(module, function_name):
                return getattr(module, function_name)(**kwargs)

        raise NotImplementedError(f"'{function_name}'에 해당하는 Action 구현 함수를 찾을 수 없습니다.")