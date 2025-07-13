import logging
import os
import json
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ================================================================
# Common Action Implementations (core_framework)
# ================================================================
# 함수 네이밍 규칙: <ActionName>_<ImplementationName>
# 모든 함수는 Action 정의에 명시된 시그니처를 그대로 따르며,
# Azure 의존성이 전혀 없습니다.
# ================================================================

# ----------------------------------------------------------------
# 1. ExecutePythonCode — Python_Exec
# ----------------------------------------------------------------

def ExecutePythonCode_Python_Exec(code_string: str, context: Dict[str, Any], expected_outputs: List[str]) -> Dict[str, Any]:
    local_ns = dict(context)  # 전달받은 변수만 포함
    try:
        exec(code_string, {}, local_ns)
    except Exception as e:
        logger.exception("Python_Exec failed")
        raise
    return {k: local_ns.get(k) for k in expected_outputs}


# ----------------------------------------------------------------
# 2. FilterByBoolean — Boolean_Filter
# ----------------------------------------------------------------

def FilterByBoolean_Boolean_Filter(condition: bool):
    """Control‑flow helper.

    If `condition` is False, raise a StopIteration to allow the upstream
    framework to skip remaining actions for the current item.
    If True, simply return and allow processing to continue.
    """
    logger.debug("FilterByBoolean_Boolean_Filter received condition=%s", condition)
    if not condition:
        logger.info("Condition is False → terminating current item processing.")
        raise StopIteration("Filtered by Boolean_Filter action.")

# ----------------------------------------------------------------
# 3. CreateDictionary — Dict_Creator
# ----------------------------------------------------------------

def CreateDictionary_Dict_Creator(data: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("CreateDictionary_Dict_Creator building dict with keys=%s", list(data.keys()))
    return dict(data)

# ----------------------------------------------------------------
# 4. GetDictionaryValue — Dict_Accessor
# ----------------------------------------------------------------

def GetDictionaryValue_Dict_Accessor(dictionary_object: Dict[str, Any], key: str) -> Any:
    logger.debug("GetDictionaryValue_Dict_Accessor fetching key=%s", key)
    if key not in dictionary_object:
        raise KeyError(f"Key '{key}' not found in dictionary.")
    return dictionary_object[key]

# ----------------------------------------------------------------
# 5. LogVariable — Console_Logger
# ----------------------------------------------------------------

def LogVariable_Console_Logger(variable_to_log: Any, message: str = ""):
    if message:
        logger.info("%s: %s", message, variable_to_log)
    else:
        logger.info("LogVariable: %s", variable_to_log)

# ----------------------------------------------------------------
# 6. SaveToFile — File_Saver
# ----------------------------------------------------------------

def _serialize_for_file(data: Any) -> bytes:
    """Best‑effort serialization helper."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    # Fallback to JSON
    try:
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    except TypeError:
        # Last resort: repr
        return repr(data).encode("utf-8")


def SaveToFile_File_Saver(data_to_save: Any, file_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    payload = _serialize_for_file(data_to_save)
    with open(file_path, "wb") as f:
        f.write(payload)
    logger.info("Data saved to %s (%d bytes)", file_path, len(payload))
