네, 알겠습니다. 현재까지 논의된 내용을 바탕으로 `DWSemDcmp` 프로젝트의 `README.md` 파일을 만들어 드리겠습니다.

-----

# DWSemDcmp: Data-Aware Semantic Operator Decomposition

[](https://www.python.org/downloads/release/python-3100/)
[](https://opensource.org/licenses/MIT)

**DWSemDcmp**는 정적인 실행 계획의 한계를 극복하고, 실제 데이터의 특성에 실시간으로 적응하며 스스로 \*\*진화(Evolve)\*\*하는 데이터 처리 프레임워크입니다.

## 📖 개요 (Overview)

기존의 데이터 처리 시스템은 초기에 수립된 하나의 '최적' 계획을 끝까지 고수합니다. 하지만 현실의 데이터는 예측 불가능하며, 샘플링 단계에서는 발견되지 않았던 다양한 특징을 가집니다. 이로 인해 초기 계획은 쉽게 실패하거나 비효율적인 결과를 낳습니다.

**DWSemDcmp**는 이 문제를 해결하기 위해, \*\*동적 라우팅(Dynamic Routing)\*\*과 \*\*자가 진화(Self-Evolution)\*\*라는 핵심 아이디어를 기반으로 동작합니다. 여러 후보 계획을 유지하며 데이터 특징에 따라 최적의 계획으로 작업을 할당하고, 실행 결과를 모니터링하여 예측과 현실의 차이가 발생하면 스스로 전략을 수정하고 진화시킵니다.

## ✨ 핵심 개념 (Core Concepts)

  * **Action Catalog**: 시스템이 수행할 수 있는 모든 기본 동작(`Action`)의 명세서입니다. 각 Action은 여러 구현체(`Implementation`)를 가질 수 있습니다.
  * **Knowledge Base**: 시스템의 두뇌입니다. 어떤 데이터 특징에 어떤 계획이 효과적인지에 대한 통계 모델과 라우팅 규칙을 저장하고 실시간으로 학습/관리합니다.
  * **Planner**: 사용자의 목표와 `Action Catalog`를 기반으로, LLM을 활용하여 여러 후보 계획(`Plan`)을 생성하고 샘플링을 통해 초기 라우팅 전략을 수립합니다.
  * **Router**: 데이터 처리의 관제탑입니다. 데이터의 특징을 분석하고 `KnowledgeBase`를 참조하여 작업을 최적의 `Plan` 큐로 보냅니다.
  * **Evolution Loop**: `이상 현상 감지 → 원인 분석 → 전략 진화`로 이어지는 시스템의 핵심적인 학습 및 성장 메커니즘입니다.

## 🏗️ 아키텍처 (Architecture)

**DWSemDcmp**는 크게 **계획/최적화 단계**와 **실행/진화 단계**로 나뉩니다.

1.  **샘플링 기반 계획 및 최적화**: `Planner`가 LLM을 통해 여러 후보 계획을 생성하고, 샘플 데이터로 각 계획의 성능을 프로파일링하여 `KnowledgeBase`에 초기 라우팅 전략을 수립합니다.
2.  **적응형 실행 및 진화**: `Router`가 `KnowledgeBase`를 참조하여 전체 데이터를 동적으로 라우팅하고 실행합니다. 실행 과정에서 예측과 다른 결과가 지속적으로 발생하면 `Evolution Loop`가 발동하여 원인을 분석하고, 새로운 계획을 생성하거나 라우팅 규칙을 수정하여 시스템을 진화시킵니다.

## 📂 프로젝트 구조 (Project Structure)

```
.
├── main.py                     # 🚀 프레임워크 실행 진입점
├── Dockerfile                  # 📜 컨테이너 설계도
├── docker-compose.yml          # 🎼 컨테이너 실행 지휘서
├── requirements.txt            # 📦 파이썬 의존성 목록
├── .env                        # 🔑 API 키 등 비밀 정보 (Git에 포함하지 않음!)
├── .env.example                # .env 파일 예시
├── prompts                     # 📝 LLM 프롬프트 템플릿
├── configs/
│   └── config.yaml             # ⚙️ API 키, 모델명, 임계값 등 전역 설정
├── core/
│   └── schemas.py              # 📦 공통 데이터 구조 (Action, Plan, Trace 등)
├── data/
│   └── ...                     # 🖼️ 실험용 샘플 데이터셋
|
├── catalog/                    # 📚 Action 명세서
│   ├── image_actions.yaml
│   └── video_actions.yaml
|
├── 🧠 knowledge/
│   └── knowledge_base.py       # ✨ 시스템의 지식 저장소 (통계 모델, 라우팅 규칙 관리)
|
├── 💡 planning/
│   └── planner.py              # 📝 LLM을 이용한 Decomposition 및 후보 계획 생성, 초기 최적화
|
├── ⚙️ execution/
│   ├── router.py               # 🚦 데이터 특징 기반의 동적 라우터
│   └── executor.py             # 🏃‍♂️ 특정 계획(Plan)을 실행하는 모듈
|
├── ✨ features/
│   ├── base_extractor.py       # 🎛️ 특징 추출기 기본 인터페이스
│   └── image_features.py       # 📸 이미지 특징(흐림, 밝기 등) 추출 로직
|
├── 🔬 monitoring/
│   └── anomaly_detector.py     # 📈 예측-실제 성능 괴리(이상 현상) 감지
|
├── 🧬 evolution/
│   ├── causal_analyzer.py      # 🔍 이상 현상 원인(Missing Feature) 분석
│   └── strategy_evolver.py     # ♟️ 라우팅 규칙 업데이트, 신규 계획 생성 등 전략 진화
|
└── models/
    └── wrappers/               # 래퍼 클래스들 (LLM, Local Model 등)
```

## 🚀 시작하기 (Getting Started)

이 프로젝트는 Docker를 사용하여 모든 의존성과 GPU 환경을 관리합니다. 로컬에 Python 가상 환경을 직접 설정할 필요가 없습니다.

**사전 요구사항:**

  * [Docker](https://www.docker.com/get-started)
  * [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (GPU 사용 시 필수)

-----

1.  **저장소 복제**

    ```bash
    git clone https://[your_repository_url]/DWSemDcmp.git
    cd DWSemDcmp
    ```

2.  **환경 변수 설정**
    `.env.example` 파일을 복사하여 `.env` 파일을 생성하고, 필요한 API 키를 입력합니다. 이 파일은 컨테이너에 안전하게 전달됩니다.

    ```bash
    cp .env.example .env
    nano .env
    # AZURE_OPENAI_API_KEY=... 와 같이 필요한 키를 입력합니다.
    ```

    혹은, azure CLI를 통해 로그인한 후, 환경 변수를 자동으로 설정할 수 있습니다:

    ```bash
    az login
    ```

3.  **도커 이미지 빌드**
    `docker compose`가 `Dockerfile`을 읽어 GPU와 모든 의존성이 포함된 이미지를 생성합니다. (최초 실행 또는 `requirements.txt` 변경 시에만 필요)

    ```bash
    docker compose build
    ```

4.  **도커 컨테이너 실행**
    `docker-compose.yml`에 정의된 설정에 따라 컨테이너를 실행합니다. 코드 수정 시, 이미지를 재빌드할 필요 없이 자동으로 컨테이너에 반영됩니다.

    ```bash
    docker compose up
    ```

    만약 GPU를 붙여 실행하고자 하는 경우

    ```bash
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
    ```

      * 컨테이너를 백그라운드에서 실행하려면 `-d` 플래그를 추가하세요: `docker compose up -d`
      * 실행을 중지하려면 `Ctrl + C`를 누르거나, 다른 터미널에서 `docker compose down`을 입력하세요.

## 💻 사용법 (Usage)

`main.py`를 통해 프레임워크를 실행할 수 있습니다.

```bash
python main.py \
  --operator "Find high-quality images of wooden tables" \
  --operator_type "Filter" \
  --modality "image" \
  --constraints '{"max_cost_usd": 0.5, "min_accuracy": 0.9}' \
  --data_path "./data/furniture_images"
```