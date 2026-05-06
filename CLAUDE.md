# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

## 프로젝트 개요

이 프로젝트는 **NVIDIA Isaac Lab 2.3.0** / **Isaac Sim 5.1.0** 기반의 2족 로봇 보행 제어를 위한 강화학습 프레임워크입니다.

## 아키텍처

### 디렉토리 구조

```
src/main/           진입점: train.py, play.py, train_reach_avoid.py
src/lib/
  env/              Gymnasium 환경 (G1, GOAT, ant, humanoid)
  agent/            RL 에이전트 (PPO, MAPPO 변형, reach_avoid)
  model/            신경망 아키텍처 + ModelFactory
  buffer/           RolloutBuffer (GAE), ReplayBuffer 스텁
  domain_randomizer/ Sim-to-real 파라미터 랜덤화
  controller/       저수준 PD/PI 컨트롤러
  utils/            정규화, 로깅, 그래프 유틸리티
  assets/           로봇 URDF/USD 파일
src/wrapper/        Isaac Lab 환경 래퍼
```

### 환경 시스템

각 태스크는 `src/lib/env/<robot>/<task>/` 아래에 위치하며 다음을 포함합니다:
- 베이스 환경을 상속하는 환경 클래스
- 물리, 보상, 리셋 설정을 담은 config 클래스 (`EnvCfg`)
- 알고리즘별 하이퍼파라미터 YAML 파일 (`ppo_cfg.yaml`, `mappo_cfg.yaml`)
- gymnasium이 문자열 이름으로 태스크를 찾을 수 있도록 `__init__.py`에 등록

지원 로봇: G1 휴머노이드, GOAT 2족 Wheeled Bi-pedal 로봇, 기본 예제 환경 (ant, humanoid).

### 에이전트 시스템

`src/lib/agent/`
- **PPO** — 클리핑 Surrogate Loss Function + GAE를 사용하는 단일 에이전트 On-policy
- **MAPPO** — 멀티에이전트 PPO; 각 신체 부위가 독립적인 에이전트이며 중앙집중식 critic 사용
- **CooperativeMAPPO** — 에이전트 간 actor 파라미터를 공유하는 MAPPO
- **reach_avoid.py** — Reach-Avoid 공식화를 위한 위험 예측기

모든 에이전트는 `Agent.py`를 상속합니다 (체크포인트 저장, 모드 전환 처리).

### 모델 시스템

`src/lib/model/ModelFactory`가 `--model` 인자에 따라 아키텍처를 생성합니다:
- **MLP** — 표준 Actor/Critic MLP
- **Shared** — 멀티에이전트용 파라미터 공유 변형

### 학습 루프 (train.py)

1. 설정 파싱 → 래퍼를 적용한 Isaac Lab 환경 생성
2. `RolloutBuffer` 초기화, `ModelFactory`로 모델 생성, 에이전트 인스턴스화
3. 루프: `agent.act()` → `env.step()` → 버퍼에 삽입 → N 스텝마다 `agent.update()` (GAE + policy/value 최적화)
4. TensorBoard에 로깅; `.pt` 및 JIT 체크포인트 저장

## 중요사항

- 논의 및 research.md, plan.md, CLAUDE.md 작성은 모두 한국어로 진행해야 합니다.
- 코드 작성 시, 설명글 및 주석은 모두 영어로 작성해야 합니다.