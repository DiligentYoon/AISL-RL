# Research Notes

---

## Curriculum Learning 시스템 분석

### 1. 개요

`src/lib/curriculum/` 디렉토리에는 강화학습 훈련 중 환경 난이도를 점진적으로 높이는 **커리큘럼 매니저**가 구현되어 있다. 이 시스템은 훈련 초반에 쉬운 조건으로 시작하고, 훈련이 진행될수록 점차 어려운 조건으로 이행하도록 환경 파라미터를 자동으로 조정한다.

---

### 2. 구성 파일

| 파일 | 역할 |
|------|------|
| `curriculum_cfg.py` | 커리큘럼 설정 데이터 클래스 정의 |
| `curriculum_manager.py` | 실제 파라미터 보간 및 적용 로직 |
| `schedules.py` | 난이도 스케줄 함수 (`linear`, `step`) |
| `__init__.py` | 공개 API 노출 |

---

### 3. 설정 레이어 (`curriculum_cfg.py`)

#### `CurriculumParamCfg`

커리큘럼으로 제어할 파라미터 하나를 표현하는 Spec 카드.

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | `str` | 식별 이름 |
| `attr_path` | `str` | 환경 내 파라미터 경로 (`/` 구분자) |
| `start_value` | `Any` | difficulty=0일 때의 값 (쉬운 조건) |
| `end_value` | `Any` | difficulty=1일 때의 값 (어려운 조건) |
| `schedule` | `str` | 난이도 스케줄 함수 이름 (`"linear"` 또는 `"step"`) |
| `schedule_kwargs` | `dict` | 스케줄 함수에 전달할 추가 인자 |

`attr_path`는 `/` 로 구분된 경로 문자열로, 중간 노드가 `dict`면 키 접근, 객체면 `getattr`로 동적 탐색한다.

```
"cfg/events/push_robot/params/velocity_range"
  → env.cfg → .events → .push_robot → .params["velocity_range"]
```

#### `CurriculumManagerCfg`

| 필드 | 타입 | 설명 |
|------|------|------|
| `warmup` | `float` | 전체 timestep 중 difficulty=0으로 유지할 비율 (기본 0.1 = 10%) |
| `params` | `list[CurriculumParamCfg]` | 제어할 파라미터 목록 |

`total_timesteps`는 여기에 정의되지 않으며, `train.py`가 `gym.make()` 전에 `env_cfg.total_timesteps`에 주입한다.

---

### 4. 매니저 로직 (`curriculum_manager.py`)

#### 초기화 흐름

1. `__init__` 호출 시 `_build_resolvers()` 실행
2. 각 `CurriculumParamCfg`에 대해 `_make_accessor()`로 getter/setter 쌍을 생성
3. difficulty=0으로 `_apply(0.0)` 호출 → 모든 파라미터를 start_value로 초기화

#### `_make_accessor(root, path)`

경로 문자열을 파싱하여 동적 getter/setter 클로저를 생성한다.
- 중간 노드가 `dict` → 키 기반 접근
- 중간 노드가 객체 → `getattr`/`setattr` 기반 접근

#### `update(current_step)` — 매 스텝 호출됨

```
t = current_step / total_timesteps        (0→1 클리핑)

if t < warmup:
    effective_t = 0.0                     (워밍업 구간: 난이도 고정)
else:
    effective_t = (t - warmup) / (1 - warmup)   (워밍업 이후 0→1 재정규화)

difficulty = schedule_fn(effective_t)
new_value  = interpolate(start_value, end_value, difficulty)
setter(new_value)
```

#### `_interpolate(start, end, t)`

| 타입 | 처리 방식 |
|------|----------|
| `float` / `int` | 선형 보간: `start + (end - start) * t` |
| `tuple` / `list` | 원소별 선형 보간, 원래 타입 유지 |
| `dict` | 키별 재귀 호출 |
| 기타 | `t >= 0.5`이면 `end`, 아니면 `start` |

---

### 5. 스케줄 함수 (`schedules.py`)

#### `linear(t)`
```
difficulty = t   (0 → 1 연속 선형 증가)
```

#### `step(t, steps=[0.33, 0.66])`
n개의 임계값 → (n+1)단계 이산 난이도.

```
steps=[0.33, 0.66]:
  t < 0.33  →  difficulty = 0.0
  t < 0.66  →  difficulty = 0.5
  t >= 0.66 →  difficulty = 1.0
```

현재 등록된 스케줄: `{"linear": linear, "step": step}`

---

### 6. 환경과의 통합 (`env.py`)

`EnvBase.__init__()` 내부:
```python
if self.cfg.curriculum is not None:
    self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
else:
    self.curriculum_manager = None
```

`EnvBase.step()` 내부:
```python
self.common_step_counter += 1
if self.curriculum_manager is not None:
    self.curriculum_manager.update(self.common_step_counter)
```

`common_step_counter`는 모든 병렬 환경에 공통으로 증가하는 글로벌 스텝 카운터다. `env_cfg.py`의 `EnvCfg`에는 `curriculum: object | None = None`과 `total_timesteps: int | None = None`이 정의되어 있으며, 후자는 `train.py`에서 주입된다.

---

### 7. G1Recovery 환경에서의 적용 (`G1_recovery_env_cfg.py`)

현재 커리큘럼으로 제어되는 파라미터는 1개:

#### `push_velocity`

| 항목 | 값 |
|------|-----|
| `attr_path` | `"cfg/events/push_robot/params/velocity_range"` |
| `start_value` | `{"x": (-0.1, 0.1), "y": (-0.1, 0.1), "roll": (0.0, 0.0), "pitch": (0.0, 0.0)}` |
| `end_value` | `{"x": (-1.5, 1.5), "y": (-1.5, 1.5), "roll": (-2.5, 2.5), "pitch": (-2.5, 2.5)}` |
| `schedule` | `"linear"` |

**의미:** 훈련 초반 10% 동안은 푸시 없음(워밍업), 이후 선형적으로 x/y 최대속도를 0.1→1.5 m/s, roll/pitch 각속도를 0→2.5 rad/s로 증가시킨다.

설정:
```python
curriculum = CurriculumManagerCfg(
    warmup=0.1,
    params=[
        CurriculumParamCfg(
            name="push_velocity",
            attr_path="cfg/events/push_robot/params/velocity_range",
            start_value={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "roll": (0.0, 0.0), "pitch": (0.0, 0.0)},
            end_value  ={"x": (-1.5, 1.5), "y": (-1.5, 1.5), "roll": (-2.5, 2.5), "pitch": (-2.5, 2.5)},
            schedule="linear",
        ),
    ]
)
```

---

### 8. 데이터 흐름 요약

```
train.py
  └─ env_cfg.total_timesteps = N 주입
       └─ gym.make() → EnvBase.__init__()
            └─ CurriculumManager(cfg, env) 생성
                 └─ 각 파라미터에 대해 getter/setter 클로저 생성
                 └─ _apply(0.0) → start_value로 초기화

매 env.step() 호출 시:
  └─ common_step_counter += 1
  └─ curriculum_manager.update(common_step_counter)
       └─ t = step / total
       └─ effective_t (워밍업 반영)
       └─ difficulty = schedule_fn(effective_t)
       └─ new_value = interpolate(start, end, difficulty)
       └─ setter(new_value) → 환경 파라미터 즉시 변경
```

---

### 9. 현재 설계의 특이사항 및 제한

1. **TensorBoard 로깅 미활성화**: `curriculum_manager.py` 내 difficulty 기록 코드가 주석 처리되어 있어 난이도 변화를 모니터링할 수 없다.
2. **step 스케줄 미사용**: `schedules.py`에 `step` 함수가 정의되어 있으나 G1Recovery에서는 사용되지 않는다.
3. **단일 파라미터 제어**: 현재 G1Recovery에서는 `push_velocity` 하나만 커리큘럼으로 관리하며, 커맨드 속도 범위(`commands/cfg/ranges/*`) 등은 고정값이다.
4. **common_step_counter 기준**: `env.step()` 호출 횟수 기준이므로, 병렬 환경 수(`num_envs`)와 무관하게 하나의 글로벌 시계를 사용한다. 즉 4096개 환경이 동시에 실행돼도 `total_timesteps`는 물리 스텝 기준이다.
