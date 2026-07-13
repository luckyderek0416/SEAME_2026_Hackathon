# D-Racer 자율주행 디버깅 매뉴얼 (에이전트 필독)

실주행 디버깅에서 반복 확인된 사실 모음. **추측하지 말고 여기 적힌 실측 근거를
우선하라.** "건드리지 말 것"은 과거에 시간을 크게 낭비했거나 상황을 악화시킨 항목이다.
(2026-07-13 현행화. 현재 로직 설명은 README, 런별 이력은 RUN_LOG, 남은 작업은
MISSING_FEATURES 참고 — 이 문서는 **절차·금지·하드웨어**에 집중한다.)

## 0. 황금 규칙

1. **한 번에 하나만 바꾼다.** 동시에 바꾸면 어떤 게 효과였는지 알 수 없다.
2. **실측 근거 없는 수치 변경 금지.** 모든 수치는 텔레메트리(`~/telemetry.jsonl`)
   실측과 대조해서 정한다.
3. **게이트/지각 로직 변경은 오프라인 검증 후에만 배포한다** — scratchpad 의
   리플레이(sw_replay, validate_*) + 텔레메트리 시뮬로 과거 런 회귀 확인. 이 절차가
   07-13 재설계에서 잘못된 설계안 2개를 배포 전에 걸러냈다.
4. **"멈춰" 지시가 오면 다른 무엇보다 먼저 `bash /tmp/stop.sh`** → 0x40/0x42 중립 확인
   → 프로세스 0 확인. 그 다음에야 분석.
5. **증상을 하드웨어/소프트웨어로 먼저 분류한다** (§4 "보드가 죽는 3가지").
6. 두 번 같은 시도가 실패하면 증상을 다시 분류하고 접근을 바꾼다(무한 패치 금지).
   특히 **문턱값 돌려막기 금지** — 두 신호의 분포가 겹치면 어떤 문턱도 동전던지기다
   (07-13 B누출 0.19 vs 약피처 0.19 실측). 그때는 문턱이 아니라 신호를 바꿔야 한다.

## 1. 절대 건드리지 말 것

- **i2c-3 버스가 잠기면 차가 소프트웨어로 안 멈춘다** (07-11 실사고) — PWM 칩이 마지막
  스로틀을 유지하고, ESC 전원이라 보드를 꺼도 계속 돈다. **비상정지 = ESC 스위치 OFF 뿐.**
- **i2c 노드를 `kill -9`로 죽이지 말 것** — 전송 중이면 버스가 잠긴다.
  반드시 SIGTERM: `pkill -TERM -f 'install/.*/lib/'`.
- **주행 중 UART를 꽂아두지 말 것** — 케이블 당김이 전원 커넥터를 흔들어
  `bootreason=cold` 재부팅 유발 이력. 텔레메트리는 fsync 로 남으니 없어도 안전.
- **`crossline` 검출은 HoughLinesP 기반** — 컬럼평균 최소제곱으로 되돌리지 말 것
  (정지선이 차선과 T자 한 성분이 되면 최소제곱 붕괴, inlier 0.05 실측).
- **`yellow_heading_gain = 0`** — 켜면 a_h 가 ±2.5 로 널뛰어 요동.
- **`center_jump_max_ratio` 0.10 위로 올리지 말 것** — 0.15 는 분기 목 쐐기 당김이
  통과 (run11 실증).
- **yaw(조향 적분) 위치창을 새로 만들지 말 것** — yaw ∝ 시간이라 런별 속도차에 창이
  통째로 밀린다. run66(옛 yaw 게이트), run86(백업 표결 스케일 오발)이 그 증거.
  위치 판별은 **군집 카운트(순서 불변량)** 가 담당한다 (README 참고).
- **시간 기반 캘리값은 배터리 밴드가 바뀌면 무효** — 이 팩은 랩타임이 15→30s 까지
  변한다. "시간 X초에 이 이벤트" 식 가정을 세우지 말 것.
- 부호(±) 민감값 — **부호는 뒤집지 말고 크기만 조정**: 최종 steer 는 >0.26=좌.
  바이어스 층은 음수=좌/안쪽 (`entry_steer_bias -0.15`, `ra_blind_bias -0.15`),
  양수=우/바깥 (`exit_steer_bias +0.18`), 병합 브리지 `merge_blind_bias +0.10`(좌,
  center 가산이라 예외적으로 양수=좌 — 주석 참고).

## 2. 빌드 / 실행 / 접속

```bash
cd ~/SEAME_2026_Hackathon-clone/D-Racer-Kit     # 보드
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select perception decision   # symlink 아님 — 수정 후 필수!
ros2 launch decision auto_race.launch.py course:=in race_dir:=left
```
- **"고쳤는데 안 바뀐다" 1순위 원인 = colcon build 안 함.** 배포 후엔 로컬/보드src/
  보드install **3-way md5 대조**가 관례.
- /tmp 는 램디스크 — 재부팅 시 launch.sh/aux.sh/stop.sh/telemetry_dual.py/frame_rec.py
  재배포 (노트북 scratchpad 에 사본).
- 보드 IP 유동: 핫스팟 서브넷(172.20.10.1~14) 포트 22 스캔. MAC `fc:22:1c:40:62:96`.
- **아이폰 핫스팟 함정 2개**: "호환성 최대화" 꺼져 있으면 5GHz 전용이라 동글이 못 봄 /
  일부 설정에선 기기 간 격리로 ssh 불가. UART 복구: `sudo picocom -b 115200 /dev/ttyUSB0`
  (topst/topst) → `sudo nmcli device wifi connect "SSID" password "PW"`.

## 3. 증상별 치트시트 (현행 스택 기준)

> "건드릴 것"의 파라미터만, 한 번에 하나, 작은 스텝. 전부 라이브(`ros2 param set`).

| 증상 | 건드릴 것 | 비고 |
|---|---|---|
| RA 오발화 / 미발화 | `gate_cluster_on_s`(0.20)·`gate_blank_s` — 단 **문턱 돌려막기 전에 RUN_LOG 07-13 절 필독** | 군집 게이트 자체는 시뮬 6/6 검증. 의심되면 텔레메트리 군집 타임라인부터 |
| B 개구부에서 링 이탈 | `sw_curv_max_a`(0.003) — 0=off 킬스위치 | 우곡률 가지 물기 방어 (run87). **race_dir=right 면 0 으로 끌 것** |
| 정지선 미검출 (STOPLINE 모드) | `stopline_cov_min`(0.25)·`stopline_ang_max`(15) ↓ / 최후엔 `stopline_mode 0`+`gate_blank_s 7.0` 페어 원복 | f59 실측: 진짜 선 cov 0.28~0.71, 각 5~19° |
| 탈출 후 병합 표류/점선 오추종 | `merge_blind_bias`(0.10)·`merge_bridge_s`(6.0) / 원복은 `merge_bridge_s 0`+`w_align_dash_fallback 1` | run83/84 우표류·좌이탈 봉합 |
| 분기 진입 실패 (1L) | `entry_oneline_frames`(80) ±10 | run17 검증 메커니즘 |
| RA 진입 언더스티어 | `entry_steer_bias`(-0.15) 좌필요→더 음수, ±0.02 | run19 확정 |
| RA 중 차선 소실 직진 | `ra_blind_bias`(-0.15) | 소실 프레임 한정 |
| 마커 정지 채터링/재출발 지연 | `marker_clear_frames`(8) | run79/80 실측 |
| 마커가 아예 안 잡힘 | **사전/ID 부터 확인** (`aruco_dict`) — 문턱 아님! | run76~78: 4X4 vs 6×6 로 3런 소모 |
| Out 갈림길 | MISSING_FEATURES A-2 의 증상→손잡이 맵 | |
| 커브 반응 느림/요동 | 해상도↓, QoS depth=1 (설정만) | 캡처 스레딩 재작성 금지 |

## 4. 보드가 죽는 3가지 (혼동 금지)

### ① Wi-Fi 드라이버/동글 사망 — 가장 흔함, 보드는 멀쩡
- **ROS 노드는 계속 돈다. SSH/대시보드만 끊긴다.** 주행 자체는 계속됨 →
  **원격 정지 불가 = 즉시 ESC 물리 컷.**
- 복구: 시리얼 콘솔에서 nmcli 재연결. `device was removed` 면 동글 재삽입. 최후 reboot.

### ② i2c-3 버스 잠김 — 전원 완전 차단만이 답
- 빈 주소까지 `errno=16 (busy)`. 재부팅으로 안 풀린다: ESC OFF → 보드 전원 분리 →
  **10초 대기**(커패시터 방전) → 재투입. 정상 버스는 빈 주소가 errno=11.

### ③ 진짜 커널 패닉/워치독 — 드묾
- `bootreason` 확인. 트레이스는 전원 투입 **전에** `picocom -g ~/serial.log` 로만 잡힌다.
- 부수: eMMC 저널 off(norecovery) — 하드락업 중 전원 차단 시 initramfs 로 떨어짐.
  USB 의 `e2fsck.static` 으로 `/dev/mmcblk0p4` 복구.

## 5. 배터리 (실패 1순위 원인)

- **충전 중 단자 전압은 잔량이 아니다.** 충전기 분리 후 5분 안정화 무부하 전압으로 판단.
- **현 팩은 노화 말기**: "완충" 무부하 7.9V(정상 8.4V), 부하 새그 ~1V, 대역별 비선형
  (흰 과속 + 노랑 저속이 같은 런에서 동시 발생 — 그래서 동적 스로틀 보정 gain 0).
  **무부하 7.7V 미만 주행 금지 권장** — 등판 정지(run63/70/71)·브라운아웃(run55/58) 영역.
- 랩타임이 15~30s 로 변한다 → **시간 기반 가정 전부 왜곡** (07-13 실패 런들의 공통 배경).
- 수동 전압 확인: bus3 addr 0x42 reg 0x02, `v = (((raw&0xff)<<8 | raw>>8) >> 3) × 0.004`.

## 6. 텔레메트리 / 분석 절차

- `/tmp/telemetry_dual.py` → `~/telemetry.jsonl` (fsync — 브라운아웃에도 생존).
  `frame_rec.py` → `~/frames/dbg_*.jpg`(오버레이)+`raw_*.jpg` (다음 aux 실행 시 삭제 —
  분석할 런이면 먼저 회수).
- ssh 비대화형은 ROS 미소싱 — 원격 실행 시 반드시 `source ... && python3 -u ...`.
- **좀비 스트리머 주의**: 노트북 ssh 를 죽여도 보드 python 이 살아남아 다음 런에
  이어붙는다. `pgrep -af telemetry` 확인, 정확한 PID 로만 kill.
- 분석 순서: **STATE 전이 이벤트 → 군집/의심 구간 프레임(dbg) → 사용자 육안 관찰과
  대조.** 사용자의 물리 관찰이 텔레메트리 추론보다 우선한다 (이 세션에서 수차례 실증).
- 개입 구간은 분석 제외 (조향 큰데 offset 이 안 따라오면 손 개입 서명).
- 발화 로그: `gate=1 exitV=2` = 표결 앵커(정상 주경로) / `exitV=0` = 카운트 경로 /
  `circle_t≈max_loop` = 강제 타임아웃(비상).

## 7. 검증 하네스 (로직 변경 시 필수 절차)

노트북 scratchpad(`/tmp/claude-*/.../scratchpad`)에 축적된 도구:
- **sw_replay.py** — dbg 프레임 → 검출기 전체 재실행 (모드 타임라인은 텔레메트리 STATE 로)
- **validate_guard / validate_stopline / validate_merge / measure_span** — 곡률 가드·
  STOPLINE·병합 브리지 검증 스크립트 (프레임 세트: f42/f53full/f54/f59/f75ra/f87)
- **텔레메트리 게이트 시뮬** — 군집 카운트+표결 재현, run73~87 정답창 대조
- 규칙: 새 지각/게이트 로직은 ①기존 성공런 비회귀 ②실패런 원인 제거 둘 다 통과해야 배포.
  ON/OFF 킬스위치 파라미터를 반드시 함께 넣는다.

## 8. 외부 레퍼런스 채택 판단 (조사 완료 — 뒤집으려면 실측 먼저)

- **공개 차선검출(다항식 피팅 계열) 이식 금지** — 컬럼평균 붕괴 문제 재발 (§1 Hough 항목).
- **Duckietown 계열 제어 루프 이식 금지** — 아키텍처 상이, 참고만.
- **성능 개선은 설정으로만** (QoS depth=1, 해상도↓, 프레임 스로틀). 캡처 스레딩
  재작성 금지 — camera_node 가 죽으면 주행 전체가 조용히 멈춘다.
