#!/bin/bash
# 현재 실행 중인 노드들의 파라미터를 파일로 저장 (트랙에서 튜닝한 값 백업)
# 사용: ./tools/save_params.sh  [저장경로]
# 예:   ./tools/save_params.sh  config/race_params.saved.yaml
set -e
OUT="${1:-config/race_params.saved.yaml}"
TMP=$(mktemp -d)

echo "현재 파라미터 저장 중 -> $OUT"
ros2 param dump /decision_node > "$TMP/decision.yaml" 2>/dev/null || { echo "decision_node 실행 중이 아님"; }
ros2 param dump /lane_node     > "$TMP/lane.yaml"     2>/dev/null || { echo "lane_node 실행 중이 아님"; }

# 두 노드 덤프를 한 파일로 합침
cat "$TMP/decision.yaml" "$TMP/lane.yaml" 2>/dev/null > "$OUT"
rm -rf "$TMP"
echo "저장 완료: $OUT"
echo "불러오려면: ros2 run decision decision_node --ros-args --params-file $OUT"
