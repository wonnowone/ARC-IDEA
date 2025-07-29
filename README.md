# ARC-IDEA # TEMPORARY
(1) 문제 입력

(2) 현재 단계 상태

├── solver 1: incremental diagram
├── solver 2: supposition following
├── solver 3: chain construction
├── solver 4: concatenation strategy
└── symbolic solver

→ few shot learner로 input

(3) 각 solver EFE 평가 (Risk+Ambiguity):

risk: Z-learning으로 정답으로 풀이된 문제 분포(사후 분포)와 solver로 풀어낼 수 있는 사전 분포의 일치성 검사

ambiguity: 각 단계의 문제 혹은 문제 조건이 구체화될 수 있는가? 정합성을 이전 단계와 가지는가

(4) 최적 solver 선택 및 output 생성
│
(5) RevThink를 이용해 역방향 문제 재구현 (원래 문제 조건으로 복원 가능한지 검증)
│
├── Yes (일관성 확인)
│     └─ 다음 단계 진행
└── No (불일치 발견)
└─ threshold 조정 & solver 재선택


(6) 다음 단계로 진행 또는 최종 출력
