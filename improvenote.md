# AAAI 개선 정리 (2026-06-10)

## 목적
현재 원고의 약점(비교 공정성, Oracle 해석, 인과 단정)을 줄이기 위해, 공통 Oracle baseline 기반으로 2-domain/5-domain/sweep 실험을 일관 프로토콜로 재구성한다.

## 1) 공통 Oracle baseline 프로토콜 고정
- Backbone: Ours와 동일 백본 사용 (가능하면 INR 통일).
- Data split: 동일 Stage-1 clean split, 동일 검증/테스트 분할.
- Budget: 동일 outer_steps, inner_steps, stride, adapt_buffer, max_train_windows.
- Selection: 동일 early-stop 기준(SWD patience 등) 사용.
- Seed: 최소 3개(42/123/456) 고정.

## 2) 실험 세트 구성
- 2-domain (SMD->MSL->SMD)
  - 비교군: Static, SDA, Oracle-reference, Ours
  - 지표: T2 AUROC, T3 AUROC, visit AUROC, collapse_count, below_random_count
- 5-domain stream
  - 비교군: Static, SDA, Oracle-reference, Ours
  - 지표: routing(P/R/F1/lag/FA) + 적응 성능(AUROC 계열) 동시 보고
- CID sweep (eta sweep)
  - 비교군: SDA vs Ours + Oracle-reference
  - 지표: mean/std, worst-case, rolling worst-case, collapse_count

## 3) 표/서술 원칙 (논문 톤)
- Oracle는 "upper bound" 대신 "clean-data retraining reference"로 통일.
- 백본이 다르면 "정책 우월" 해석 금지, "representation + protocol 결과"로 한정.
- "solely due to CID" 같은 단정 대신 "primarily contamination-driven" 사용.

## 4) 통계 보고 최소 요건
- 모든 핵심 표에 seed 평균/표준편차 또는 95% CI 표기.
- 가능하면 paired test (동일 seed 매칭) 추가.
- 본문에는 대표 수치, 부록에는 per-seed raw 표 제공.

## 5) 리뷰어 대응용 핵심 문장 (요지)
- 공정성: "모든 주요 비교는 동일 backbone/동일 budget 프로토콜에서 수행했다."
- Oracle 해석: "Oracle은 clean-data retraining reference이며, 고정 스케줄 효과와 구분해 해석한다."
- CID 인과: "SDA 대비 열화는 contamination-driven이며, 본 실험 설정에서 CID가 지배적 메커니즘으로 관찰된다."

## 6) 실행 체크리스트
- [ ] 공통 Oracle 설정 파일/CLI 인자 고정
- [ ] 2-domain 3-seed 실행
- [ ] 5-domain 3-seed 실행
- [ ] sweep 3-seed 실행
- [ ] 집계 JSON 생성 (mean/std/worst/collapse)
- [ ] 본문 표/캡션/해석 문구 동기화
