# Synthetic few-shot evaluation

This test follows the homework's examples of calculating operating margin, comparing it with the company's industry, and asking a knowledgeable employee about working conditions. It tests the implemented rules on authored synthetic inputs. It is separate from the 20-company public benchmark and does not estimate the private grading score.

## Design

There are two demonstration interviews and twelve held-out company cases. One demonstration shows positive labor relations and the other shows substantial labor problems. Demonstration answers are supplied only in the two-shot condition and are never scored. None of the twelve held-out interviews is a demonstration. Expected labels for the test cases are stored separately from the tool responses and are not passed to the agent or model.

The zero-shot condition runs the existing production prompt. The two-shot condition wraps the same model with two human/assistant example pairs before the existing prompt. Both conditions run the unchanged `analyze_companies` function with synthetic SQL and interview tools. Each case has separate tool objects. The model is `deepseek-v4-flash`, with temperature zero and the same timeout settings. Each condition is repeated three times, and the order of conditions alternates between trials. Case order is shuffled with a fixed seed.

The numeric fixtures use revenue of 1,000 in USD thousands, selling and administration costs of 100, R&D costs of 50, and cost of sales chosen to produce the margins below. The tool returns these totals, and production Python code computes the ratio. Point 5 requires a positive FY2025 margin at least equal to the industry median. Point 6 compares FY2025 with FY2024. Point 7 uses the employee testimony. Only point 7 uses model interpretation, so the few-shot intervention cannot improve the deterministic financial comparisons.

## Held-out cases

All numbers below are percentage margins. Expected verdicts were fixed before running the test.

| Case | FY2024 | FY2025 | Industry | Interview focus | Point 5 | Point 6 | Point 7 |
|---|---:|---:|---:|---|---|---|---|
| 01 | 10 | 20 | 15 | Fair pay, long tenure, responsive supervisor | PASS | PASS | PASS |
| 02 | 20 | 10 | 15 | Positive management slogan, unpaid overtime and resignations | FAIL | FAIL | FAIL |
| 03 | 10 | 10 | 10 | Staff can decline overtime; safety issues are resolved | PASS | PASS | PASS |
| 04 | 10 | 15 | 20 | Good bonus but ignored safety problems | FAIL | PASS | FAIL |
| 05 | 20 | 15 | 10 | Historical dispute resolved; current conditions are good | PASS | FAIL | PASS |
| 06 | -10 | -5 | 10 | Public wellbeing claim conflicts with mistreatment | FAIL | PASS | FAIL |
| 07 | 0 | 0 | 0 | Employee cannot assess strategy but knows conditions are good | FAIL | PASS | PASS |
| 08 | -5 | 5 | 4 | Nice cafeteria but wages are late and experienced staff leave | PASS | PASS | FAIL |
| 09 | 25 | 20 | 18 | Seasonal overtime is voluntary, paid and agreed with staff | PASS | FAIL | PASS |
| 10 | 8 | 6 | 9 | Explicitly poor relations, high turnover and skipped breaks | FAIL | FAIL | FAIL |
| 11 | 8 | 12 | 15 | A colleague left for commuting reasons; current conditions are good | FAIL | PASS | PASS |
| 12 | 15 | 18 | 12 | Respect promised in handbook, retaliation in practice | PASS | PASS | FAIL |

Cases 03 and 07 check equality and zero-margin boundaries. Cases 04, 05, 06 and 09 separate profitability level from direction. The interviews test negation, misleading positive framing, a resolved historical problem, and inability to comment on an unrelated topic.

## Results

| Condition | Trial 1 | Trial 2 | Trial 3 | Point 7 across runs | Overall balanced score |
|---|---|---|---|---|---|
| Existing zero-shot prompt | 36/36 | 36/36 | 36/36 | 36/36 | 100.0% |
| Two demonstration interviews | 36/36 | 36/36 | 36/36 | 36/36 | 100.0% |

Each condition scored 108/108 point verdicts across the three runs. All three points separately had 100% accuracy and balanced accuracy. No execution failures or timeout fallback summaries were recorded. There are only twelve unique test companies: the repeated runs are not thirty-six independent companies. All raw predictions, summaries, examples, case definitions and run times are in `few_shot_results.json`.

Both conditions reached the ceiling of this small test. Adding two examples produced an observed accuracy change of zero percentage points, so this experiment provides no reason to replace the production prompt. The cases have clear authored answers and cannot establish robustness to all ambiguous or unfamiliar interview language. The financial labels test consistency with our own stated rules, not agreement with an independently known hidden grading rule.

## Reproduce

Use the normal project environment and DeepSeek key, then run `python few_shot_test.py`. The script performs six batches and normally uses 72 model calls before any retries. It saves `few_shot_results.json` after each completed batch. It does not overwrite the public `results.csv` or alter production prompts.
