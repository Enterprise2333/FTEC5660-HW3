"""Synthetic, held-out comparison of the existing prompt and two-shot labor prompting.

No real company data is read. Expected labels are test oracles only and are
never passed to analyze_companies or the model. Production prompts are unchanged.
"""
import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import hw3


EXAMPLES = [
    ('Staff have predictable shifts, paid training, and a trusted grievance process.',
     {'verdict': 'PASS', 'reason': 'Predictable work and trusted grievance handling support good relations.'}),
    ('Supervisors withhold earned wages and punish workers who report injuries.',
     {'verdict': 'FAIL', 'reason': 'Withheld wages and retaliation indicate serious labor problems.'}),
]

# Previous margin, current margin, industry median, interview, expected points.
# These fixtures are authored for this test, not copied from the public answer key.
CASES = [
    (10, 20, 15, 'Pay is fair, most colleagues stay for years, and the supervisor resolves complaints promptly.', ['PASS','PASS','PASS']),
    (20, 10, 15, 'Management calls this a family, but compulsory unpaid overtime is routine and colleagues keep resigning.', ['FAIL','FAIL','FAIL']),
    (10, 10, 10, 'We can decline extra shifts without being punished. Safety concerns are fixed and relations are respectful.', ['PASS','PASS','PASS']),
    (10, 15, 20, 'The bonus is attractive, but broken safety equipment has been ignored for months and staff fear speaking up.', ['FAIL','PASS','FAIL']),
    (20, 15, 10, 'There was a dispute last year. It was resolved through consultation; pay is fair and people now choose to stay.', ['PASS','FAIL','PASS']),
    (-10, -5, 10, 'The company praises wellbeing publicly. In practice supervisors humiliate workers and requests for rest are dismissed.', ['FAIL','PASS','FAIL']),
    (0, 0, 0, 'I cannot judge corporate strategy. On working conditions, we have fair pay, safe shifts and managers who listen.', ['FAIL','PASS','PASS']),
    (-5, 5, 4, 'A new cafeteria is nice, but wages are frequently late and experienced workers leave because complaints go nowhere.', ['PASS','PASS','FAIL']),
    (25, 20, 18, 'Demand is seasonal, but extra work is voluntary and paid. Employee representatives agree schedules with management.', ['PASS','FAIL','PASS']),
    (8, 6, 9, 'We do not have good relations with supervisors. Turnover is high and exhausted workers are pressured to skip breaks.', ['FAIL','FAIL','FAIL']),
    (8, 12, 15, 'One colleague left for a shorter commute, not because of our employer. Conditions are safe, wages fair and grievances handled properly.', ['FAIL','PASS','PASS']),
    (15, 18, 12, 'The handbook promises respect. Workers who challenge unsafe workloads get worse shifts, and complaints remain unresolved.', ['PASS','PASS','FAIL']),
]


class SQLTool:
    def __init__(self, previous, current, median):
        self.previous, self.current, self.median = previous, current, median

    async def ainvoke(self, arguments):
        query = arguments['query']
        if 'JOIN industry_medians' in query:
            return f'industry | median\nSynthetic industry | {self.median}'
        half = '2025-07-01' in query or '2026-07-01' in query
        margin = self.previous if '2024-01-01' in query else self.current
        # Revenue 1,000; selling/admin 100; R&D 50; COGS completes the chosen margin.
        cogs = 1000 - margin * 10 - 150
        return ('revenue | cost_of_sales | selling_admin | research_development | cost_months\n'
                f'1000 | {cogs} | 100 | 50 | {6 if half else 12}')


class InterviewTool:
    def __init__(self, text):
        self.text = text

    async def ainvoke(self, arguments):
        assert arguments['topic'] == 'working_conditions'
        assert arguments['role'] in ('current_employee', 'former_employee')
        return self.text


class FewShotModel:
    def __init__(self, model):
        self.model = model

    async def ainvoke(self, prompt):
        messages = []
        for evidence, answer in EXAMPLES:
            messages.extend([
                ('human', 'Assess Fisher point 7 from employee testimony. Return verdict and reason as JSON.\n' + evidence),
                ('ai', json.dumps(answer)),
            ])
        return await self.model.ainvoke(messages + [('human', prompt)])


def metrics(records):
    scores = {}
    for point in (5, 6, 7):
        rows = [r for r in records if r['point'] == point]
        rates = [sum(r['correct'] for r in rows if r['expected'] == label) /
                 sum(r['expected'] == label for r in rows) for label in hw3.VERDICTS]
        scores[str(point)] = {'correct': sum(r['correct'] for r in rows), 'total': len(rows),
                              'accuracy': sum(r['correct'] for r in rows) / len(rows),
                              'balanced_accuracy': sum(rates) / 2}
    rates = [sum(r['correct'] for r in records if r['expected'] == label) /
             sum(r['expected'] == label for r in records) for label in hw3.VERDICTS]
    scores['all'] = {'correct': sum(r['correct'] for r in records), 'total': len(records),
                     'accuracy': sum(r['correct'] for r in records) / len(records),
                     'balanced_accuracy': sum(rates) / 2}
    return scores


async def main():
    hw3.load_env_file()
    model = hw3.build_agent()
    records, runs = [], []
    for trial in range(1, 4):
        for mode in (['zero_shot', 'two_shot'] if trial % 2 else ['two_shot', 'zero_shot']):
            indexed = list(enumerate(CASES, 1))
            random.Random(5660 + trial).shuffle(indexed)
            companies = {f'synthetic_{i:02}': [None, None, SQLTool(*case[:3]), InterviewTool(case[3])]
                         for i, case in indexed}
            start = time.monotonic()
            outputs = await hw3.analyze_companies(model if mode == 'zero_shot' else FewShotModel(model), companies)
            trial_records = []
            for i, case in indexed:
                output = outputs[f'synthetic_{i:02}']
                for point, expected in zip((5, 6, 7), case[4]):
                    predicted = output['points'].get(point)
                    trial_records.append({'mode': mode, 'trial': trial, 'case': i, 'point': point,
                                          'expected': expected, 'predicted': predicted,
                                          'correct': expected == predicted, 'summary': output['summary']})
            records.extend(trial_records)
            runs.append({'mode': mode, 'trial': trial, 'seconds': round(time.monotonic()-start, 2),
                         'metrics': metrics(trial_records)})
            Path('few_shot_results.json').write_text(json.dumps(
                {'created_at': datetime.now(timezone.utc).isoformat(), 'model': hw3.MODEL_NAME,
                 'examples': EXAMPLES, 'cases': CASES, 'runs': runs, 'records': records},
                indent=2, ensure_ascii=False), encoding='utf8')
            print(mode, trial, metrics(trial_records)['all'], flush=True)


if __name__ == '__main__':
    asyncio.run(main())
