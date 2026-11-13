"""Offline regression tests for deadlines and independent evidence handling."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import hw3


class SQLTool:
    def __init__(self, missing=None):
        self.missing = missing

    async def ainvoke(self, arguments):
        query = arguments['query']
        if 'JOIN industry_medians' in query:
            return 'industry | median\n(no rows)' if self.missing == 'industry' else 'industry | median\nExample | 15'
        half = '2025-07-01' in query or '2026-07-01' in query
        previous = '2024-01-01' in query
        if (half and self.missing == 'half') or (previous and self.missing == 'previous'):
            return 'SQL error: simulated unavailable period'
        months = 6 if half else 12
        cost = 90 if previous else 80
        return ('revenue | cost_of_sales | selling_admin | research_development | cost_months\n'
                f'100 | {cost} | 0 | 0 | {months}')


class InterviewTool:
    async def ainvoke(self, arguments):
        return 'Employees report fair treatment and good working conditions.'


class Model:
    def __init__(self, delay=0, invalid_first=False):
        self.delay = delay
        self.invalid_first = invalid_first
        self.calls = 0
        self.active = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        self.active += 1
        try:
            await asyncio.sleep(self.delay)
            content = 'invalid' if self.invalid_first and self.calls == 1 else json.dumps(
                {'verdict': 'PASS', 'reason': 'Employees describe fair treatment.'})
            return SimpleNamespace(content=content)
        finally:
            self.active -= 1


def company(missing=None):
    return [None, None, SQLTool(missing), InterviewTool()]


class ResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def assess(self, missing=None):
        result = await hw3.analyze_companies(Model(), {'example': company(missing)})
        return result['example']

    async def test_healthy(self):
        self.assertEqual((await self.assess())['points'], {5: 'PASS', 6: 'PASS', 7: 'PASS'})

    async def test_half_year_error_preserves_all_verdicts(self):
        result = await self.assess('half')
        self.assertEqual(result['points'], {5: 'PASS', 6: 'PASS', 7: 'PASS'})
        self.assertIn('Supplementary half-year evidence unavailable', result['summary'])

    async def test_missing_industry_does_not_change_point6(self):
        self.assertEqual((await self.assess('industry'))['points'], {5: 'FAIL', 6: 'PASS', 7: 'PASS'})

    async def test_missing_previous_year_does_not_change_point5(self):
        self.assertEqual((await self.assess('previous'))['points'], {5: 'PASS', 6: 'FAIL', 7: 'PASS'})

    async def test_request_deadline_bounds_retries(self):
        model = Model(delay=10)
        with patch.multiple(hw3, REQUEST_TIMEOUT_SECONDS=0.02, RETRY_DELAY_SECONDS=0):
            result = await asyncio.wait_for(hw3.analyze_companies(model, {'example': company()}), 1)
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.active, 0)
        self.assertEqual(result['example']['points'], {5: 'PASS', 6: 'PASS', 7: 'FAIL'})
        self.assertIn('TimeoutError', result['example']['summary'])

    async def test_invalid_output_has_one_repair_attempt(self):
        model = Model(invalid_first=True)
        with patch.object(hw3, 'RETRY_DELAY_SECONDS', 0):
            result = await hw3.analyze_companies(model, {'example': company()})
        self.assertEqual(model.calls, 2)
        self.assertEqual(result['example']['points'][7], 'PASS')

    async def test_company_deadline_preserves_completed_points(self):
        model = Model(delay=10)
        with patch.object(hw3, 'COMPANY_TIMEOUT_SECONDS', 0.03):
            result = await asyncio.wait_for(hw3.analyze_companies(model, {'example': company()}), 1)
        self.assertEqual(result['example']['points'], {5: 'PASS', 6: 'PASS', 7: 'FAIL'})
        self.assertIn('Company time limit reached', result['example']['summary'])
        self.assertEqual(model.active, 0)

    async def test_batch_deadline_includes_waiting_companies(self):
        model = Model(delay=10)
        companies = {name: company() for name in ('active', 'queued')}
        with patch.multiple(hw3, COMPANY_CONCURRENCY=1, BATCH_TIMEOUT_SECONDS=0.03):
            result = await asyncio.wait_for(hw3.analyze_companies(model, companies), 1)
        self.assertEqual(set(result), set(companies))
        self.assertEqual(result['active']['points'], {5: 'PASS', 6: 'PASS', 7: 'FAIL'})
        self.assertEqual(result['queued']['points'], {5: 'FAIL', 6: 'FAIL', 7: 'FAIL'})
        self.assertIn('Batch time limit reached', result['queued']['summary'])
        self.assertEqual(model.active, 0)

    async def test_empty_batch(self):
        self.assertEqual(await hw3.analyze_companies(Model(), {}), {})

    def test_sdk_uses_explicit_timeout_without_nested_retries(self):
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'offline-test-key'}):
            model = hw3.build_agent()
        self.assertEqual(model.request_timeout, 45)
        self.assertEqual(model.max_retries, 0)
        self.assertEqual(model.async_client._client.timeout, 45)


if __name__ == '__main__':
    unittest.main()
