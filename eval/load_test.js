// k6 Load Test for Perfect RAG
// Run: k6 run eval/load_test.js
//
// Requirements:
// - k6 installed (https://k6.io/docs/getting-started/installation/)
// - Perfect RAG API running on localhost:8000
//
// Metrics reported:
// - p50/p95/p99 latency
// - Requests per second (RPS)
// - Error rate

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const latencyP50 = new Trend('latency_p50');
const latencyP95 = new Trend('latency_p95');
const latencyP99 = new Trend('latency_p99');

// Test configuration
export const options = {
    scenarios: {
        // Light load test
        light_load: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '30s', target: 5 },
                { duration: '1m', target: 5 },
                { duration: '30s', target: 0 },
            ],
            gracefulRampDown: '10s',
        },
    },
    thresholds: {
        http_req_duration: ['p(50)<500', 'p(95)<2000', 'p(99)<5000'],
        errors: ['rate<0.1'], // Less than 10% errors
    },
};

// Test queries
const queries = [
    'clasificación TNM del cáncer colorrectal',
    'tratamiento de primera línea para Helicobacter pylori',
    'criterios diagnósticos de cirrosis compensada',
    'indicaciones de trasplante hepático',
    'screening de carcinoma hepatocelular',
];

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
    // Pick a random query
    const query = queries[Math.floor(Math.random() * queries.length)];

    // Test chat completions endpoint
    const payload = JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
            { role: 'user', content: query }
        ],
        stream: false,
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
        timeout: '30s',
    };

    const response = http.post(`${BASE_URL}/v1/chat/completions`, payload, params);

    // Check response
    const success = check(response, {
        'status is 200': (r) => r.status === 200,
        'has response': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.choices && body.choices.length > 0;
            } catch {
                return false;
            }
        },
    });

    errorRate.add(!success);

    // Record latency
    if (success) {
        latencyP50.add(response.timings.duration);
        latencyP95.add(response.timings.duration);
        latencyP99.add(response.timings.duration);
    }

    sleep(0.5); // 2 requests per second per VU
}

// Summary function
export function handleSummary(data) {
    const metrics = {
        timestamp: new Date().toISOString(),
        test_duration_seconds: data.state.testRunDurationMs / 1000,
        requests_total: data.metrics.http_reqs.values.count,
        requests_per_second: data.metrics.http_reqs.values.rate,
        latency: {
            p50_ms: data.metrics.http_req_duration.values['p(50)'],
            p95_ms: data.metrics.http_req_duration.values['p(95)'],
            p99_ms: data.metrics.http_req_duration.values['p(99)'],
            avg_ms: data.metrics.http_req_duration.values.avg,
        },
        error_rate: data.metrics.errors ? data.metrics.errors.values.rate : 0,
        iterations: data.metrics.iterations.values.count,
    };

    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        'eval/results/load_test_results.json': JSON.stringify(metrics, null, 2),
    };
}

function textSummary(data, opts = {}) {
    const indent = opts.indent || '  ';
    const colors = opts.enableColors || false;

    let summary = '\n' + '='.repeat(50) + '\n';
    summary += 'LOAD TEST SUMMARY\n';
    summary += '='.repeat(50) + '\n\n';

    summary += `Duration: ${(data.state.testRunDurationMs / 1000).toFixed(1)}s\n`;
    summary += `Total Requests: ${data.metrics.http_reqs.values.count}\n`;
    summary += `Requests/sec: ${data.metrics.http_reqs.values.rate.toFixed(2)}\n\n`;

    summary += 'Latency:\n';
    summary += `  p50: ${data.metrics.http_req_duration.values['p(50)'].toFixed(0)}ms\n`;
    summary += `  p95: ${data.metrics.http_req_duration.values['p(95)'].toFixed(0)}ms\n`;
    summary += `  p99: ${data.metrics.http_req_duration.values['p(99)'].toFixed(0)}ms\n`;
    summary += `  avg: ${data.metrics.http_req_duration.values.avg.toFixed(0)}ms\n\n`;

    const errorRate = data.metrics.errors ? data.metrics.errors.values.rate : 0;
    summary += `Error Rate: ${(errorRate * 100).toFixed(1)}%\n`;

    return summary;
}
