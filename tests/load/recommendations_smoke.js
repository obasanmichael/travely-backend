import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 5,
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<2000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://127.0.0.1:8000";

export default function () {
  const health = http.get(`${BASE_URL}/health`);
  check(health, { "health ok": (r) => r.status === 200 });

  const payload = JSON.stringify({
    budget: 20000,
    destination_type: "Nature/Adventure",
    activity_type: "Hiking",
  });

  const res = http.post(`${BASE_URL}/recommendations`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(res, {
    "recommendations 200": (r) => r.status === 200,
    "has results": (r) => JSON.parse(r.body).recommendations.length > 0,
  });

  sleep(1);
}
