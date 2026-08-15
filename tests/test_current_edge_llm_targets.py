import json
import unittest
from unittest import mock

from scenes.freeway_traffic.traffic_system import measure_current_edge_llm_targets as targets


class _StreamResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(
            [
                b'data: {"choices":[{"delta":{"content":"A"}}]}\n',
                b"data: [DONE]\n",
            ]
        )


class CurrentEdgeTargetsTests(unittest.TestCase):
    def test_same_protocol_request_disables_template_thinking(self):
        captured = {}

        def open_request(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _StreamResponse()

        with mock.patch.object(targets.urllib.request, "urlopen", side_effect=open_request):
            with mock.patch.object(
                targets.time,
                "perf_counter",
                side_effect=[10.0, 10.01, 10.02],
            ):
                result = targets._openai_stream_chat(
                    "http://127.0.0.1:1234",
                    "test-model",
                    "system",
                    "user",
                    4,
                    30,
                )

        self.assertEqual(result["text"], "A")
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(
            captured["payload"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertIs(captured["payload"]["think"], False)
        self.assertEqual(captured["payload"]["reasoning_effort"], "none")


if __name__ == "__main__":
    unittest.main()
