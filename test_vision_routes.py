import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import server


class VisionRouteTests(unittest.TestCase):
    def setUp(self):
        self.token = "vision-route-test-browser-session"
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.App)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]

        with server.STATE_LOCK:
            server.SESSIONS[self.token] = time.time() + 600
            server.VISION_SESSIONS.pop(self.token, None)

        self.observer_patch = patch("server.inspect_visual")
        self.ask_patch = patch("server.ask")
        self.inspect_visual = self.observer_patch.start()
        self.ask = self.ask_patch.start()
        self.addCleanup(self.ask_patch.stop)
        self.addCleanup(self.observer_patch.stop)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=3)
        with server.STATE_LOCK:
            server.SESSIONS.pop(self.token, None)
            server.VISION_SESSIONS.pop(self.token, None)

    def post(self, route, origin="http://172.30.50.10:8765",
             cookie=True, payload=None):
        headers = {"Content-Type": "application/json"}
        if origin is not None:
            headers["Origin"] = origin
        if cookie:
            headers["Cookie"] = "baby_approval_session=" + self.token

        request = urllib.request.Request(
            self.url + route,
            data=json.dumps({} if payload is None else payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def seed_session(self, **overrides):
        session = {
            "token": "test-windows-vision-session-token",
            "expires_at": time.time() + 60,
            "captures": 0,
            "last_capture": 0.0,
        }
        session.update(overrides)
        with server.STATE_LOCK:
            server.VISION_SESSIONS[self.token] = session
        return session

    def test_vision_requires_exact_origin(self):
        status, body = self.post(
            "/api/vision/start", origin="http://evil.example"
        )
        self.assertEqual(status, 403)
        self.inspect_visual.assert_not_called()

    def test_vision_requires_browser_session(self):
        status, body = self.post("/api/vision/start", cookie=False)
        self.assertEqual(status, 403)
        self.inspect_visual.assert_not_called()

    def test_start_creates_bounded_nonpersistent_session(self):
        self.inspect_visual.return_value = {
            "session_token": "test-windows-vision-session-token",
            "expires_in": 60,
            "screenshot_persistence": False,
        }
        status, body = self.post("/api/vision/start")
        self.assertEqual(status, 200)
        self.assertTrue(body["active"])
        self.assertFalse(body["screenshot_persistence"])
        self.assertEqual(body["expires_in"], 60)
        self.assertIn(self.token, server.VISION_SESSIONS)

    def test_expired_session_is_rejected(self):
        self.seed_session(expires_at=time.time() - 1)
        status, body = self.post("/api/vision/capture")
        self.assertEqual(status, 410)
        self.inspect_visual.assert_not_called()
        self.ask.assert_not_called()

    def test_capture_rate_limit_is_enforced(self):
        self.seed_session(last_capture=time.time())
        status, body = self.post("/api/vision/capture")
        self.assertEqual(status, 429)
        self.inspect_visual.assert_not_called()
        self.ask.assert_not_called()

    def test_capture_limit_is_enforced(self):
        self.seed_session(captures=server.VISION_MAX_CAPTURES)
        status, body = self.post("/api/vision/capture")
        self.assertEqual(status, 429)
        self.inspect_visual.assert_not_called()
        self.ask.assert_not_called()

    def test_capture_returns_image_and_nonempty_assessment(self):
        self.seed_session()
        self.inspect_visual.return_value = {
            "image_b64": "/9j/eP/Z",
            "media_type": "image/jpeg",
            "width": 640,
            "height": 360,
            "captured_at": 1234567890.0,
        }
        self.ask.return_value = {
            "message": {"content": "A desktop application is visible."}
        }

        status, body = self.post("/api/vision/capture")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(
            body["analysis"], "A desktop application is visible."
        )
        self.assertEqual(body["media_type"], "image/jpeg")
        self.assertFalse(body["screenshot_persistence"])
        self.assertEqual(body["captures_remaining"],
                         server.VISION_MAX_CAPTURES - 1)
        self.assertEqual(
            server.VISION_SESSIONS[self.token]["visual_context"],
            {
                "image_b64": "/9j/eP/Z",
                "media_type": "image/jpeg",
                "captured_at": 1234567890.0,
            },
        )

        model_payload = self.ask.call_args.args[1]
        self.assertEqual(model_payload["model"], "qwen3-vl:4b")
        self.assertEqual(model_payload["options"]["num_predict"], 1024)
        self.assertFalse(model_payload["think"])

    def test_ask_requires_origin_and_browser_authentication(self):
        self.seed_session(visual_context={
            "image_b64": "test-image",
            "media_type": "image/jpeg",
            "captured_at": 123.0,
        })
        status, _ = self.post(
            "/api/vision/ask",
            origin="http://evil.example",
            payload={"question": "What is visible?"},
        )
        self.assertEqual(status, 403)

        status, _ = self.post(
            "/api/vision/ask",
            cookie=False,
            payload={"question": "What is visible?"},
        )
        self.assertEqual(status, 403)
        self.ask.assert_not_called()

    def test_ask_requires_a_captured_screenshot(self):
        self.seed_session()
        status, body = self.post(
            "/api/vision/ask",
            payload={"question": "What is visible?"},
        )
        self.assertEqual(status, 409)
        self.assertNotIn("answer", body)
        self.ask.assert_not_called()

    def test_ask_uses_retained_image_and_returns_only_answer(self):
        session = self.seed_session(visual_context={
            "image_b64": "retained-image-data",
            "media_type": "image/jpeg",
            "captured_at": 123.0,
        })
        self.ask.return_value = {
            "message": {"content": "A terminal window is visible."},
            "thinking": "This must not be returned to the browser.",
        }

        status, body = self.post(
            "/api/vision/ask",
            payload={"question": "What application is open?"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "ok": True,
            "answer": "A terminal window is visible.",
        })

        payload = self.ask.call_args.args[1]
        self.assertEqual(payload["model"], "qwen3-vl:4b")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 1024)
        self.assertEqual(
            payload["messages"][0]["images"],
            ["retained-image-data"],
        )
        self.assertIn(
            "What application is open?",
            payload["messages"][0]["content"],
        )
        self.assertIn("untrusted", payload["messages"][0]["content"])
        self.assertNotIn("thinking", body)
        self.assertIn("visual_context", session)

    def test_ask_rejects_invalid_questions(self):
        self.seed_session(visual_context={
            "image_b64": "retained-image-data",
            "media_type": "image/jpeg",
            "captured_at": 123.0,
        })
        invalid_payloads = [
            {"question": ""},
            {"question": "   "},
            {"question": "x" * 1001},
            {"question": "valid", "extra": True},
            {"question": 123},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload_type=type(payload.get("question")).__name__,
                              length=len(str(payload.get("question")))):
                status, _ = self.post(
                    "/api/vision/ask",
                    payload=payload,
                )
                self.assertEqual(status, 400)
        self.ask.assert_not_called()

    def test_stop_removes_session_and_its_visual_context(self):
        self.seed_session(visual_context={
            "image_b64": "retained-image-data",
            "media_type": "image/jpeg",
            "captured_at": 123.0,
        })
        status, body = self.post("/api/vision/stop")
        self.assertEqual(status, 200)
        self.assertFalse(body["active"])
        self.assertNotIn(self.token, server.VISION_SESSIONS)
        self.inspect_visual.assert_called_once_with(
            "vision_stop",
            "test-windows-vision-session-token",
        )

    def test_unexpected_ask_error_logs_metadata_without_request_data(self):
        self.seed_session(visual_context={
            "image_b64": "PRIVATE_IMAGE_PAYLOAD",
            "media_type": "image/jpeg",
            "captured_at": 123.0,
        })
        self.ask.side_effect = RuntimeError(
            "PRIVATE_QUESTION_AND_IMAGE_MUST_NOT_APPEAR_IN_LOGS"
        )

        with patch("builtins.print") as diagnostic_log:
            status, body = self.post(
                "/api/vision/ask",
                payload={"question": "PRIVATE_QUESTION_AND_IMAGE_MUST_NOT_APPEAR_IN_LOGS"},
            )

        self.assertEqual(status, 502)
        self.assertEqual(
            body,
            {"error": "The operation was rejected or could not be completed"},
        )
        diagnostic_log.assert_called_once()
        logged = " ".join(str(arg) for arg in diagnostic_log.call_args.args)
        self.assertIn("RuntimeError", logged)
        self.assertIn("/api/vision/ask", logged)
        self.assertNotIn("PRIVATE_QUESTION", logged)
        self.assertNotIn("PRIVATE_IMAGE", logged)

    def test_empty_model_assessment_is_not_reported_as_success(self):
        self.seed_session()
        self.inspect_visual.return_value = {
            "image_b64": "/9j/eP/Z",
            "media_type": "image/jpeg",
            "width": 640,
            "height": 360,
            "captured_at": 1234567890.0,
        }
        self.ask.return_value = {
            "message": {"content": "   "}
        }

        status, body = self.post("/api/vision/capture")
        self.assertEqual(status, 502)
        self.assertNotIn("analysis", body)
        self.assertNotIn("image_b64", body)


if __name__ == "__main__":
    unittest.main()
