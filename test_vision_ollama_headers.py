import json
import unittest
from unittest.mock import patch

import server


class OllamaRequestHeaderTests(unittest.TestCase):
    @patch("server.urllib.request.urlopen")
    def test_json_payload_sets_application_json_content_type(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = (
            b'{"message":{"content":"vision response"}}'
        )

        payload = {
            "model": "qwen3-vl:4b",
            "messages": [{
                "role": "user",
                "content": "Describe this image",
                "images": ["ZmFrZQ=="],
            }],
            "stream": False,
        }

        result = server.ask("/api/chat", payload)

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.get_header("Content-type"),
            "application/json",
        )
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            payload,
        )
        self.assertEqual(
            result["message"]["content"],
            "vision response",
        )

    @patch("server.urllib.request.urlopen")
    def test_get_without_payload_remains_bodyless(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = b'{}'

        result = server.ask("/api/version")

        request = urlopen.call_args.args[0]
        self.assertIsNone(request.data)
        self.assertIsNone(request.get_header("Content-type"))
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
