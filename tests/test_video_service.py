"""Unit tests for the event-driven Selenium video service."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_video_service_module():
    """Load the video service without requiring pyzmq on the test host.

    Args:
        None.

    Returns:
        The imported ``video_service`` module.
    """
    fake_zmq = types.ModuleType("zmq")
    fake_zmq_asyncio = types.ModuleType("zmq.asyncio")
    fake_zmq_asyncio.Context = object
    fake_zmq_asyncio.Socket = object
    fake_zmq.asyncio = fake_zmq_asyncio

    sys.modules.setdefault("zmq", fake_zmq)
    sys.modules.setdefault("zmq.asyncio", fake_zmq_asyncio)

    module_path = Path(__file__).parents[1] / "Video" / "video_service.py"
    spec = importlib.util.spec_from_file_location("selenium_video_service", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


video_service = load_video_service_module()


class VideoServiceReconciliationTests(unittest.IsolatedAsyncioTestCase):
    """Verify recovery from missed event-bus session lifecycle messages."""

    def setUp(self) -> None:
        """Create a video service configured for side-effect-free tests.

        Args:
            None.

        Returns:
            None.
        """
        self.environment = patch.dict(
            os.environ,
            {
                "SE_VIDEO_RECORD_STANDALONE": "true",
                "SE_RECORD_VIDEO": "false",
                "SE_VIDEO_FILE_NAME_SUFFIX": "false",
                "SE_VIDEO_POLL_INTERVAL": "1",
            },
        )
        self.environment.start()
        self.service = video_service.VideoService()
        self.service.node_id = "node-1"

    async def asyncTearDown(self) -> None:
        """Cancel delayed cleanup tasks and restore the test environment.

        Args:
            None.

        Returns:
            None.
        """
        for task in self.service._cleanup_tasks:
            task.cancel()
        if self.service._cleanup_tasks:
            await video_service.asyncio.gather(*self.service._cleanup_tasks, return_exceptions=True)
        self.environment.stop()

    @staticmethod
    def status_payload(*sessions: dict) -> dict:
        """Build a standalone Node status payload with active sessions.

        Args:
            *sessions: Session dictionaries to place in Node slots.

        Returns:
            A decoded Selenium Node status response.
        """
        return {
            "value": {
                "nodes": [
                    {
                        "id": "node-1",
                        "slots": [{"session": session} for session in sessions],
                    }
                ]
            }
        }

    async def test_reconciliation_recovers_missed_create_and_close_events(self) -> None:
        """Recover a session after create and close events are both missed.

        Args:
            None.

        Returns:
            None.
        """
        active_payload = self.status_payload(
            {
                "sessionId": "session-1",
                "capabilities": {"se:name": "recovered-session"},
            }
        )
        payloads = iter([active_payload, active_payload, self.status_payload(), self.status_payload()])
        self.service._fetch_node_status = lambda: next(payloads)

        self.assertTrue(await self.service.reconcile_once())
        recovered_session = self.service.sessions["session-1"]
        self.assertEqual(recovered_session.status, video_service.SessionStatus.CREATED)
        self.assertEqual(recovered_session.video_file, "recovered-session.mp4")

        self.assertTrue(await self.service.reconcile_once())
        self.assertIs(self.service.sessions["session-1"], recovered_session)

        self.assertTrue(await self.service.reconcile_once())
        self.assertEqual(recovered_session.status, video_service.SessionStatus.CREATED)

        self.assertTrue(await self.service.reconcile_once())
        self.assertEqual(recovered_session.status, video_service.SessionStatus.CLOSED)
        self.assertEqual(len(self.service._cleanup_tasks), 1)

    async def test_status_failure_does_not_close_an_active_recording(self) -> None:
        """Keep sessions open while the Node status endpoint is unavailable.

        Args:
            None.

        Returns:
            None.
        """
        self.service.sessions["session-1"] = video_service.SessionState(
            session_id="session-1",
            video_file="session-1.mp4",
        )
        self.service._fetch_node_status = lambda: None

        self.assertFalse(await self.service.reconcile_once())
        self.assertEqual(self.service.sessions["session-1"].status, video_service.SessionStatus.CREATED)
        self.assertEqual(self.service.missing_status_counts, {})

    async def test_duplicate_close_event_is_idempotent(self) -> None:
        """Schedule session cleanup only once when close is repeated.

        Args:
            None.

        Returns:
            None.
        """
        self.service.sessions["session-1"] = video_service.SessionState(
            session_id="session-1",
            video_file="session-1.mp4",
        )
        close_event = {
            "sessionId": "session-1",
            "nodeId": "node-1",
            "reason": video_service.SessionClosedReason.QUIT_COMMAND.value,
        }

        await self.service.handle_session_closed(close_event)
        await self.service.handle_session_closed(close_event)

        self.assertEqual(self.service.sessions["session-1"].status, video_service.SessionStatus.CLOSED)
        self.assertEqual(len(self.service._cleanup_tasks), 1)


if __name__ == "__main__":
    unittest.main()
