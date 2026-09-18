"""Unit tests for the event-driven Selenium video service."""

import asyncio
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch


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

    async def test_shutdown_finishes_reconciled_recording_before_cleanup(self) -> None:
        """Finish FFmpeg and queue its video before shutdown cleanup runs.

        Args:
            None.

        Returns:
            None.
        """
        stop_started = asyncio.Event()
        finish_stop = asyncio.Event()
        poll_finished = asyncio.Event()

        async def drain() -> None:
            """Pause finalization after the session's process reference is cleared.

            Args:
                None.

            Returns:
                None.
            """
            stop_started.set()
            await finish_stop.wait()

        async def poll(timeout: int) -> bool:
            """Wake the subscriber when shutdown is requested during finalization.

            Args:
                timeout: Socket polling timeout in milliseconds, unused by this fake.

            Returns:
                False because no event-bus message is available.
            """
            await self.service.shutdown_event.wait()
            poll_finished.set()
            return False

        process = Mock()
        process.stdin.is_closing.return_value = False
        process.stdin.drain = AsyncMock(side_effect=drain)
        process.communicate = AsyncMock(return_value=(b"", b""))
        process.returncode = 0
        session = video_service.SessionState(
            session_id="session-1",
            status=video_service.SessionStatus.RECORDING,
            video_file="session-1.mp4",
            ffmpeg_process=process,
        )
        self.service.sessions[session.session_id] = session
        self.service.missing_status_counts[session.session_id] = 1
        self.service._fetch_node_status = Mock(return_value=self.status_payload())
        self.service.upload_enabled = True
        self.service.upload_destination = "test:videos"

        subscriber = Mock(poll=AsyncMock(side_effect=poll))
        context = Mock()
        context.socket.return_value = subscriber
        with patch.multiple(
            video_service.zmq, SUB=1, LINGER=2, SUBSCRIBE=3, ZMQError=OSError, create=True
        ), patch.object(video_service.zmq.asyncio, "Context", return_value=context), patch.object(
            video_service.Path, "exists", return_value=True
        ), patch.object(
            self.service, "wait_for_file_integrity", new_callable=AsyncMock, return_value=True
        ) as integrity_check:
            subscriber_task = asyncio.create_task(self.service.subscribe_events())
            try:
                await asyncio.wait_for(stop_started.wait(), timeout=2)
                self.assertIsNone(session.ffmpeg_process)
                self.service.shutdown_event.set()
                await asyncio.wait_for(poll_finished.wait(), timeout=2)
                self.assertFalse(self.service.recorder_done.is_set())
            finally:
                finish_stop.set()
                self.service.shutdown_event.set()
                await asyncio.wait_for(subscriber_task, timeout=2)

            process.communicate.assert_awaited_once()
            integrity_check.assert_awaited_once()
            self.assertEqual(self.service.recorded_count, 1)
            self.assertTrue(self.service.recorder_done.is_set())
            await self.service.cleanup()

        upload = self.service.upload_queue.get_nowait()
        self.assertEqual(upload.session_id, session.session_id)
        self.assertEqual(upload.destination, "test:videos")
        self.assertIsNone(self.service.upload_queue.get_nowait())
        self.assertTrue(self.service.upload_queue.empty())
        subscriber.close.assert_called_once()
        context.term.assert_called_once()

    async def test_subscriber_failure_signals_reconciler_shutdown(self) -> None:
        """Stop an idle reconciler when the subscriber exits unexpectedly.

        Args:
            None.

        Returns:
            None.
        """
        reconcile_started = asyncio.Event()
        self.service.node_poll_interval = 3600
        self.service.reconcile_once = AsyncMock(side_effect=reconcile_started.set)

        async def poll(timeout: int) -> bool:
            """Fail the subscriber after its reconciliation task has started.

            Args:
                timeout: Socket polling timeout in milliseconds, unused by this fake.

            Returns:
                Never returns; raises a simulated subscriber failure.
            """
            await reconcile_started.wait()
            raise RuntimeError("subscriber failed")

        subscriber = Mock(poll=AsyncMock(side_effect=poll))
        context = Mock()
        context.socket.return_value = subscriber
        with patch.multiple(
            video_service.zmq, SUB=1, LINGER=2, SUBSCRIBE=3, ZMQError=OSError, create=True
        ), patch.object(video_service.zmq.asyncio, "Context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, "subscriber failed"):
                await asyncio.wait_for(self.service.subscribe_events(), timeout=2)

        self.assertTrue(self.service.shutdown_event.is_set())
        self.assertTrue(self.service.recorder_done.is_set())
        self.service.reconcile_once.assert_awaited_once()
        subscriber.close.assert_called_once()
        context.term.assert_called_once()

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
