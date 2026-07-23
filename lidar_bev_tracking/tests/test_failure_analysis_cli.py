import io
import unittest
from contextlib import redirect_stderr

from bev_tracking.error_codes import ErrorCode
from bev_tracking.failure_analysis import (
    DuplicateFrameIdError,
    InvalidFailureAnalysisConfigError,
    SourceConsistencyError,
    SourceContractError,
)
from scripts import run_failure_analysis


class FailureAnalysisCliTest(unittest.TestCase):
    def test_cli_returns_nonzero_and_prints_stable_analysis_error_codes(self):
        errors = [
            SourceContractError("missing required metrics"),
            SourceConsistencyError("summary and CSV disagree"),
            DuplicateFrameIdError("duplicate frame"),
            InvalidFailureAnalysisConfigError("invalid top_k"),
        ]
        original = run_failure_analysis.generate_failure_cases_from_run_directory

        try:
            for error in errors:
                def fail_analysis(*args, _error=error, **kwargs):
                    raise _error

                run_failure_analysis.generate_failure_cases_from_run_directory = fail_analysis
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = run_failure_analysis.main(["--run-dir", "unused"])

                self.assertEqual(exit_code, 1)
                self.assertIn(error.error_code.value, stderr.getvalue())
        finally:
            run_failure_analysis.generate_failure_cases_from_run_directory = original

    def test_real_invalid_top_k_is_rejected_before_source_io(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = run_failure_analysis.main(["--run-dir", "missing-run", "--top-k", "0"])

        self.assertEqual(exit_code, 1)
        self.assertIn(ErrorCode.INVALID_FAILURE_ANALYSIS_CONFIG.value, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
