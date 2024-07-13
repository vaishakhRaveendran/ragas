from __future__ import annotations

import asyncio
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from ragas.exceptions import MaxRetriesExceeded
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


def is_event_loop_running() -> bool:
    try:
        loop = asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


async def run_with_semaphore(semaphore, coro):
    async with semaphore:
        return await coro


@dataclass
class Executor:
    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    jobs: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    run_config: t.Optional[RunConfig] = field(default=None, repr=False)

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        async def wrapped_callable_async(*args, **kwargs):
            result = np.nan
            try:
                if asyncio.iscoroutinefunction(callable):
                    result = await callable(*args, **kwargs)
                else:
                    result = callable(*args, **kwargs)
            except MaxRetriesExceeded as e:
                logger.warning(f"Max retries exceeded for {e.evolution}")
                if self.raise_exceptions:
                    raise
            except Exception as e:
                logger.error(f"Runner in Executor raised an exception: {str(e)}", exc_info=True)
                if self.raise_exceptions:
                    raise
            return counter, result

        return wrapped_callable_async

    def submit(
            self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        callable_with_index = self.wrap_callable_with_index(callable, len(self.jobs))
        self.jobs.append((callable_with_index, args, kwargs, name))

    def results(self) -> t.List[t.Any]:
        if is_event_loop_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise ImportError(
                    "It seems like you're running this in a jupyter-like environment. Please install nest_asyncio with `pip install nest_asyncio` to make it work."
                )
            return asyncio.get_event_loop().run_until_complete(self._aresults())
        else:
            return asyncio.run(self._aresults())

    async def _aresults(self) -> t.List[t.Any]:
        max_workers = (self.run_config or RunConfig()).max_workers
        semaphore = asyncio.Semaphore(max_workers if max_workers > 0 else asyncio.tasks._MAX_WORKERS)

        coros = [run_with_semaphore(semaphore, afunc(*args, **kwargs)) for afunc, args, kwargs, _ in self.jobs]

        results = []
        for future in tqdm(
                asyncio.as_completed(coros),
                desc=self.desc,
                total=len(self.jobs),
                leave=self.keep_progress_bar,
        ):
            try:
                r = await future
                results.append(r)
            except Exception as e:
                logger.error(f"Error processing job: {str(e)}", exc_info=True)
                if self.raise_exceptions:
                    raise

        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]