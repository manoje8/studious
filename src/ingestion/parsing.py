import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import logfire
from tqdm import tqdm

from src.common.utils.constants import HTML_FORMATS, OFFICE_FORMATS, TEXT_FORMATS, ParseMethod
from src.ingestion.chunking.chunk import BatchProcess
from src.ingestion.processor import get_parser_method


class Parsing:
    def __init__(
        self,
        parser_method: ParseMethod = ParseMethod.DOCLING,
        max_workers: int = 4,
        show_progress: bool = True,
        timeout_per_file: int = 300,
    ):
        self.parser_method = parser_method
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.timeout_per_file = timeout_per_file

        try:
            self.parser = get_parser_method(self.parser_method)
        except Exception as exec:
            raise ValueError(f"Unsupported parser method {parser_method}") from exec

        if not self.parser.check_installation():
            logfire.warn("Required package is not installed")

    def _supported_extensions(self) -> list[str]:
        return list(HTML_FORMATS | OFFICE_FORMATS | TEXT_FORMATS | {".pdf"})

    def _filter_supported_files(self, file_paths: list[str], recursive: bool = False):
        supported_extensions = set(self._supported_extensions())
        supported_files = []

        for file_path in file_paths:
            path = Path(file_path)
            if path.is_dir():
                if recursive:
                    for inside_path in path.rglob("*"):
                        if (
                            inside_path.is_file()
                            and inside_path.suffix.lower() in supported_extensions
                        ):
                            supported_files.append(str(inside_path))
                else:
                    for inside_path in path.glob("*"):
                        if (
                            inside_path.is_file()
                            and inside_path.suffix.lower() in supported_extensions
                        ):
                            supported_files.append(str(inside_path))
            elif path.is_file():
                if path.suffix.lower() in supported_extensions:
                    supported_files.append(str(path))
                else:
                    logfire.warn(f"Unsupported file format : {file_path}")

            else:
                logfire.warn(f"Path doesn't exist: {path}")

        return supported_files

    async def process_single_file(self, file_path: str | Path, output_dir: str, **kwargs):
        try:
            start_time = time.time()
            path = Path(file_path)

            content_list = await self.parser.parse_doc(
                file_path=path, method=self.parser_method.value, **kwargs
            )

            processing_time = time.time() - start_time

            logfire.info(
                f"Successfully processed {file_path} "
                f"({len(content_list)} content blocks, {processing_time:.2f}s)"
            )

            return True, file_path, None

        except Exception as e:
            error_msg = f"parser failed to process {file_path}: {str(e)}"
            logfire.error(error_msg)
            return False, file_path, error_msg

    async def process_batch(
        self, file_paths: list[str], output_dir: str, recursive: bool = False, **kwargs
    ):
        start_time = time.time()
        supported_files = self._filter_supported_files(file_paths, recursive)

        if not supported_files:
            return BatchProcess(
                successful_files=[],
                failed_files=[],
                total_files=0,
                processing_time=0.0,
                errors={},
                output_dir=output_dir,
            )

        logfire.info(f"Found {len(supported_files)} file to process")

        success_files = []
        failed_files = []
        errors = {}
        pbar = None
        if self.show_progress:
            pbar = tqdm(
                total=len(success_files),
                desc=f"processing files {self.parser_method}",
                unit="file",
            )

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = {
                    executor.submit(
                        self.process_single_file,
                        file_path,
                        output_dir,
                        **kwargs,
                    )
                    for file_path in file_paths
                }

                for result in as_completed(results, timeout=self.timeout_per_file):
                    success, file_path, error_msg = result.result()
                    if success:
                        success_files.append(file_path)
                    else:
                        failed_files.append(file_path)
                        errors[file_path] = error_msg

                    if pbar:
                        pbar.update(1)

        except Exception as e:
            logfire.error(f"batch processing failed: {str(e)}")

            for result in results:
                if not result.done():
                    file_path = results[result]
                    failed_files.append(file_path)
                    errors[file_path] = f"Processing interrupted: {str(e)}"

                    if pbar:
                        pbar.close()

        finally:
            if pbar:
                pbar.close()

        processing_time = time.time() - start_time

        output = BatchProcess(
            successful_files=success_files,
            failed_files=failed_files,
            total_files=len(supported_files),
            processing_time=processing_time,
            errors=errors,
            output_dir=output_dir,
        )

        logfire.info(output.summary())

        return result

    async def process_batch_async(
        self, file_paths: list[str], output_dir: str, recursive: bool = False, **kwargs
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_batch,
            file_paths,
            output_dir,
            recursive,
            **kwargs,
        )
