import concurrent.futures
import hashlib
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Union

import pyghidra  # noqa
from loguru import logger

if TYPE_CHECKING:
    from ghidra.app.decompiler import DecompInterface
    from ghidra.base.project import GhidraProject
    from ghidra.framework.model import DomainFile
    from ghidra.program.flatapi import FlatProgramAPI
    from ghidra.program.model.listing import Program


@dataclass
class ProgramInfo:
    """Information about a loaded program"""

    name: str
    program: "Program"
    flat_api: "FlatProgramAPI | None"
    decompiler: "DecompInterface"
    metadata: dict  # Ghidra program metadata
    ghidra_analysis_complete: bool
    file_path: Path | None = None
    load_time: float | None = None

    @property
    def analysis_complete(self) -> bool:
        """Check if Ghidra analysis is complete."""
        return self.ghidra_analysis_complete


class PyGhidraContext:
    """
    Manages a Ghidra project, including its creation, program imports, and cleanup.
    """

    # Lock file patterns used by Ghidra for project locking
    _LOCK_FILE_PATTERNS: ClassVar[list[str]] = ["*.lock", "*.lock~"]

    # Maximum retry attempts after LockException (1 retry = 2 total attempts).
    # This handles stale locks from crashes while avoiding infinite loops
    # if another process legitimately holds the lock.
    _MAX_LOCK_RETRIES = 1

    def __init__(
        self,
        project_name: str,
        project_path: str | Path,
        force_analysis: bool = False,
        verbose_analysis: bool = False,
        no_symbols: bool = False,
        gdts: list | None = None,
        program_options: dict | None = None,
        gzfs_path: str | Path | None = None,
        threaded: bool = True,
        max_workers: int | None = None,
        wait_for_analysis: bool = False,
    ):
        """
        Initializes a new Ghidra project context.

        Args:
            project_name: The name of the Ghidra project.
            project_path: The directory where the project will be created.
            force_analysis: Force a new binary analysis each run.
            verbose_analysis: Verbose logging for analysis step.
            no_symbols: Turn off symbols for analysis.
            gdts: List of paths to GDT files for analysis.
            program_options: Dictionary with program options (custom analyzer settings).
            gzfs_path: Location to store GZFs of analyzed binaries.
            threaded: Use threading during analysis.
            max_workers: Number of workers for threaded analysis.
            wait_for_analysis: Wait for initial project analysis to complete.
        """
        from ghidra.base.project import GhidraProject

        self.project_name = project_name
        self.project_path = Path(project_path)
        self.project: GhidraProject = self._get_or_create_project()

        self.programs: dict[str, ProgramInfo] = {}
        self._init_project_programs()

        # From GhidraDiffEngine
        self.force_analysis = force_analysis
        self.verbose_analysis = verbose_analysis
        self.no_symbols = no_symbols
        self.gdts = gdts if gdts is not None else []
        self.program_options = program_options
        self.gzfs_path = Path(gzfs_path) if gzfs_path else self.project_path / "gzfs"
        if self.gzfs_path:
            self.gzfs_path.mkdir(exist_ok=True, parents=True)

        self.threaded = threaded
        cpu_count = multiprocessing.cpu_count() or 4
        self.max_workers = max_workers if max_workers else cpu_count

        if not self.threaded:
            logger.warning("--no-threaded flag forcing max_workers to 1")
            self.max_workers = 1
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            if self.threaded
            else None
        )
        self.import_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1) if self.threaded else None
        )
        self.wait_for_analysis = wait_for_analysis

    def close(self, save: bool = True):
        """Saves changes to all open programs and closes the project."""
        # Shutdown thread pools FIRST before closing programs
        # Background tasks may still need to access programs/project
        if self.import_executor:
            self.import_executor.shutdown(wait=True)

        if self.executor:
            self.executor.shutdown(wait=True)

        # Now safe to close all programs
        for _program_name, program_info in self.programs.items():
            program = program_info.program
            self.project.close(program)

        # Close the Ghidra project
        self.project.close()
        logger.info(f"Project {self.project_name} closed.")

    def _get_or_create_project(self) -> "GhidraProject":
        """
        Creates a new Ghidra project if it doesn't exist, otherwise opens the existing project.

        Handles stale lock files from previous crashes by automatically cleaning and retrying.

        Returns:
            The Ghidra project object.

        Raises:
            RuntimeError: If unable to open project after lock cleanup and retry.
        """
        from ghidra.base.project import GhidraProject
        from ghidra.framework.model import ProjectLocator
        from ghidra.framework.store import LockException

        project_dir = self.project_path / self.project_name
        project_dir.mkdir(exist_ok=True, parents=True)
        project_dir_str = str(project_dir.absolute())
        locator = ProjectLocator(project_dir_str, self.project_name)

        if not locator.exists():
            logger.info(f"Creating new project: {self.project_name}")
            return GhidraProject.createProject(project_dir_str, self.project_name, False)

        # Try opening with stale lock recovery
        for attempt in range(self._MAX_LOCK_RETRIES + 1):
            try:
                logger.info(f"Opening existing project: {self.project_name}")
                return GhidraProject.openProject(project_dir_str, self.project_name, True)
            except LockException:
                if attempt >= self._MAX_LOCK_RETRIES:
                    raise RuntimeError(
                        f"Failed to open project '{self.project_name}' "
                        f"after {attempt + 1} attempt(s). "
                        f"Another Ghidra instance may be using this project."
                    ) from None

                # List lock files for diagnostic info
                lock_files = [
                    f for pattern in self._LOCK_FILE_PATTERNS
                    for f in project_dir.glob(pattern)
                ]
                logger.warning(
                    f"Project locked. Found {len(lock_files)} lock file(s): "
                    f"{[f.name for f in lock_files]}. Cleaning stale locks..."
                )
                self._clean_lock_files(project_dir)

    @staticmethod
    def _clean_lock_files(project_dir: Path) -> list[str]:
        """
        Remove stale Ghidra lock files from a project directory.

        WARNING: This method deletes files without additional confirmation.
        In multi-user environments, ensure no other Ghidra instance is actively
        using the project before calling this method.

        This is called when opening a project fails due to LockException,
        typically caused by previous crashes or forced process termination.

        Args:
            project_dir: Path to the Ghidra project directory.

        Returns:
            List of successfully removed lock file names.
        """
        removed = []

        for pattern in PyGhidraContext._LOCK_FILE_PATTERNS:
            for lock_file in project_dir.glob(pattern):
                try:
                    lock_file.unlink()
                    removed.append(lock_file.name)
                    logger.info(f"Removed lock file: {lock_file}")
                except OSError as e:
                    logger.warning(f"Could not remove lock file {lock_file}: {e}")

        if removed:
            logger.info(f"Successfully cleaned {len(removed)} lock file(s): {', '.join(removed)}")
        else:
            logger.warning("No lock files were removed (may be held by another process)")

        return removed

    def _init_project_programs(self):
        """
        Initializes the programs dictionary with existing programs in the project.
        """
        from ghidra.program.model.listing import Program

        all_binary_paths = self.list_binaries()
        for binary_path_s in all_binary_paths:
            # Get the parent folder path and filename from Ghidra pathname
            # Ghidra pathnames use '/' as separator and are relative to project root
            if '/' in binary_path_s:
                folder_path, program_name = binary_path_s.rsplit('/', 1)
                # Ensure folder_path starts with '/' for absolute path within project
                if not folder_path.startswith('/'):
                    folder_path = '/' + folder_path
            else:
                folder_path = '/'
                program_name = binary_path_s

            program: Program = self.project.openProgram(
                folder_path, program_name, False
            )
            program_info = self._init_program_info(program)
            self.programs[binary_path_s] = program_info

    def list_binaries(self) -> list[str]:
        """List all the binaries within the Ghidra project."""

        def list_folder_contents(folder) -> list[str]:
            names: list[str] = []
            for subfolder in folder.getFolders():
                names.extend(list_folder_contents(subfolder))

            names.extend([f.getPathname() for f in folder.getFiles()])
            return names

        return list_folder_contents(self.project.getRootFolder())

    def list_binary_domain_files(self) -> list["DomainFile"]:
        """Return a list of DomainFile objects for all binaries in the project.

        This mirrors `list_binaries` but returns the DomainFile objects themselves
        (filtered by content type == "Program").
        """

        from ghidra.framework.model import DomainFile

        def list_folder_domain_files(folder) -> list["DomainFile"]:
            files: list[DomainFile] = []
            for subfolder in folder.getFolders():
                files.extend(list_folder_domain_files(subfolder))

            files.extend([f for f in folder.getFiles() if f.getContentType() == "Program"])
            return files

        return list_folder_domain_files(self.project.getRootFolder())

    def delete_program(self, program_name: str) -> bool:
        """
        Deletes a program from the Ghidra project and saves the project.

        Args:
            program_name: The name of the program to delete.

        Returns:
            True if the program was deleted successfully, False otherwise.
        """
        program_info = self.programs.get(program_name)
        if not program_info:
            available_progs = list(self.programs.keys())
            raise ValueError(
                f"Binary {program_name} not found. Available binaries: {available_progs}"
            )
        else:
            logger.info(f"Deleting program: {program_name}")
            try:
                program_to_delete: Program = program_info.program
                program_to_delete_df: DomainFile = program_to_delete.getDomainFile()
                self.project.close(program_to_delete)
                program_to_delete_df.delete()
                # clean up program reference
                del self.programs[program_name]
                return True
            except Exception as e:
                logger.error(f"Error deleting program '{program_name}': {e}")
                return False

    def import_binary(
        self, binary_path: str | Path, analyze: bool = False, relative_path: Path | None = None
    ) -> None:
        """
        Imports a single binary into the project.

        Args:
            binary_path: Path to the binary file.
            analyze: Perform analysis on this binary. Useful if not importing in bulk.
            relative_path: Relative path within the project hierarchy (Path("bin") or Path("lib")).

        Returns:
            None
        """
        from ghidra.program.model.listing import Program

        binary_path = Path(binary_path)
        if binary_path.is_dir():
            return self.import_binaries([binary_path], analyze=analyze)

        program_name = PyGhidraContext._gen_unique_bin_name(binary_path)

        program: Program
        root_folder = self.project.getRootFolder()

        # Create folder hierarchy if relative_path is provided
        if relative_path:
            ghidra_folder = self._create_folder_hierarchy(root_folder, relative_path)
        else:
            ghidra_folder = root_folder

        # Check if program already exists at this location
        full_path = str(Path(ghidra_folder.pathname) / program_name)
        if self.programs.get(full_path):
            logger.info(f"Opening existing program: {program_name}")
            program = self.programs[full_path].program
            program_info = self.programs[full_path]
        else:
            logger.info(f"Importing new program: {program_name}")
            program = self.project.importProgram(binary_path)
            program.name = program_name
            if program:
                self.project.saveAs(program, ghidra_folder.pathname, program_name, True)

            program_info = self._init_program_info(program)
            self.programs[program.getDomainFile().pathname] = program_info

        if not program:
            raise ImportError(f"Failed to import binary: {binary_path}")

        if analyze:
            self.analyze_program(program_info.program)

        logger.info(f"Program {program_name} is ready for use.")

    @staticmethod
    def _create_folder_hierarchy(root_folder, relative_path: Path):
        """
        Recursively creates folder hierarchy in Ghidra project.

        Args:
            root_folder: The root folder of the Ghidra project.
            relative_path: The path hierarchy to create (e.g., Path("bin/subfolder")).

        Returns:
            The folder object at the end of the hierarchy.
        """
        current_folder = root_folder

        # Split the path into parts and iterate through them
        for part in relative_path.parts:
            existing_folder = current_folder.getFolder(part)
            if existing_folder:
                current_folder = existing_folder
                logger.debug(f"Using existing folder: {part}")
            else:
                current_folder = current_folder.createFolder(part)
                logger.debug(f"Created folder: {part}")

        return current_folder

    def import_binaries(self, binary_paths: list[str | Path], analyze: bool = False):
        """
        Imports a list of binaries into the project.
        If an entry is a directory it will be walked recursively
        and all regular files found will be imported, preserving directory structure.

        Note: Ghidra does not directly support multithreaded importing into the same project.
        Args:
            binary_paths: A list of paths to the binary files or directories.
            analyze: Whether to analyze the imported binaries.
        """
        resolved_paths: list[Path] = [Path(p) for p in binary_paths]

        # Tuple of (full system path, relative path from provided path)
        files_to_import: list[tuple[Path, Path | None]] = []
        for p in resolved_paths:
            if p.is_dir():
                logger.info(f"Discovering files in directory: {p}")
                for f in p.rglob("*"):
                    if f.is_file() and self._is_binary_file(f):
                        # Store the relative path (e.g., "bin" or "lib/subfolder")
                        relative = f.relative_to(p).parent
                        files_to_import.append((f, relative))
            elif p.is_file() and self._is_binary_file(p):
                files_to_import.append((p, None))

        if not files_to_import:
            logger.info("No files found to import from provided paths.")
            return

        logger.info(f"Importing {len(files_to_import)} binary files into project...")
        for bin_path, relative_path in files_to_import:
            try:
                self.import_binary(bin_path, analyze=analyze, relative_path=relative_path)
            except Exception as e:
                logger.error(f"Failed to import {bin_path}: {e}")
                # continue importing remaining files

    @staticmethod
    def _is_binary_file(path: Path) -> bool:
        """
        Quick header-based check for common binary formats.
        Recognizes ELF (0x7f 'ELF') and PE ('MZ' DOS header) signatures.
        Returns False on read errors or unknown signatures.
        """
        try:
            with path.open("rb") as f:
                header = f.read(4)
                if not header:
                    return False
                # ELF: 0x7f 'ELF'
                if header.startswith(b"\x7fELF"):
                    return True
                # PE executables typically start with 'MZ' (DOS stub)
                if header.startswith(b"MZ"):
                    return True
                return False
        except Exception as e:
            logger.debug(f"Could not read file header for {path}: {e}")
            return False

    def _import_callback(self, future: concurrent.futures.Future):
        """
        A callback function to handle results or exceptions from the import task.
        """
        try:
            result = future.result()
            logger.info(f"Background import task completed successfully. Result: {result}")
        except Exception as e:
            logger.error(f"FATAL ERROR during background binary import: {e}", exc_info=True)
            raise e

    def import_binary_backgrounded(self, binary_path: str | Path):
        """
        Spawns a thread and imports a binary into the project.
        When the binary is analyzed it will be added to the project.

        Args:
            binary_path: The path of the binary to import.
        """
        if not Path(binary_path).exists():
            raise FileNotFoundError(f"The file {binary_path} cannot be found")

        if self.import_executor:
            future = self.import_executor.submit(self.import_binary, binary_path, analyze=True)
            future.add_done_callback(self._import_callback)
        else:
            self.import_binary(binary_path, analyze=True)

    def get_program_info(self, binary_name: str) -> "ProgramInfo":
        """Get program info or raise ValueError if not found."""
        program_info = self.programs.get(binary_name)
        if not program_info:
            # Exact program name not in the list
            available_progs = list(self.programs.keys())

            # If the LLM gave us just the binary name, use that
            available_prog_names = {
                Path(prog).name: prog_info for prog, prog_info in self.programs.items()
            }
            program_info = available_prog_names.get(binary_name)

            if not program_info:
                raise ValueError(
                    f"Binary {binary_name} not found. Available binaries: {available_progs}"
                )
        if not program_info.analysis_complete:
            raise RuntimeError(
                f"Analysis incomplete for binary '{binary_name}'. "
                f"Ghidra analysis: {program_info.ghidra_analysis_complete}. "
                "Wait and try tool call again."
            )
        return program_info

    def _init_program_info(self, program):
        from ghidra.program.flatapi import FlatProgramAPI

        assert program is not None

        metadata = self.get_metadata(program)

        program_info = ProgramInfo(
            name=program.name,
            program=program,
            flat_api=FlatProgramAPI(program),
            decompiler=self.setup_decompiler(program),
            metadata=metadata,
            ghidra_analysis_complete=False,
            file_path=metadata["Executable Location"],
            load_time=time.time(),
        )

        return program_info

    @staticmethod
    def _gen_unique_bin_name(path: Path):
        """
        Generate unique program name from binary for Ghidra Project
        """

        path = Path(path)

        def _sha1_file(path: Path) -> str:
            sha1 = hashlib.sha1()

            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha1.update(chunk)

            return sha1.hexdigest()

        return "-".join((path.name, _sha1_file(path.absolute())[:6]))

    # Callback function that runs when the future is done to catch any exceptions
    def _analysis_done_callback(self, future: concurrent.futures.Future):
        try:
            future.result()
            logger.info("Asynchronous analysis finished successfully.")
        except Exception as e:
            logger.error(f"Asynchronous analysis failed with exception: {e}")
            raise e

    def analyze_project(
        self,
        require_symbols: bool = True,
        force_analysis: bool = False,
        verbose_analysis: bool = False,
    ) -> concurrent.futures.Future | None:
        if self.executor:
            future = self.executor.submit(
                self._analyze_project,
                require_symbols,
                force_analysis,
                verbose_analysis,
            )

            future.add_done_callback(self._analysis_done_callback)

            if self.wait_for_analysis:
                logger.info("Waiting for analysis to complete...")
                try:
                    future.result()
                    logger.info("Analysis complete.")
                except Exception as e:
                    logger.error(f"Analysis completed with an exception: {e}")
                return None
            return future
        else:
            # No executor: just run synchronously
            self._analyze_project(require_symbols, force_analysis, verbose_analysis)
            return None

    def _analyze_project(
        self,
        require_symbols: bool = True,
        force_analysis: bool = False,
        verbose_analysis: bool = False,
    ) -> None:
        """
        Analyzes all files found within the Ghidra project
        """
        domain_files = self.list_binary_domain_files()

        logger.info(f"Starting analysis for {len(domain_files)} binaries")

        prog_count = len(domain_files)
        completed_count = 0

        if self.executor:
            futures = {
                self.executor.submit(
                    self.analyze_program,
                    domainFile,
                    require_symbols,
                    force_analysis,
                    verbose_analysis,
                )
                for domainFile in domain_files
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                logger.info(f"Analysis complete for {result.getName()}")
                completed_count += 1
                logger.info(f"Completed {completed_count}/{prog_count} programs")
        else:
            for domain_file in domain_files:
                self.analyze_program(domain_file, require_symbols, force_analysis, verbose_analysis)
                completed_count += 1
                logger.info(f"Completed {completed_count}/{prog_count} programs")

        logger.info("All programs analyzed.")

    def analyze_program(  # noqa C901
        self,
        df_or_prog: Union["DomainFile", "Program"],
        require_symbols: bool = True,
        force_analysis: bool = False,
        verbose_analysis: bool = False,
    ):
        from ghidra.app.script import GhidraScriptUtil
        from ghidra.framework.model import DomainFile
        from ghidra.program.flatapi import FlatProgramAPI
        from ghidra.program.model.listing import Program
        from ghidra.program.util import GhidraProgramUtilities
        from ghidra.util.task import ConsoleTaskMonitor

        df = df_or_prog
        if not isinstance(df_or_prog, DomainFile):
            df = df_or_prog.getDomainFile()

        if self.programs.get(df.pathname):
            # program already opened and initialized
            program = self.programs[df.pathname].program
        else:
            # open program from Ghidra Project
            program = self.project.openProgram(df.getParent().pathname, df_or_prog.getName(), False)
            self.programs[df.pathname] = self._init_program_info(program)

        assert isinstance(program, Program)

        logger.info(f"Analyzing: {program}")

        for gdt in self.gdts:
            logger.info(f"Loading GDT: {gdt}")
            if not Path(gdt).exists():
                raise FileNotFoundError(f"GDT Path not found {gdt}")
            self.apply_gdt(program, gdt)

        gdt_names = [name for name in program.getDataTypeManager().getSourceArchives()]
        if len(gdt_names) > 0:
            logger.debug(f"Using file gdts: {gdt_names}")

        if verbose_analysis or self.verbose_analysis:
            monitor = ConsoleTaskMonitor()
            flat_api = FlatProgramAPI(program, monitor)
        else:
            flat_api = FlatProgramAPI(program)

        if (
            GhidraProgramUtilities.shouldAskToAnalyze(program)
            or force_analysis
            or self.force_analysis
        ):
            GhidraScriptUtil.acquireBundleHostReference()

            if program and program.getFunctionManager().getFunctionCount() > 1000:
                # Force Decomp Param ID is not set
                if (
                    self.program_options is not None
                    and self.program_options.get("program_options", {})
                    .get("Analyzers", {})
                    .get("Decompiler Parameter ID")
                    is None
                ):
                    self.set_analysis_option(program, "Decompiler Parameter ID", True)

            if self.program_options:
                analyzer_options = self.program_options.get("program_options", {}).get(
                    "Analyzers", {}
                )
                for k, v in analyzer_options.items():
                    logger.info(f"Setting prog option:{k} with value:{v}")
                    self.set_analysis_option(program, k, v)

            if self.no_symbols:
                logger.warning(
                    f"Disabling symbols for analysis! --no-symbols flag: {self.no_symbols}"
                )
                self.set_analysis_option(program, "PDB Universal", False)

            logger.info(f"Starting Ghidra analysis of {program}...")
            try:
                flat_api.analyzeAll(program)
                if hasattr(GhidraProgramUtilities, "setAnalyzedFlag"):
                    GhidraProgramUtilities.setAnalyzedFlag(program, True)
                elif hasattr(GhidraProgramUtilities, "markProgramAnalyzed"):
                    GhidraProgramUtilities.markProgramAnalyzed(program)
                else:
                    raise Exception("Missing set analyzed flag method!")
            finally:
                GhidraScriptUtil.releaseBundleHostReference()
                self.project.save(program)
        else:
            logger.info(f"Analysis already complete.. skipping {program}!")

        # Save program as gzfs
        if self.gzfs_path is not None:
            from java.io import File  # type: ignore

            pathname = df.pathname.replace("/", "_")
            gzf_file = self.gzfs_path / f"{pathname}.gzf"
            self.project.saveAsPackedFile(program, File(str(gzf_file.absolute())), True)

        logger.info(f"Analysis for {df_or_prog.getName()} complete")
        self.programs[df.pathname].ghidra_analysis_complete = True
        return df_or_prog

    def set_analysis_option(  # noqa: C901
        self,
        prog: "Program",
        option_name: str,
        value: Any,
    ) -> None:
        """
        Set boolean program analysis options
        Inspired by: Ghidra/Features/Base/src/main/java/ghidra/app/script/GhidraScript.java#L1272
        """
        from ghidra.program.model.listing import Program

        prog_options = prog.getOptions(Program.ANALYSIS_PROPERTIES)
        option_type = prog_options.getType(option_name)

        match str(option_type):
            case "INT_TYPE":
                logger.debug("Setting type: INT")
                prog_options.setInt(option_name, int(value))
            case "LONG_TYPE":
                logger.debug("Setting type: LONG")
                prog_options.setLong(option_name, int(value))
            case "STRING_TYPE":
                logger.debug("Setting type: STRING")
                prog_options.setString(option_name, value)
            case "DOUBLE_TYPE":
                logger.debug("Setting type: DOUBLE")
                prog_options.setDouble(option_name, float(value))
            case "FLOAT_TYPE":
                logger.debug("Setting type: FLOAT")
                prog_options.setFloat(option_name, float(value))
            case "BOOLEAN_TYPE":
                logger.debug("Setting type: BOOLEAN")
                if isinstance(value, str):
                    temp_bool = value.lower()
                    if temp_bool in {"true", "false"}:
                        prog_options.setBoolean(option_name, temp_bool == "true")
                elif isinstance(value, bool):
                    prog_options.setBoolean(option_name, value)
                else:
                    raise ValueError(f"Failed to setBoolean on {option_name} {option_type}")
            case "ENUM_TYPE":
                logger.debug("Setting type: ENUM")
                from java.lang import Enum  # type: ignore

                enum_for_option = prog_options.getEnum(option_name, None)
                if enum_for_option is None:
                    raise ValueError(
                        f"Attempted to set an Enum option {option_name} without an "
                        + "existing enum value alreday set."
                    )
                new_enum = None
                try:
                    new_enum = Enum.valueOf(enum_for_option.getClass(), value)
                except Exception:
                    for enum_value in enum_for_option.values():  # type: ignore
                        if value == enum_value.toString():
                            new_enum = enum_value
                            break
                if new_enum is None:
                    raise ValueError(
                        f"Attempted to set an Enum option {option_name} without an "
                        + "existing enum value alreday set."
                    )
                prog_options.setEnum(option_name, new_enum)
            case _:
                logger.warning(f"option {option_type} set not supported, ignoring")

    def configure_symbols(
        self,
        symbols_path: str | Path,
        symbol_urls: list[str] | None = None,
        allow_remote: bool = True,
    ):
        """
        Configures symbol servers and attempts to load PDBs for programs.
        """
        from ghidra.app.plugin.core.analysis import (
            PdbAnalyzer,  # type: ignore
            PdbUniversalAnalyzer,  # type: ignore
        )
        from ghidra.app.util.pdb import PdbProgramAttributes  # type: ignore

        logger.info("Configuring symbol search paths...")
        # This is a simplification. A real implementation would need to configure the symbol server
        # which is more involved. For now, we'll focus on enabling the analyzers.

        for program_name, program in self.programs.items():
            logger.info(f"Configuring symbols for {program_name}")
            try:
                if hasattr(PdbUniversalAnalyzer, "setAllowUntrustedOption"):  # Ghidra 11.2+
                    PdbUniversalAnalyzer.setAllowUntrustedOption(program, allow_remote)
                    PdbAnalyzer.setAllowUntrustedOption(program, allow_remote)
                else:  # Ghidra < 11.2
                    PdbUniversalAnalyzer.setAllowRemoteOption(program, allow_remote)
                    PdbAnalyzer.setAllowRemoteOption(program, allow_remote)

                # The following is a placeholder for actual symbol loading logic
                pdb_attr = PdbProgramAttributes(program)
                if not pdb_attr.pdbLoaded:
                    logger.warning(
                        f"PDB not loaded for {program_name}. Manual loading might be required."
                    )

            except Exception as e:
                logger.error(f"Failed to configure symbols for {program_name}: {e}")

    def apply_gdt(
        self,
        program: "Program",
        gdt_path: str | Path,
        verbose: bool = False,
    ):
        """
        Apply GDT to program
        """
        from ghidra.app.cmd.function import ApplyFunctionDataTypesCmd
        from ghidra.program.model.data import FileDataTypeManager
        from ghidra.program.model.symbol import SourceType
        from ghidra.util.task import ConsoleTaskMonitor
        from java.io import File  # type: ignore
        from java.util import List  # type: ignore

        gdt_path = Path(gdt_path)

        if verbose:
            monitor = ConsoleTaskMonitor()
        else:
            monitor = ConsoleTaskMonitor().DUMMY_MONITOR

        archive_gdt = File(str(gdt_path))
        archive_dtm = FileDataTypeManager.openFileArchive(archive_gdt, False)
        always_replace = True
        create_bookmarks_enabled = True
        cmd = ApplyFunctionDataTypesCmd(
            List.of(archive_dtm),
            None,  # type: ignore
            SourceType.USER_DEFINED,
            always_replace,
            create_bookmarks_enabled,
        )
        cmd.applyTo(program, monitor)

    def get_metadata(self, prog: "Program") -> dict:
        """
        Generate dict from program metadata
        """
        meta = prog.getMetadata()
        return dict(meta)

    def setup_decompiler(self, program: "Program") -> "DecompInterface":
        from ghidra.app.decompiler import DecompileOptions, DecompInterface

        prog_options = DecompileOptions()

        decomp = DecompInterface()

        # grab default options from program
        prog_options.grabFromProgram(program)

        # increase maxpayload size to 100MB (default 50MB)
        prog_options.setMaxPayloadMBytes(100)

        decomp.setOptions(prog_options)
        decomp.openProgram(program)

        return decomp
