import json
import logging
from pathlib import Path

from articulate_core.cli.console import create_console
from articulate_core.exceptions import GenError, UserCancelledError
from articulate_core.pipeline.codegen_engine import CodeGenerationEngine, UserCallbacks
from articulate_core.pipeline.models import GeneratedCode, StageContext, SubTaskResult
from articulate_core.pipeline.orchestrator import BaseStage

logger = logging.getLogger(__name__)
console = create_console()


def _default_arbitrate(sub_task: str, lib_result: SubTaskResult, prompt_result: SubTaskResult) -> str:
    """Default user arbitration - ask user to choose between library and prompt."""
    print(f"\n{'=' * 50}")
    print(f"Sub-task: {sub_task}")
    print(f"{'=' * 50}")
    print(f"  [A] Library call (confidence: {lib_result.confidence:.2f})")
    print(f"  [B] Prompt generation (confidence: {prompt_result.confidence:.2f})")
    print(f"  [C] Cancel this sub-task")
    choice = input("Choose path [a/b/c]: ").strip().lower()
    if choice == "a":
        return "library"
    elif choice == "b":
        return "prompt"
    elif choice == "c":
        return "cancel"
    return "prompt"


def _write_generated_code(ctx: StageContext):
    """Write generated code files to disk.

    The package_structure keys already include the ros_ws/src/ prefix,
    so we write relative to the project directory.
    """
    if not ctx.generated_code or not ctx.generated_code.package_structure:
        return

    written = 0
    for rel_path, content in sorted(ctx.generated_code.package_structure.items()):
        file_path = ctx.project_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        written += 1

    ros_ws = ctx.project_dir / "ros_ws"
    logger.info("[Stage 3] Wrote %d files to %s", written, ros_ws)


def _default_code_summary(code: GeneratedCode):
    """Display summary of generated code."""
    print(f"\n{'=' * 60}")
    print("GENERATED CODE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Package name: {code.ros2_package_name}")
    print(f"  Total files: {len(code.package_structure)}")
    print(f"\n  Files:")
    for path in sorted(code.package_structure.keys()):
        content_len = len(code.package_structure[path])
        print(f"    {path} ({content_len} chars)")
    if code.lint_results:
        if code.lint_results.get("errors"):
            print(f"\n  [red]Errors:[/red] {len(code.lint_results['errors'])}")
            for e in code.lint_results["errors"][:3]:
                print(f"    ! {e}")
        if code.lint_results.get("warnings"):
            print(f"\n  Warnings: {len(code.lint_results['warnings'])}")
    print(f"{'=' * 60}")


class CodeGenerationStage(BaseStage):
    stage_id: int = 3
    stage_name: str = "code_generation"

    async def execute(self, ctx: StageContext) -> StageContext:
        logger.info("[Stage 3] Starting code generation")

        if not ctx.technical_approach:
            logger.error("No technical approach found. Run 'articulate plan' first.")
            ctx.should_continue = False
            return ctx

        # Set up callbacks with CLI interaction
        callbacks = UserCallbacks(
            confirm=lambda msg: input(f"{msg} (y/n): ").strip().lower() == "y",
            arbitrate_route=_default_arbitrate,
            render_code_summary=_default_code_summary,
        )

        engine = CodeGenerationEngine(self.skill, self.llm, callbacks, self.config)

        # 1. Decompose approach into sub-tasks
        print("\n[Stage 3] Analyzing technical approach and decomposing into sub-tasks...")
        sub_tasks = await engine.decompose(ctx.technical_approach)
        print(f"  Identified {len(sub_tasks)} sub-tasks:")
        for task in sub_tasks:
            print(f"    - {task.name}: {task.description[:60]}")

        if not await self._confirm("Proceed with code generation?"):
            return await self.rollback(ctx)

        # 2. Route and generate each sub-task
        results = []
        for i, task in enumerate(sub_tasks):
            print(f"\n  [{i+1}/{len(sub_tasks)}] Generating: {task.name}...")
            result = await engine.generate_subtask(task, ctx.technical_approach)
            results.append(result)
            route_symbol = "L" if result.route_used == "library" else "P"
            status = "OK" if result.success else "FAIL"
            print(f"    -> [{route_symbol}] {status} ({len(result.files)} file(s), conf={result.confidence:.2f})")

        # 3. Assemble into package
        print("\n[Stage 3] Assembling package...")
        package = await engine.assemble(results, ctx.technical_approach)
        print(f"  Merged {len(package.package_structure)} files")

        # 4. Validate
        print("[Stage 3] Validating generated code...")
        package = await engine.validate(package)
        if package.lint_results:
            errors = package.lint_results.get("errors", [])
            warnings = package.lint_results.get("warnings", [])
            print(f"  Validation: {'PASS' if package.lint_results.get('valid', True) else 'FAIL'}")
            if errors:
                print(f"  Errors ({len(errors)}):")
                for e in errors[:3]:
                    print(f"    ! {e}")
            if warnings:
                print(f"  Warnings ({len(warnings)}):")
                for w in warnings[:3]:
                    print(f"    ~ {w}")

        ctx.generated_code = package

        # 5. Display summary
        _default_code_summary(package)

        # 6. Write files to disk
        _write_generated_code(ctx)

        # 7. User confirmation
        if not await self._confirm("Does the generated code look correct?"):
            return await self.rollback(ctx)

        logger.info("[Stage 3] Code generation complete (%d files)", len(package.package_structure))
        return ctx

    async def _confirm(self, msg: str) -> bool:
        response = input(f"\n{msg} (y/n): ").strip().lower()
        return response in ("y", "yes")

    async def rollback(self, ctx: StageContext) -> StageContext:
        ctx.generated_code = None
        ctx.should_continue = False
        return ctx
