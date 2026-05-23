"""Stage 5: Deployment package generation."""

import logging
from pathlib import Path
from typing import Optional

from articulate_core.pipeline.deployment_manager import DeploymentManager
from articulate_core.pipeline.models import StageContext
from articulate_core.pipeline.orchestrator import BaseStage
from articulate_core.skill.converters.factory import ConverterFactory

logger = logging.getLogger(__name__)


class DeploymentStage(BaseStage):
    stage_id: int = 5
    stage_name: str = "deployment"
    SUPPORTED_BRANDS = {"ur": "Universal Robots (.script)", "kuka": "KUKA KRL (.src/.dat)", "abb": "ABB RAPID (.mod)"}

    async def execute(self, ctx: StageContext) -> StageContext:
        logger.info("[Stage 5] Starting deployment package generation")

        if not ctx.generated_code:
            logger.error("No generated code found. Run 'articulate codegen' first.")
            ctx.should_continue = False
            return ctx

        print("\n" + "=" * 60)
        print("STAGE 5: DEPLOYMENT PACKAGE GENERATION")
        print("=" * 60)

        # 1. Select target brand
        brand = await self._select_brand(ctx)
        if brand is None:
            return await self.rollback(ctx)

        ctx.target_brand = brand
        print(f"\n  Target brand: {brand.upper()}")

        # 2. Create deployment output directory
        deploy_dir = ctx.project_dir / "deploy" / brand
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # 3. Generate deployment package
        print("[Stage 5] Generating deployment package...")
        manager = DeploymentManager(ctx.generated_code, ctx.technical_approach)
        pkg = manager.prepare(brand, deploy_dir)

        # 4. Display results
        self._display_package(pkg)

        # 5. Save to context
        ctx.deployment_package = pkg

        # 6. User confirmation
        if not await self._confirm("Does this deployment package look acceptable?"):
            return await self.rollback(ctx)

        logger.info(
            "[Stage 5] Deployment package generated for %s (%d files)",
            brand, len(pkg.files),
        )
        return ctx

    async def _select_brand(self, ctx: StageContext) -> Optional[str]:
        """Let user select target brand, using ctx.target_brand as default."""
        print("\n  Available target brands:")
        brands = list(self.SUPPORTED_BRANDS.keys())
        for i, (key, desc) in enumerate(self.SUPPORTED_BRANDS.items()):
            default_mark = " (default)" if key == ctx.target_brand else ""
            print(f"    [{i + 1}] {key.upper()} — {desc}{default_mark}")

        print(f"    [{len(brands) + 1}] Enter custom brand")

        response = input(f"\n  Select brand [1-{len(brands) + 1}] (default: {ctx.target_brand}): ").strip().lower()

        if not response:
            return ctx.target_brand

        # Treat confirmation input as accepting default
        if response in ("y", "yes"):
            return ctx.target_brand

        # Check by number
        try:
            idx = int(response) - 1
            if 0 <= idx < len(brands):
                return brands[idx]
            if idx == len(brands):
                custom = input("  Enter custom brand name: ").strip().lower()
                if custom:
                    # Try factory first
                    try:
                        ConverterFactory.get_converter(custom)
                    except ValueError:
                        print(f"  [yellow]Warning:[/yellow] No built-in converter for '{custom}', but proceeding.")
                    return custom
                return None
        except ValueError:
            pass

        # Check by name
        if response in brands:
            return response

        print(f"  Unknown brand '{response}', using default '{ctx.target_brand}'.")
        return ctx.target_brand

    def _display_package(self, pkg):
        """Show deployment package summary."""
        from articulate_core.cli.console import create_console
        console = create_console()

        print(f"\n  Deployment Package: [bold]{pkg.target_brand.upper()}[/bold]")
        print(f"  Output directory: {pkg.output_dir}")
        print(f"  Files generated: {len(pkg.files)}")
        for name, path in sorted(pkg.files.items()):
            size = path.stat().st_size if path.exists() else 0
            console.print(f"    [green]OK[/green] {name} ({size} bytes)")
        print(f"  Deployment guide: {pkg.guide_path.name}")
        print(f"  Safety checklist: {pkg.checklist_path.name}")

    async def _confirm(self, msg: str) -> bool:
        response = input(f"\n{msg} (y/n): ").strip().lower()
        return response in ("y", "yes")

    async def rollback(self, ctx: StageContext) -> StageContext:
        ctx.deployment_package = None
        return ctx
