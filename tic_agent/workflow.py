from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import json
import re

from agno.workflow.loop import Loop
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.steps import Steps
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

from .agents import AgentFactory, PipelineMode
from .config import AppConfig
from .knowledge import build_excel_knowledge


def create_workflow(config: AppConfig) -> Workflow:
    knowledge = build_excel_knowledge(config)
    factory = AgentFactory(config=config, knowledge=knowledge)

    analyze_executor = _analysis_executor(config)

    general_steps = Steps(
        name="general_pipeline",
        steps=_build_route_steps(factory, config, mode="general"),
    )
    industry_steps = Steps(
        name="industry_pipeline",
        steps=_build_route_steps(factory, config, mode="industry"),
    )

    router = Router(
        name="route_workflow",
        selector=_build_router_selector(general_steps, industry_steps),
        choices=[general_steps, industry_steps],
    )

    workflow = Workflow(
        name="TIC Research Workflow",
        description="Agno workflow for TIC 客户/行业调研",
        steps=[
            Step(name="generate_queries", agent=factory.query_agent()),
            Step(name="analyze_scope", executor=analyze_executor),
            router,
            Step(name="final_answer", agent=factory.final_answer_agent()),
        ],
        session_state={},
        add_workflow_history_to_steps=False,
    )

    return workflow


# --- Helpers -----------------------------------------------------------------


def _build_route_steps(factory: AgentFactory, config: AppConfig, mode: PipelineMode) -> List:
    loop = Loop(
        name=f"web_research_loop_{mode}",
        steps=[Step(name=f"web_research_{mode}", agent=factory.web_research_agent(mode))],
        max_iterations=max(1, config.workflow.max_research_loops),
        end_condition=_loop_end_condition(config.workflow.min_research_chars),
    )

    knowledge_step = Step(name=f"excel_knowledge_{mode}", agent=factory.knowledge_agent(mode))
    summary_step = Step(name=f"route_summary_{mode}", agent=factory.route_summary_agent(mode))

    return [loop, knowledge_step, summary_step]


def _loop_end_condition(min_chars: int) -> Callable[[List[StepOutput]], bool]:
    def _checker(outputs: List[StepOutput]) -> bool:
        if not outputs:
            return False
        latest = outputs[-1].content if outputs else None
        if not latest:
            return False
        return len(str(latest)) >= min_chars

    return _checker


def _build_router_selector(general_steps: Steps, industry_steps: Steps) -> Callable:
    def _selector(step_input: StepInput, session_state: Optional[Dict] = None):  # type: ignore[override]
        state = session_state or {}
        if state.get("industry_mode"):
            return [industry_steps]
        return [general_steps]

    return _selector


# --- Analysis Step -----------------------------------------------------------


@dataclass
class QueryPlan:
    queries: List[str]
    industry_mode: bool
    knowledge_focus: List[str]
    reasoning: str

    def serialize(self, question: str) -> str:
        plan = {
            "original_question": question,
            "industry_mode": self.industry_mode,
            "queries": self.queries,
            "knowledge_focus": self.knowledge_focus,
            "reasoning": self.reasoning,
        }
        return json.dumps(plan, ensure_ascii=False, indent=2)


def _analysis_executor(config: AppConfig) -> Callable[[StepInput, Dict, Optional[object]], StepOutput]:
    def _executor(step_input: StepInput, session_state: Optional[Dict] = None, *_):
        session = session_state if session_state is not None else {}
        question = step_input.get_input_as_string() or ""
        raw_plan = step_input.get_last_step_content() or step_input.get_input_as_string() or ""
        plan = _parse_query_plan(raw_plan, config.workflow.initial_queries)

        session["user_question"] = question
        session["query_plan"] = plan.serialize(question)
        session["queries"] = plan.queries
        session["knowledge_focus"] = plan.knowledge_focus
        session["industry_mode"] = plan.industry_mode

        summary_lines = [
            f"原始问题: {question}",
            f"行业模式: {'是' if plan.industry_mode else '否'}",
            "搜索 Query:",
        ]
        summary_lines.extend([f"- {q}" for q in plan.queries])
        if plan.knowledge_focus:
            summary_lines.append(f"Excel 关键词: {', '.join(plan.knowledge_focus)}")
        summary_lines.append(f"推理: {plan.reasoning}")

        return StepOutput(content="\n".join(summary_lines))

    return _executor


def _parse_query_plan(raw: str, limit: int) -> QueryPlan:
    extracted = _extract_json(raw)
    if extracted:
        try:
            data = json.loads(extracted)
            return QueryPlan(
                queries=_normalize_list(data.get("queries"), limit),
                industry_mode=bool(data.get("industry_mode", False)),
                knowledge_focus=_normalize_list(data.get("knowledge_focus"), limit),
                reasoning=str(data.get("reasoning") or ""),
            )
        except Exception:
            pass

    lines = [line.strip("- •") for line in raw.splitlines() if line.strip()]
    queries = [line for line in lines if len(line) > 2][:limit]
    return QueryPlan(
        queries=queries or ["TIC 行业最新趋势"],
        industry_mode=False,
        knowledge_focus=queries[:2],
        reasoning="Fallback parsing",
    )


def _extract_json(text: str) -> Optional[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```(json)?", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0) if match else None


def _normalize_list(value: Optional[Iterable], limit: int) -> List[str]:
    if not value:
        return []
    normalized: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized
