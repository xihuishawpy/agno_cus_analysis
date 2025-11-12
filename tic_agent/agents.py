from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.models.dashscope import DashScope
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools

from .config import AppConfig


PipelineMode = Literal["general", "industry"]


@dataclass
class AgentFactory:
    config: AppConfig
    knowledge: Knowledge

    def query_agent(self) -> Agent:
        return Agent(
            name="Query Planner",
            description="将 TIC 业务问题拆解为可执行的检索计划",
            model=self._model(temperature=0.1),
            markdown=False,
            instructions=[
                "阅读用户问题，结合 TIC (检测认证) 行业语境判断信息粒度。",
                f"生成 2~{self.config.workflow.initial_queries} 条精准的中文或英文 Web 搜索 query。",
                "推断是否需要行业级别综述 (industry_mode)，如果问题涉及赛道/行业/领域概念则置为 true。",
                "给出 knowledge_focus 关键词，便于后续在 Excel 向量库中过滤。",
                "以 JSON 形式输出：{\"queries\":[], \"industry_mode\":bool, \"knowledge_focus\":[], \"reasoning\":str}。",
                "不要添加解释或代码块标记。",
            ],
        )

    def web_research_agent(self, mode: PipelineMode) -> Agent:
        emphasis = (
            "针对行业/赛道视角，优先找出玩家版图、上下游、近期资本/政策动向。"
            if mode == "industry"
            else "聚焦客户痛点、最新合作、产品特点以及可能的 TIC 切入点。"
        )
        return Agent(
            name=f"Web Research ({mode})",
            description="多源 Web 检索",
            model=self._model(),
            tools=[
                TavilyTools(search_depth="advanced", max_tokens=4096),
                DuckDuckGoTools(),
            ],
            instructions=[
                "根据上一阶段给出的 queries 逐条检索，必要时自定更多 query。",
                "每条新信息必须附来源链接，引用格式如 [1](url)。",
                emphasis,
                "输出结构：发现摘要、3~5 个可信来源、对 TIC 机会的初步判断。",
            ],
            add_datetime_to_context=True,
            markdown=True,
        )

    def knowledge_agent(self, mode: PipelineMode) -> Agent:
        focus = (
            "请优先输出行业核心子赛道、代表性企业、共性检测需求。"
            if mode == "industry"
            else "侧重列出潜在客户、产品线与认证需求。"
        )
        return Agent(
            name=f"Excel Knowledge ({mode})",
            description="从 Excel 向量库提炼结构化洞察",
            model=self._model(),
            knowledge=self.knowledge,
            search_knowledge=True,
            instructions=[
                "利用上一阶段给出的知识关键词进行语义检索，最多引用 5 条记录。",
                "请在结果中包含 `source`、`ticker`、`company`、行业/概念等字段。",
                focus,
                "输出 Markdown 表格 + 重点洞察，并在表格后统一列出引用键 (source/ticker)。",
            ],
            markdown=True,
        )

    def route_summary_agent(self, mode: PipelineMode) -> Agent:
        title = "行业调研总结" if mode == "industry" else "客户调研总结"
        return Agent(
            name=f"Route Summary ({mode})",
            description="整合 web 与知识库信息形成结构化摘要",
            model=self._model(temperature=0.4),
            instructions=[
                "综合前面步骤的输出，形成 3 个部分：1) 结论摘要 2) 重点发现列表 3) TIC 行动建议。",
                "引用时沿用 [编号](链接) 或 Excel metadata 标识 (source:ticker)。",
                "保持中文输出并限制在 400~500 中文字符。",
            ],
            markdown=True,
        )

    def final_answer_agent(self) -> Agent:
        return Agent(
            name="TIC Research Writer",
            description="输出最终答复",
            model=self._model(temperature=0.5),
            instructions=[
                "结合用户原始问题与上一阶段的总结，写出完整的调研报告。",
                "必须包含：问题复述、行业/客户洞察、潜在客户清单、TIC 机会与下一步行动。",
                "引用统一放在结尾，按 [编号] url 或 (source:ticker) 展示。",
            ],
            markdown=True,
        )

    def _model(self, temperature: float | None = None) -> DashScope:
        kwargs = {
            "id": self.config.model.model_id,
            "temperature": temperature if temperature is not None else self.config.model.temperature,
            "base_url": self.config.dashscope_base_url,
        }
        if self.config.model.enable_thinking:
            kwargs["enable_thinking"] = True
        if self.config.model.thinking_budget:
            kwargs["thinking_budget"] = self.config.model.thinking_budget
        return DashScope(**kwargs)
