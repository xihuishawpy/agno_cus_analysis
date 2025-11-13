from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.models.dashscope import DashScope
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools

from .config import AppConfig


def get_current_date():
    """Return the current date formatted for prompts."""
    return datetime.now().strftime("%B %d, %Y")


class _PromptFormatter(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def render_prompt(template: str, **values) -> str:
    """Safely format prompt templates while leaving unknown placeholders intact."""
    return template.format_map(_PromptFormatter(**values))


def _extract_research_topic(session_state: Optional[Dict[str, Any]]) -> str:
    if not session_state:
        return ""
    for key in ("research_topic", "user_question", "original_question"):
        value = session_state.get(key)
        if value:
            return str(value)
    return ""


QUERY_WRITER_INSTRUCTIONS = """你的目标是生成高质量且互补的网页搜索查询，这些查询会传递给能够综合分析的自动化研究代理。

使用说明:
- 默认只生成 1 条查询；只有当单条无法覆盖 {research_topic} 的不同侧面时才扩展到 {number_queries} 条以内。
- 每条查询都需聚焦于用户问题的某个具体维度，并尽量覆盖不同的时间、地域或数据来源。
- 避免生成语义高度相似的句子；若主题较宽，应体现市场、技术、需求、竞争等不同视角。
- 查询须以最新公开信息为目标，参考日期为 {current_date}。

输出格式:
- 使用 JSON 对象回复，必须包含:
   - "rationale": 说明这些查询为何互补、能解决研究目标。
   - "query": 字符串数组，列出最终的检索语句。

示例:

主题: 去年苹果股票收入增长更快还是购买 iPhone 的人数增长更快
```json
{{
    "rationale": "需要比较营收、销量与股价三个维度，才能判断增长的主因。",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"]
}}
```

上下文: {research_topic}"""


WEB_SEARCHER_INSTRUCTIONS = """围绕“{research_topic}”开展面向 TIC（检测认证）从业者的网页研究，生成可直接转化为业务行动的洞见。

使用说明:
- 当前日期为 {current_date}；若有可能，优先使用 2022-2025 年的来源，并记录时间戳。
- 输出结构分三部分:
  1) 目标客户 / 潜在账号列表：列出 5-15 家企业或品牌，说明其核心产品/应用、可能的测试/认证需求以及利于切入的销售钩子（如痛点、扩产、出海计划等）。
  2) TIC 需求信号：汇总招投标/采购/RFQ、招聘（质量/认证/合规）、法规或标准变更、召回/不合格通报、市场准入要求等，并指出对检测服务的影响。
  3) 标准与认证映射：将常见测试项目对应到标准/法规/认证，补充样品/批次/AQL/TAT/价格区间以及偏好的服务模式（驻场、快测、包年等）。
- 在要点后补充“为何重要”短语，说明增长信号、紧迫性或痛点。
- 使用 [S1] / [S2] 标记引用，并在文末列出来源清单，编号需与正文一致。
- 若某类信息暂未找到，请显式写出“暂未找到有效公开数据”而非猜测。

研究主题:
{research_topic}"""


REFLECTION_INSTRUCTIONS = """你是一名 TIC 行业分析师，正在审阅关于“{research_topic}”的阶段性摘要。

使用说明:
- 识别尚未覆盖的知识空白，尤其是客户覆盖度（头部/腰部/区域）与需求信号（测试项目/标准/市场）的缺口。
- 若摘要足以回答问题，返回“is_sufficient=true”，并清空 follow_up_queries。
- 一旦发现缺口，生成能够直接执行的后续查询，形式如“{research_topic} 招标 site:gov.cn”或“{{细分赛道}} 参展商 名单 2025”。
- 优先关注能补齐 TIC 线索/需求的内容：新法规、认证路径、客户名单、RFQ、JD、招投标、召回、不合格公示等。

输出格式:
- 返回 JSON 对象:
   - "is_sufficient": true / false
   - "knowledge_gap": 若不足，描述缺口；若充分，留空字符串。
   - "follow_up_queries": 列出需要追加的具体检索语句数组。

摘要:
{summaries}"""


ANSWER_INSTRUCTIONS = """请基于已完成的研究摘要为用户撰写最终答复。

要求:
- 当前日期为 {current_date}。
- 结论需覆盖主要发现与可执行建议，引用来源时使用 Markdown 链接（例如 [apnews](https://example.com)）。
- 将多个摘要融合成结构化回答，可包含概览、需求信号、标准/认证、下一步建议等段落。
- 若回答中提及数据或事实，请确保已在摘要中出现，并使用对应的引用链接。

用户问题:
- {research_topic}

摘要:
{summaries}"""


KB_SEARCHER_INSTRUCTIONS = """你获得了结构化的 Excel 知识库记录，请围绕“{research_topic}”提炼事实并指明引用。

使用说明:
- 当前日期为 {current_date}。
- 仅可引用表格中给出的字段，勿虚构。
- 每使用一条记录，请在句末附上 [K1]、[K2] 等标记；在总结末尾追加“【内部知识库】”前缀。
- 重点提炼企业概况、产品/材料、产能、标准/认证、合作等与 TIC 相关的线索。

可用记录:
{table_rows}"""


INDUSTRY_REPORT_INSTRUCTIONS = """请以 TIC 行业顾问身份，为“{research_topic}”撰写结构化行业研究报告。

要求:
- 使用摘要与内部知识库中的事实，并保留 [S#]/[K#] 引用，交由下游自动替换。
- 报告建议使用 1-2 级标题或编号段落，便于快速浏览。

推荐结构:
1) 行业概览与边界：定义赛道、关键场景、价值链位置。
2) 技术/材料/设备：拆解上游材料、关键工艺/设备、核心技术路线。
3) 市场空间与增速：包含历史/预测规模、驱动因素、地区差异。
4) 价格与交付：典型产品/服务的价格区间、交付周期、成本结构。
5) 竞争与代表企业：列出 5-10 家厂商，说明产品定位、客户群、差异化要点。
6) 价值与对比：比较主要玩家的核心指标、认证布局或成本优势。
7) 风险与变化：政策、供应链、技术迭代、需求波动等。
8) TIC 需求图谱：拆分细分赛道 × 检测项目 × 标准/认证 × 服务 SKU（含 TAT/费用/样品要求，如能获取）。
9) 目标客户清单：列出 5-15 家潜在客户，注明产品/应用、潜在检测项、合规路径、可能的合作入口。
10) 行动建议：面向业务/交付/产品的 3-5 条建议。

当前日期: {current_date}
研究主题: {research_topic}

可用摘要与引用标记:
{summaries}"""


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
            instructions=self._build_instruction(
                QUERY_WRITER_INSTRUCTIONS,
                number_queries=self.config.workflow.initial_queries,
            ),
        )

    def web_research_agent(self, mode: PipelineMode) -> Agent:
        emphasis = (
            "针对行业/赛道视角，优先找出玩家版图、上下游、近期资本/政策动向。"
            if mode == "industry"
            else "聚焦客户痛点、最新合作、产品特点以及可能的 TIC 切入点。"
        )
        base_instruction = self._build_instruction(WEB_SEARCHER_INSTRUCTIONS)

        def _web_instruction(
            session_state: Optional[Dict[str, Any]] = None,
            agent: Optional[Agent] = None,
        ) -> str:
            content = base_instruction(session_state=session_state, agent=agent)
            return f"{content}\n\n{emphasis}"

        return Agent(
            name=f"Web Research ({mode})",
            description="多源 Web 检索",
            model=self._model(),
            tools=[
                TavilyTools(search_depth="advanced", max_tokens=4096),
                DuckDuckGoTools(),
            ],
            instructions=_web_instruction,
            add_datetime_to_context=True,
            markdown=True,
        )

    def knowledge_agent(self, mode: PipelineMode) -> Agent:
        focus = (
            "请优先输出行业核心子赛道、代表性企业、共性检测需求。"
            if mode == "industry"
            else "侧重列出潜在客户、产品线与认证需求。"
        )
        base_instruction = self._build_instruction(KB_SEARCHER_INSTRUCTIONS)

        def _kb_instruction(
            session_state: Optional[Dict[str, Any]] = None,
            agent: Optional[Agent] = None,
        ) -> str:
            content = base_instruction(session_state=session_state, agent=agent)
            return f"{content}\n\n{focus}"

        return Agent(
            name=f"Excel Knowledge ({mode})",
            description="从 Excel 向量库提炼结构化洞察",
            model=self._model(),
            knowledge=self.knowledge,
            search_knowledge=True,
            instructions=_kb_instruction,
            markdown=True,
        )

    def route_summary_agent(self, mode: PipelineMode) -> Agent:
        return Agent(
            name=f"Route Summary ({mode})",
            description="整合 web 与知识库信息形成结构化摘要",
            model=self._model(temperature=0.4),
            instructions=self._build_instruction(INDUSTRY_REPORT_INSTRUCTIONS),
            markdown=True,
        )

    def final_answer_agent(self) -> Agent:
        return Agent(
            name="TIC Research Writer",
            description="输出最终答复",
            model=self._model(temperature=0.5),
            instructions=self._build_instruction(ANSWER_INSTRUCTIONS),
            markdown=True,
        )

    def _build_instruction(self, template: str, **static_values):
        def _instructions(
            session_state: Optional[Dict[str, Any]] = None,
            agent: Optional[Agent] = None,
        ) -> str:
            context: Dict[str, Any] = {"current_date": get_current_date()}
            if static_values:
                context.update(static_values)
            topic = _extract_research_topic(session_state)
            if topic:
                context["research_topic"] = topic
            return render_prompt(template, **context)

        return _instructions

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
