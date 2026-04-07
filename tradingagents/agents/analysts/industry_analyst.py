"""
Industry Analyst Agent

Performs long-term sector/industry tracking for a target company. Unlike other
analysts that focus on short-term price action or recent events, the Industry
Analyst maintains a persistent thesis for each sector and evaluates how current
conditions compare to the accumulated historical view.

Workflow:
1. Load existing industry thesis from IndustryThesisMemory (if any)
2. Use tools to gather current sector data (classification, performance, news)
3. Generate a comprehensive industry report
4. The caller (TradingAgentsGraph) extracts structured fields from the report
   and updates IndustryThesisMemory so knowledge compounds over time.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.industry_data_tools import (
    get_sector_overview,
    get_sector_performance_vs_benchmark,
    get_industry_peer_metrics,
    get_industry_news,
)


def create_industry_analyst(llm, industry_memory=None):
    """
    Create an Industry Analyst node function.

    Args:
        llm: LangChain-compatible LLM instance
        industry_memory: Optional IndustryThesisMemory instance for loading
                         prior thesis context

    Returns:
        A node function compatible with the LanGraph StateGraph
    """

    def industry_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_sector_overview,
            get_sector_performance_vs_benchmark,
            get_industry_peer_metrics,
            get_industry_news,
        ]

        # Load prior thesis if memory is available
        prior_thesis_context = ""
        if industry_memory is not None:
            # We don't know the industry yet at this point, so we load after
            # the first tool call; pass a placeholder to the prompt
            prior_thesis_context = (
                "Note: You will retrieve the sector/industry classification first "
                "via get_sector_overview. Once you know the industry name, your "
                "prior thesis context will be automatically injected into your analysis."
            )

        system_message = (
            "You are a senior industry analyst specializing in long-term sector tracking. "
            "Your job is to analyze the industry and sector that a company belongs to, "
            "tracking structural trends, catalysts, headwinds, and competitive dynamics "
            "that evolve over months and years — not just the latest news cycle.\n\n"
            "Your analysis should:\n"
            "1. Identify the sector and industry classification for the target company\n"
            "2. Assess how the company's stock has performed relative to its sector benchmark\n"
            "3. Evaluate industry-level valuation metrics and where the company stands vs peers\n"
            "4. Identify major sector news, catalysts, and regulatory/macro themes\n"
            "5. Articulate the long-term investment thesis for the sector including:\n"
            "   - Structural tailwinds (durable multi-year growth drivers)\n"
            "   - Structural headwinds (persistent risks that don't go away)\n"
            "   - Current cycle position (early/mid/late cycle dynamics)\n"
            "   - Key inflection points or upcoming catalysts to watch\n\n"
            "Use the available tools:\n"
            "- `get_sector_overview`: Get sector/industry classification and business description\n"
            "- `get_sector_performance_vs_benchmark`: Compare stock vs sector ETF\n"
            "- `get_industry_peer_metrics`: Get valuation metrics vs industry context\n"
            "- `get_industry_news`: Get recent sector and company news\n\n"
            "Format your final report with clear sections for: Sector Overview, "
            "Relative Performance, Valuation Context, Industry Catalysts & News, "
            "Structural Tailwinds, Structural Headwinds, and Investment Thesis Summary.\n"
            "Append a Markdown table at the end summarizing key industry metrics.\n\n"
            f"{prior_thesis_context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    " For your reference, the current date is {current_date}."
                    " The company we want to analyze is {ticker}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([t.name for t in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

            # Update industry memory with this session's observation
            if industry_memory is not None and report:
                _update_industry_memory(industry_memory, ticker, report, current_date)

        return {
            "messages": [result],
            "industry_report": report,
        }

    return industry_analyst_node


def _update_industry_memory(industry_memory, ticker: str, report: str, trade_date: str):
    """
    Extract key information from the analyst report and persist it in
    IndustryThesisMemory so future sessions can build on this analysis.

    This does a lightweight parse of the report to extract the industry name
    and a one-line observation. The full thesis update is handled via a
    dedicated reflection step.
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker.upper()).info
        industry = info.get("industry", "Unknown Industry")
        sector = info.get("sector", "Unknown Sector")
        industry_key = f"{sector} / {industry}"

        # Extract a brief observation from the beginning of the report
        lines = [line.strip() for line in report.split("\n") if line.strip()]
        # Use the first substantive line as the observation summary
        observation = next(
            (line for line in lines if len(line) > 30 and not line.startswith("#")),
            f"Industry analysis completed for {ticker} on {trade_date}",
        )
        # Truncate to keep it concise
        if len(observation) > 200:
            observation = observation[:197] + "..."

        industry_memory.update_thesis(
            industry_key=industry_key,
            observation=observation,
            trade_date=trade_date,
        )
    except Exception:
        # Non-critical — don't let memory update failures break the pipeline
        pass
