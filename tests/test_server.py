import json
import sys

import pytest
from mcp import Client

from churn_mcp import model, server
from churn_mcp.llm import MockLLM

EXPECTED_TOOLS = {
    "ask_data",
    "run_sql",
    "describe_dataset",
    "segment_churn",
    "at_risk_customers",
    "model_card",
}


def routed(question="What is the overall churn rate?"):
    return {"intent": "data_query", "normalized_question": question}


def drafted(sql):
    return {
        "sql": sql,
        "interpretation": "Quarterly churn rate",
        "assumptions": [],
        "confidence": "high",
    }


def narrated(answer):
    return {"answer": answer, "caveats": []}


@pytest.fixture
def mcp(con, layer, config):
    llm = MockLLM(
        [
            routed(),
            drafted("SELECT AVG(churn) AS churn_rate, COUNT(*) AS n FROM customers"),
            narrated("The quarterly churn rate was 26.5%."),
        ]
    )
    return server.build_server(con, layer, config, llm)


async def test_lists_exactly_the_six_tools(mcp):
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == EXPECTED_TOOLS


async def test_lists_the_schema_and_model_card_resources(mcp):
    async with Client(mcp) as client:
        resources = await client.list_resources()
        assert {str(r.uri) for r in resources.resources} == {
            "churn://schema",
            "churn://model-card",
        }


async def test_ask_data_returns_the_envelope_shape(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("ask_data", {"question": "What is the overall churn rate?"})
        body = result.structured_content
        assert body["result"]["grounding"] == "passed"
        assert "26.5" in body["result"]["answer"]
        assert body["meta"]["input_tokens"] > 0


async def test_run_sql_executes_a_valid_query(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("run_sql", {"sql": "SELECT COUNT(*) AS n FROM customers"})
        body = result.structured_content
        assert body["result"]["table"]["rows"] == [[7043]]
        assert body["caveats"] == []


async def test_run_sql_blocks_a_mutation(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("run_sql", {"sql": "DROP TABLE customers"})
        body = result.structured_content
        assert body["result"] is None
        assert "DROP" in body["meta"]["error"]


async def test_describe_dataset_returns_requested_columns(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("describe_dataset", {"columns": ["contract"]})
        body = result.structured_content
        assert set(body["result"]["columns"]) == {"contract"}
        assert "Month-to-Month" in body["result"]["columns"]["contract"]["values"]


async def test_describe_dataset_rejects_an_unknown_column(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("describe_dataset", {"columns": ["not_a_column"]})
        body = result.structured_content
        assert body["result"] is None
        assert "not_a_column" in body["meta"]["error"]


async def test_segment_churn_returns_ranked_segments(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("segment_churn", {"group_by": ["contract"]})
        body = result.structured_content
        assert len(body["result"]) == 3
        rates = [s["churn_rate"] for s in body["result"]]
        assert rates == sorted(rates, reverse=True)


async def test_segment_churn_rejects_an_unknown_filter_value(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "segment_churn",
            {"group_by": ["contract"], "filters": {"internet_type": "fiber"}},
        )
        body = result.structured_content
        assert body["result"] is None


async def test_at_risk_customers_returns_ranked_active_customers(mcp):
    async with Client(mcp) as client:
        result = await client.call_tool("at_risk_customers", {"top_n": 3})
        body = result.structured_content
        assert len(body["result"]) == 3
        assert body["caveats"]
        losses = [r["expected_loss"] for r in body["result"]]
        assert losses == sorted(losses, reverse=True)


async def test_schema_resource_lists_known_columns(mcp):
    async with Client(mcp) as client:
        read = await client.read_resource("churn://schema")
        parsed = json.loads(read.contents[0].text)
        assert "contract" in parsed["columns"]


async def test_model_card_resource_reports_real_auc_after_prepare(
    con, layer, sample_card, artifacts_config
):
    model.write_model_card(sample_card, artifacts_config)
    async with Client(server.build_server(con, layer, artifacts_config, None)) as client:
        read = await client.read_resource("churn://model-card")
        parsed = json.loads(read.contents[0].text)
        assert parsed["available"] is True
        assert parsed["auc"] == sample_card.auc


async def test_model_card_resource_falls_back_when_absent(con, layer, config, tmp_path):
    missing = config.model_copy(
        update={"data": config.data.model_copy(update={"artifacts_dir": tmp_path / "empty"})}
    )
    mcp_without_card = server.build_server(con, layer, missing, None)
    async with Client(mcp_without_card) as client:
        read = await client.read_resource("churn://model-card")
        parsed = json.loads(read.contents[0].text)
        assert parsed["available"] is False
        assert "churn-mcp prepare" in parsed["message"]


async def test_calling_every_tool_writes_nothing_to_stdout(mcp, capfd):
    async with Client(mcp) as client:
        await client.call_tool("run_sql", {"sql": "SELECT 1"})
        await client.call_tool("describe_dataset", {})
        await client.call_tool("segment_churn", {"group_by": ["contract"]})
        await client.call_tool("at_risk_customers", {"top_n": 1})
        await client.call_tool("ask_data", {"question": "What is the overall churn rate?"})
    captured = capfd.readouterr()
    assert captured.out == ""


def test_stdout_check_actually_detects_pollution(capfd):
    print("this would fail the real test", file=sys.stdout)
    captured = capfd.readouterr()
    assert captured.out != ""


def _scoped(config, artifacts_dir):
    return config.model_copy(
        update={"data": config.data.model_copy(update={"artifacts_dir": artifacts_dir})}
    )


async def test_model_card_tool_returns_the_auc(con, layer, sample_card, artifacts_config):
    model.write_model_card(sample_card, artifacts_config)
    async with Client(server.build_server(con, layer, artifacts_config, None)) as client:
        body = (await client.call_tool("model_card", {})).structured_content
        assert body["result"]["auc"] == sample_card.auc
        assert body["caveats"] == list(sample_card.caveats())


async def test_model_card_tool_falls_back_when_absent(con, layer, config, tmp_path):
    mcp_without_card = server.build_server(con, layer, _scoped(config, tmp_path / "none"), None)
    async with Client(mcp_without_card) as client:
        body = (await client.call_tool("model_card", {})).structured_content
        assert body["result"] is None
        assert "churn-mcp prepare" in body["meta"]["error"]


async def test_model_questions_point_to_the_model_card_tool(mcp):
    async with Client(mcp) as client:
        tools = {t.name: t.description for t in (await client.list_tools()).tools}
        assert "model_card" in tools["at_risk_customers"]
        assert "@" not in tools["ask_data"]


async def test_every_tool_and_resource_has_a_description(mcp):
    async with Client(mcp) as client:
        for tool in (await client.list_tools()).tools:
            assert tool.description, tool.name
        for resource in (await client.list_resources()).resources:
            assert resource.description, resource.uri
