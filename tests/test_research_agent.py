from agents.research_agent import research_node


def test_research_agent():
    result = research_node({
        "question": "What is the largest planet in our solar system?",
        "trace": []
    })

    output = result["research_output"]

    assert isinstance(output["output"], str)
    assert len(output["output"].strip()) > 0
    assert 0.0 <= output["confidence"] <= 1.0