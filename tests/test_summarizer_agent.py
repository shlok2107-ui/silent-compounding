from agents.summarizer_agent import summarize_node


def test_summarizer_agent():
    result = summarize_node({
        "research_output": {
            "output": (
                "Jupiter is the largest planet in our solar system. "
                "It is a gas giant and is larger than all the other planets combined."
            ),
            "confidence": 1.0
        },
        "trace": []
    })

    output = result["summary_output"]

    assert isinstance(output["output"], str)
    assert len(output["output"].strip()) > 0
    assert 0.0 <= output["confidence"] <= 1.0