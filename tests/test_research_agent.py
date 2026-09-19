from agents.research_agent import research_node


def test_research_agent():
    result = research_node(
        {
            "question": "What is the largest planet in our solar system?",
            "context": {
                "title": ["Jupiter"],
                "sentences": [
                    [
                        "Jupiter is the largest planet in the Solar System."
                    ]
                ],
            },
            "trace": [],
        }
    )

    assert "research_output" in result
    assert "trace" in result
    assert len(result["trace"]) == 1
    assert result["trace"][0]["agent"] == "research"