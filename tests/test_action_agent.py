from agents.action_agent import action_node


def test_action_agent():
    result = action_node({
        "question": "What is the largest planet in our solar system?",
        "summary_output": {
            "output": (
                "Jupiter is the largest planet in our solar system. "
                "It is a gas giant and is larger than all the other planets combined."
            ),
            "confidence": 1.0
        },
        "trace": []
    })

    output = result["action_output"]

    assert isinstance(output["output"], str)
    assert len(output["output"].strip()) > 0
    assert 0.0 <= output["confidence"] <= 1.0