from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel
from smol_agents.ollama_model import OllamaModel


def run_story_generation():
    model = OllamaModel(model_id="hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q5_K_S")
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        additional_authorized_imports=["math"],
        planning_interval=3,
        max_iterations=20,
    )

    result = agent.run(
        """How many apples go in a tiny bushel?"""
    )

    print("\nFinal Story:")
    print(result)


if __name__ == "__main__":
    run_story_generation()
