
import asyncio
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.stores import InMemoryStore


load_dotenv(verbose=True, override=True)

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1000,
    temperature=0.3
)


checkpointer = InMemorySaver()
store = InMemoryStore()

config = {
    "configurable": {
        "thread_id": "thread-1"
    }
}


class ClassificationState(BaseModel):
    query: str = Field(
        description="User's question"
    )

    category: str | None = Field(
        default=None,
        description="Question category"
    )

    resolution: str | None = Field(
        default=None,
        description="Final answer"
    )


class ClassificationOutput(BaseModel):
    category: Literal[
        "biology",
        "coding",
        "generic",
        "history",
        "mathematics",
        "philosophical",
        "physics"
    ]


class SpecialistOutput(BaseModel):
    resolution: str = Field(
        description="Detailed answer to the user's question"
    )

classification_agent = create_agent(
    model=model,
    system_prompt="""
    You are a question classification agent.

    Classify the user's question into exactly one category:

    biology, coding, generic, history,
    mathematics, philosophical, physics.

    Return only the structured classification.
    """,
    response_format=ClassificationOutput
)


def create_specialist_agent(category: str):
    return create_agent(
        model=model,
        system_prompt=f"""
        You are a specialized {category} agent.

        Your responsibility is to answer questions related
        to {category}.

        Provide accurate, detailed and helpful explanations.
        If the question is outside your domain, explain
        the limitation rather than inventing information.

        Return your answer using the required structured output.
        """,
        response_format=SpecialistOutput
    )


specialist_agents = {
    "biology": create_specialist_agent("biology"),
    "coding": create_specialist_agent("coding"),
    "generic": create_specialist_agent("general knowledge"),
    "history": create_specialist_agent("history"),
    "mathematics": create_specialist_agent("mathematics"),
    "philosophical": create_specialist_agent("philosophy"),
    "physics": create_specialist_agent("physics")
}


async def classification_node(state: ClassificationState):

    print("Inside classification node")

    result = await classification_agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": state.query
            }
        ]
    })

    structured_result: ClassificationOutput = (
        result["structured_response"]
    )

    print("Detected category:", structured_result.category)

    return {
        "category": structured_result.category
    }


def create_specialist_node(category: str):

    agent = specialist_agents[category]

    async def specialist_node(state: ClassificationState):

        print(f"Inside {category} node")

        result = await agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": state.query
                }
            ]
        })

        structured_result: SpecialistOutput = (
            result["structured_response"]
        )

        return {
            "resolution": structured_result.resolution
        }

    return specialist_node


def conditional_route(state: ClassificationState):

    print("Inside conditional routing")

    if state.category not in specialist_agents:
        raise ValueError(
            f"Unknown category: {state.category}"
        )

    return state.category

builder = StateGraph(ClassificationState)

builder.add_node(
    "classification",
    classification_node
)

for category in specialist_agents:
    builder.add_node(
        category,
        create_specialist_node(category)
    )


builder.add_edge(START, "classification")

builder.add_conditional_edges(
    "classification",
    conditional_route,
    {
        category: category
        for category in specialist_agents
    }
)

for category in specialist_agents:
    builder.add_edge(category, END)



graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)


async def main():

    input_data = ClassificationState(
        query="Why spooky action at distance is not proven theory?"
    )

    response = await graph.ainvoke(
        input_data,
        config=config
    )

    print("\nFinal Response:")
    print(response["resolution"])


if __name__ == "__main__":
    asyncio.run(main())
