from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

# from langchain.agents.load_tools import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(animal_type:str, pet_color:str, pet_gender:str) -> str:
    '''Function to generate random pet names using llms and prompts.'''
    llm = OpenAI(temperature = 0.7)
    prompt_template_name = PromptTemplate(
                                            input_variables=["animal_type", "pet_color", "pet_gender"],
                                            template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color, and it is {pet_gender} as a gender. Suggest me five cool names for my pet. Return only the names (not any description)"
                                        )
    
    name_chain = prompt_template_name | llm | StrOutputParser()
    response = name_chain.invoke({"animal_type": animal_type, "pet_color":pet_color, "pet_gender":pet_gender}, config={"return_full_response": True})

    return response


def langchain_agent():
    '''Funcion which returns the result based on agents logics.'''
    llm = OpenAI(temperature = 0.5)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)

    agent = initialize_agent(tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, llm=llm)

    result = agent.invoke("What is the average age of a dog? Search it on wikipedia, and then multiply this average age by 3.")

    print(result)


if __name__ == "__main__":
    # print(generate_pet_name(animal_type="dog", pet_color="white", pet_gender="male"))
    print(langchain_agent())