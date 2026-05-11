

prompts_dict = {
    "filtered_triples": """
                        Instruction:
                        
                        You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers
                        worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information
                        is presented to these stakeholders. The query requires careful analysis and possibly multi-hop reasoning to connect
                        different pieces of information.
                        You must select up to 4 relevant facts from the provided candidate list that have a strong connection to the query, aiding
                        in reasoning and providing an accurate answer.
                        The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, and if no facts are relevant, return
                        an empty list, {"fact": []}.
                        The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level
                        stakeholders. You must only use facts from the candidate list and not generate new facts. The future of critical decision-
                        making relies on your ability to accurately filter and present relevant information.
                        
                        Demonstration:
                        Question: Are Imperial River (Florida) and Amaradia (Dolj) both located in the same country?
                        Fact Before Filter: "{"fact": [["imperial river", "is located in", "florida"], ["imperial river", "is a river in", "united states"],
                        ["imperial river", "may refer to", "south america"], ["amaradia", "flows through", "ro ia de amaradia"], ["imperial river",
                        "may refer to", "united states"]]}",
                        Fact After Filter: "{"fact":[["imperial river","is located in","florida"],["imperial river","is a river in","united
                        states"],["amaradia","flows through","ro ia de amaradia"]]}”
                        Question: When is the director of film The Ancestor 's birthday?
                        Fact Before Filter: "{"fact": [["jean jacques annaud", "born on", "1 october 1943"], ["tsui hark", "born on", "15 february
                        1950"], ["pablo trapero", "born on", "4 october 1971"], ["the ancestor", "directed by", "guido brignone"], ["benh zeitlin",
                        "born on", "october 14 1982"]]}
                        Fact After Filter: "{"fact":[["the ancestor","directed by","guido brignone"]]}
                        
                        """,

                        

}


class PromptManager:
    def __init__(self):
        pass
    
    def get_prompt(key : str):
        return prompts_dict[key]
