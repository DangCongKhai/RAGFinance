from typing import List, List
from langchain_core.output_parsers import ListOutputParser
from langchain_core.prompts import PromptTemplate
import re

class CustomizedListParser(ListOutputParser):
    def parse(self, text) -> List[List[List[str]]]:
        pattern = r"\[\s*(?:\[\s*(?:\[[^\]]*?\]\s*,?\s*\n*)*\s*\]\s*,?\s*)*\s*\]"
        result = re.findall(pattern, text)
        if result:
            extracted_text = re.findall(pattern, text)[0]
            return eval(extracted_text)
        return []
    
PROMPT_TEMPLATE = PromptTemplate.from_template(
    """
Instruction:
Your task is to construct an RDF (Resource Description Framework) graph from the given passages and
named entity lists.
Respond with a list of lists of triples with each triple representing a relationship in the RDF graph.
Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each
passage.
- If there is only one named entity provided, add an empty list to the result.
- Clearly resolve pronouns to their specific names to maintain clarity.
- Do not include entity that does not exist in the provided named entities list to form triple. In the case you cannot find any relations for that entity, no need to find relation for it.

For each paragraph, convert it to a list of triples and put all of them in a list in sequential order.
If the named_entities list provided is empty, you can simply add empty list to that result
One-Shot Demonstration for the case I provide only 1 paragraph:
Paragraph:
```
Radio City
Radio City is India’s first private FM radio station and was started on 3 July 2001. It plays Hindi, English
and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music
portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related
features.
```
named_entities : ["Radio City", "India", "3 July 2001", "Hindi","English", "May 2008", "PlanetRadiocity.com"]
You return:
 [
 [["Radio City", "located in", "India"],
 ["Radio City", "is", "private FM radio station"],
 ["Radio City", "started on", "3 July 2001"],
 ["Radio City", "plays songs in", "Hindi"],
 ["Radio City", "plays songs in", "English"],
 ["Radio City", "forayed into", "New Media"],
 ["Radio City", "launched", "PlanetRadiocity.com"],
 ["PlanetRadiocity.com", "launched in", "May 2008"],
 ["PlanetRadiocity.com", "is", "music portal"],
 ["PlanetRadiocity.com", "offers", "news"],
 ["PlanetRadiocity.com", "offers", "videos"],
 ["PlanetRadiocity.com", "offers", "songs"]]
 ]
Here are passages and their corresponding entities that you need to extract relations:
{message}
"""
)

    



