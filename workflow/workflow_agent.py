from typing import TypedDict, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from logger import setup_logging, get_logger
from langgraph.graph import StateGraph, END
from google import genai
from dotenv import load_dotenv
import os
import re
import base64
import json


setup_logging()
# logger = get_logger(__name__)

load_dotenv()

class ResultNode(BaseModel):
    field: str = ""
    found: bool = False
    found_by: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    probable_value: str = ""
    confidences: List[int] = Field(default_factory=list)
    low_confidence: bool = False
    found_multiple: bool = False

    def to_json(self, **kwargs) -> str:
        '''Convert to JSON'''
        return self.model_dump_json(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ResultNode":
        '''Create from a JSON string'''
        return cls.model_validate_json(json_str)

class InferenceState(BaseModel):
    local_results: List[Tuple[str, Dict[str, Any]]] = Field(default_factory=list)
    inferences: List[str] = Field(default_factory=list)
    results: List[ResultNode] = Field(default_factory=list)
    success: bool = False
    response_format: List[Dict[str, Any]] = Field(default_factory=list)


class InferenceAgent:
    def __init__(self):
        self.finished = False
        self.logger = get_logger(__name__)
        self.prompt = None
        self.fields = None
        self.uploaded_file = None
        self.graph = self._build_graph()


    def feed_file_to_agent(self, data: bytes):
        '''
        Method to assign the contents of the file to the agent.
        '''
        self.uploaded_file = data

    
    def capture_data(self, initial_state: InferenceState) -> InferenceState:
        final_state =self.graph.invoke(initial_state)
        return final_state


    def _build_graph(self) -> StateGraph:
        '''
        Build the graph to run the inference
        '''

        self.logger.info("Building the inference graph...")

        # Create the workflow
        workflow = StateGraph(InferenceState)

        # Add the initial node for setup
        workflow.add_node("setup", self._run_setup)
        
        # Add multiple inference nodes to deal with different LLMs
        workflow.add_node("infer_with_gemini", self._run_gemini_inference)
        # workflow.add_node("infer_with_deepseek", self._run_deepseek_inference)

        # Add a consolidation node
        workflow.add_node("consolidate", self._run_consolidation)

        # Add the rest of the nodes
        # workflow.add_node("determine_success", self._determine_success)
        workflow.add_node("create_output", self._create_output)
        workflow.add_node("handle_failure", self._run_handle_failure)
        
        workflow.set_entry_point("setup")

        # Branch out nodes to run every inference in parallel
        workflow.add_edge("setup", "infer_with_gemini")
        # workflow.add_edge("setup", "infer_with_deepseek")

        # Join back inference nodes to consolidate results
        # workflow.add_edge(["infer_with_gemini", "infer_with_deepseek"], "consolidate")
        workflow.add_edge("infer_with_gemini", "consolidate")

        # workflow.add_edge("consolidate", "determine_success")

        # Add extra nodes to complete workflow
        # workflow.add_edge("consolidate", "determine_success")

        # Add node that defines whether the inference was successful
        workflow.add_conditional_edges(
            "consolidate",
            self._determine_success,
            {
                "success": "create_output",
                "failure": "handle_failure"
            }
        )

        return workflow.compile()


    def _run_setup(self, state: InferenceState) -> InferenceState:
        '''
        This node has to build the initial state
        '''
        self.logger.info("Entering RUN SETUP node...")

        # Create the appropiate field list and prompt
        # This impacts the AGENT. The STATE remains unchanged
        self._set_fields_to_infer()
        self._set_prompt(self.fields)

        return state
        

    def _run_gemini_inference(self, state: InferenceState) -> InferenceState:
        '''
        Method to run the query through Gemini
        '''
        self.logger.info("Entering GEMINMI INFERENCE node...")

        # Obtain variables related to Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL")

        # Create Gemini client
        client = genai.Client(api_key=api_key)


        base64_data = base64.b64encode(self.uploaded_file).decode('utf-8')

        # Run inference on Gemini client. Pass the file as an inline stream of bytes
        response = client.models.generate_content(
            model=model,
            contents=[
                self.prompt,
                {
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": base64_data
                    }
                }
            ]
        )

        # Parse and normalize response
        content_json = self._clean_and_parse_json(response.candidates[0].content.parts[0].text)
        json_response = self._normalize_fields(content_json, self.fields)

        self.logger.info("Information obtained from GEMINI...")

        # Add information to the state.
        # First, add the inference tool to inferences.
        # Then, add the results to the list of local results.
        state.inferences.append("gemini")
        state.local_results.append(("gemini", json_response))

        # Return the state with the changes
        return state


    def _run_deepseek_inference(self, state: InferenceState):
        pass

    def _run_consolidation(self, state: InferenceState) -> InferenceState:
        '''
        Method to consolidate the results of multiple different calls.
        '''
        self.logger.info("Entering CONSOLIDATION node...")

        # Iterate through every LLM used and the results obtained in local results
        for llm_name, result in state.local_results:

            # Take each field and the information related to that field
            for field, info in result.items():

                # Find the fields already captured in the final result. If the current
                # field is not there, create a new node and populate values
                fields_in_results = {node.field for node in state.results}
                if not (field in fields_in_results):
                    new_result = ResultNode()
                    new_result.field = field
                    new_result.found = True if info["match"] else False
                    new_result.values.append(info["value"])
                    new_result.probable_value = info["value"]
                    new_result.confidences.append(info["confidence"])
                    new_result.found_by.append(llm_name)
                    new_result.found_multiple = False
                    new_result.low_confidence = True if info["confidence"] < 90 else False
                    state.results.append(new_result)

                # If the field was already appended, tweak the values --- TO DO!!!                
                else:
                    pass
        
        return state
    

    def _create_output(self, state: InferenceState) -> InferenceState:
        '''
        Method to create the output that will constitute the response of the API call.
        '''
        self.logger.info("Entering CREATE OUTPUT node...")

        # output_format = json.dumps(
        #         [result.model_dump() for result in state.results],
        #         ensure_ascii=False
        #     )
        
        output_data = [result.model_dump() for result in state.results]
        state.response_format = output_data
        return state


    def _run_handle_failure(self, state: InferenceState):
        pass

    def _determine_success(self, state: InferenceState) -> str:
        '''
        Method to determine if the inference was a success.
        
        We consider success if at least one field was found.'''
        self.logger.info("Entering DETERMINE SUCCESS node...")

        if len(state.results) >= 1:
            state.success = True
            return "success"
        else:
            state.success = False
            return "failure"
        

    def build_initial_state(self) -> InferenceState:
        '''
        Method to build the initial workflow state for the agent
        '''
        
        # Set the field list
        self._set_fields_to_infer()

        # Set the correct prompt
        self._set_prompt()

        state = InferenceState()

        return state


    def _set_fields_to_infer(self):
        '''
        Method to add the fields needed for this particula inference.
        
        For now there is only one option. In the future, there will be the option of selecting
        from a list of different possibilities.
        '''

        self.logger.info("Creating the right fields to infer...")
        
        # Create the fields
        fields = {
            "nombre_contacto": "Nombre del contacto principal contenido en el document",
            "rut_contacto": "RUT del contacto principal del comercio. Puede estar en formato '12345678-9' o '12.345.678-9', pero SIEMPRE expresar en formato '12345678-9'.",
            "num_serie": "Número de serie del documento de identidad del contacto principal. Formato '111.111.111' o '111111111'. Puede contener letras pero NUNCA guiones. El formato de salida es siempre '111111111'."
        }
        
        # Add them to the agent property.
        self.fields = fields

    
    def _set_prompt(self, fields_dict: Dict[str, str]):
        '''
        Method to create the prompt for the inteference.
        
        In principle, this prompt could be generic, though for simplicity it's probably a
        good option to have more than one version.
        '''

        self.logger.info("Creating the prompt...")

        prompt=f'''
        Eres un asistente de extracción de información.

        Vas a recibir 2 entradas:
        1) Un documento, quue consiste en una imagen que captura información relevante
        2) Un diccionario de campos (field descriptions) para extraer. Cada campo del diccionario contiene un nombre y la explicación de lo que hay que extraer.

        Por cada campo DEBES determinar si el documento contiene o no la información buscada.

        Retorna un diccionario JSON cuyas claves son nombres de campo y cuyos valores son objetos con la estructura de abajo.
        DEBE existir un objeto con exactamente estas atributos en todos los casos, incluso para campos no encontrados:
        {{
            "match": boolean,
            "value": string | null,
            "explanation": string | null,
            "confidence": int (0-100) | null
        }}

        Rules:
        - match=true only if the field is **explicitly present** in the document.
        - If match=false, then set value=null, explanation=null, confidence=null.
        - explanation must reference **where** or **how** the model inferred the value
        (e.g., “Found in line about business owner: ‘Razon social:…’”).
        - confidence is 0 to 100. Use higher confidence when text is direct and explicit.
        - If a field has "rut" in its name, express the value without '.' in it, no matter how it comes
        (e.g. if it is '10.345.678-2', express it as '10345678-2').
        - one-word values for fields must start with an uppercase or be all caps (only if this is how they appeared in the document)
        -'num_serie' must also be expressed with no '.' in it (e.g., instead of '123.456.789', express it as '123456789').
        - If inferred but not explicit, match=true but confidence must be <70 and explanation must state inference.
        - DO NOT hallucinate values not suggested in the text.
        - If not all conditions for a value are present, confidence must be <70.
        - Answer only JSON. No prose outside JSON.

        Field descriptions:
        {fields_dict}

        '''

        # Add this prompt to the agent
        self.prompt = prompt
        
        pass


    def _clean_and_parse_json(self, text: str):
    # Remove markdown fences like ```json ... ```
        cleaned = re.sub(r"```[a-zA-Z]*", "", text)   # remove ```json or ``` or ```xyz
        cleaned = cleaned.replace("```", "")          # remove closing fences
        cleaned = cleaned.strip()

        return json.loads(cleaned)
    

    def _normalize_fields(self, text, expected_fields):
        '''Method to normalize fields, in case of nulls or some other problem.'''
        result = {}

        for field in expected_fields:
            val =  text.get(field, None)
    
            # If the value is found, create it in the new result
            if isinstance(val, dict):
                result[field] = {
                    "match": bool(val.get("match")) if val.get("match") is not None else False,
                    "value": val.get("value", None),
                    "explanation": val.get("explanation", None),
                    "confidence": val.get("confidence", 0) if val.get("confidence") is not None else 0
                }

            # If the value is None, create an empty template
            elif val is None:
                result[field] = {"match": False, "value": None, "explanation": None, "confidence": 0}

            # If there is a value, but with no structure, create structure with low confidence
            else:
                result[field] = {
                    "match": True,
                    "value": val,
                    "explanation": "Modelo retornó valor sin estructura, asumiremos baja confianza",
                    "confidence": 30
                }

        return result
