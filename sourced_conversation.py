import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun, Callbacks
from langchain.chains import ReduceDocumentsChain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.map_reduce_prompt import COMBINE_PROMPT, EXAMPLE_PROMPT, QUESTION_PROMPT
from langchain.schema import BasePromptTemplate, BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage

# Depending on the memory type and configuration, the chat history format may differ.
# This needs to be consolidated.
CHAT_TURN_TYPE = Union[Tuple[str, str], BaseMessage]

class ConversationalQAWithSourcesChain(RetrievalQAWithSourcesChain):
    """Question-answering with memory and sources over an index."""
    
    history_key: str = "chat_history"  #: :meta private:
    return_generated_question: bool = False
    """Return the generated question as part of the final result."""
    get_chat_history: Optional[Callable[[List[CHAT_TURN_TYPE]], str]] = None
    """An optional function to get a string of the chat history.
    If None is provided, will use a default."""
    question_generator: LLMChain
    """The chain used to generate a new question for the sake of retrieval.
    This chain will take in the current question (with variable `question`)
    and any chat history (with variable `chat_history`) and will produce
    a new standalone question to be used later on."""
    rephrase_question: bool = True
    """Whether or not to pass the new generated question to the combine_docs_chain.
    If True, will pass the new generated question along.
    If False, will only use the new generated question for retrieval and pass the
    original question along to the combine_docs_chain."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        callbacks: Callbacks = None,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = COMBINE_PROMPT,
        **kwargs: Any,
    ) -> RetrievalQAWithSourcesChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into a standalone question.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into a standalone question. If none is
                provided, will default to `llm`.
            callbacks: Callbacks to pass to all subchains.
            **kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )

        combine_results_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=combine_prompt),
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        llm_question_chain = LLMChain(llm=llm, prompt=question_prompt)
        reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_results_chain)
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",
        )

        return cls(
            retriever=retriever,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            combine_documents_chain=combine_documents_chain,
            **kwargs,
        )

    @property
    def output_keys(self) -> List[str]:

        """Return output key.

        :meta private:
        """
        output = super().output_keys
        if self.return_generated_question:
            _output_keys = _output_keys + ["generated_question"]
        return output
    
    def _split_sources(self, answer: str) -> Tuple[str, str]:
        """Split sources from answer."""
        if re.search(r"SOURCES:\s", answer):
            answer, sources = re.split(r"SOURCES:\s|QUESTION:\s", answer)[:2]
            sources = re.split(r"\n", sources)[0]
        else:
            sources = ""
        return answer, sources

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None, ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs[self.history_key])

        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = self.question_generator.run(question=question, chat_history=chat_history_str, callbacks=callbacks)
        else:
            new_question = question

        accepts_run_manager = ("run_manager" in inspect.signature(self._get_docs).parameters)
        if accepts_run_manager:
            docs = self._get_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(new_question, inputs)  # type: ignore[call-arg]
        
        new_inputs = inputs.copy()
        if self.rephrase_question:
            new_inputs[self.question_key] = new_question
        new_inputs[self.history_key] = chat_history_str

        answer = self.combine_documents_chain.run(input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs)
        answer, sources = self._split_sources(answer)
        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        if self.return_generated_question:
            result["generated_question"] = new_question
        return result
    
    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None, ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.question_key]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs[self.history_key])

        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(question=question, chat_history=chat_history_str, callbacks=callbacks)
        else:
            new_question = question

        accepts_run_manager = ("run_manager" in inspect.signature(self._get_docs).parameters)
        if accepts_run_manager:
            docs = await self._get_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs = await self._get_docs(new_question, inputs)  # type: ignore[call-arg]

        new_inputs = inputs.copy()
        if self.rephrase_question:
            new_inputs[self.question_key] = new_question
        new_inputs[self.history_key] = chat_history_str

        answer = await self.combine_documents_chain.arun(input_documents=docs, callbacks=_run_manager.get_child(), **inputs)
        answer, sources = self._split_sources(answer)
        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        if self.return_generated_question:
            result["generated_question"] = new_question
        return result
    
    def save(self, file_path: Union[Path, str]) -> None:
        if self.get_chat_history:
            raise ValueError("Chain not saveable when `get_chat_history` is not None.")
        super().save(file_path)
    
    def _get_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: CallbackManagerForChainRun) -> List[Document]:
        """
        Get docs.
        """
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )
        return self._reduce_tokens_below_limit(docs)

    async def _aget_docs(self, question: str, inputs: Dict[str, Any], *, run_manager: AsyncCallbackManagerForChainRun) -> List[Document]:
        """
        Get docs.
        """
        docs = await self.retriever.aget_relevant_documents(question, callbacks=run_manager.get_child())
        return self._reduce_tokens_below_limit(docs)