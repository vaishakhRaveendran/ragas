import typing as t
import re
import numpy as np
from langchain_core.documents import Document
from ragas_experimental.testset.extractors import (
    DocumentExtractor,
    email_extractor,
    headline_extractor,
    keyphrase_extractor,
    link_extractor,
    summary_extractor,
    title_extractor,
)
from ragas_experimental.testset.generators import (
    QADistribution,
    TestGenerator,
)
from ragas_experimental.testset.graph import Node, NodeLevel
from ragas_experimental.testset.questions import (
    DEFAULT_DISTRIBUTION,
    ComparativeAbstractQA,
)
from ragas_experimental.testset.relationships import (
    Cosine,
    Jaccard,
    RelationshipBuilder,
)
from ragas_experimental.testset.splitters import HeadlineSplitter
from ragas_experimental.testset.utils import rng
from ragas_experimental.testset.generators.base import TestDataset

from ragas.embeddings import embedding_factory
from ragas.executor import Executor
from ragas._analytics import TestsetGenerationEvent, track
from ragas.llms.base import llm_factory
from ragas.utils import check_if_sum_is_close

# Only ComparativeQA is used
comparative_qa = ComparativeAbstractQA(distribution=DEFAULT_DISTRIBUTION)

QA_DISTRIBUTION = QADistribution(
    question_types=[comparative_qa],
    probabilities=[1.0],
)

class SimpleTestGenerator(TestGenerator):
    def __post_init__(self):
        self.llm = self.llm or llm_factory()
        print(f'llm used will be {self.llm}')
        self.embedding = self.embedding or embedding_factory()
        print(f'embedding used will be {self.embedding}')

    def _document_extraction(self, docs: t.Sequence[Document]) -> t.Sequence[Document]:
        exec = Executor(
            desc="Document extraction",
            keep_progress_bar=True,
            raise_exceptions=True,
            run_config=None,
        )
        extractors = [
            summary_extractor,
            link_extractor,
            email_extractor,
            keyphrase_extractor,
            title_extractor,
            headline_extractor,
        ]
        doc_extractor = DocumentExtractor(extractors=extractors)
        exec.submit(doc_extractor.extract, docs)
        docs = exec.results()
        return docs

    def generate(
            self,
            docs: t.Sequence[Document],
            test_size: int,
            distribution: QADistribution = QA_DISTRIBUTION,
    ) -> TestDataset:
        try:
            # Input validation
            if not check_if_sum_is_close(list(distribution.values()), 1.0, 3):
                raise ValueError(
                    f"Distribution does not sum to 1.0 [got {sum(list(distribution.values()))}]. Please check the distribution."
                )

            # Document extraction
            try:
                extractors = [
                    summary_extractor,
                    link_extractor,
                    email_extractor,
                    keyphrase_extractor,
                    title_extractor,
                    headline_extractor,
                ]

                doc_extractor = DocumentExtractor(
                    extractors=extractors, llm=self.llm, embedding=self.embedding
                )
                docs = doc_extractor.extract(docs)

            except re.error as e:
                print(f"Regular expression error in extractor: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Error in document extraction: {str(e)}")

            # Document splitting
            try:
                splitter = HeadlineSplitter(common_metadata_keys=["source", "title"])
                nodes, relationships = splitter.split_documents(docs, "headlines")
            except Exception as e:
                raise RuntimeError(f"Error in document splitting: {str(e)}")

            # Node embedding and extraction
            try:
                nodes = doc_extractor.embed(
                    nodes,
                    ["page_content", "summary"],
                    {
                        "page_content": [
                            NodeLevel.LEVEL_1,
                            NodeLevel.LEVEL_2,
                            NodeLevel.LEVEL_3,
                        ],
                        "summary": [NodeLevel.LEVEL_0],
                    },
                )
                node_extractor = DocumentExtractor(
                    extractors=[keyphrase_extractor], llm=self.llm, embedding=self.embedding
                )
                nodes = node_extractor.extract(
                    nodes, [NodeLevel.LEVEL_1, NodeLevel.LEVEL_2, NodeLevel.LEVEL_3]
                )
            except Exception as e:
                raise RuntimeError(f"Error in node embedding and extraction: {str(e)}")

            # Relationship building
            try:
                jaccard = Jaccard(
                    name="jaccard_over_keyphrases",
                    attribute1="keyphrases",
                    attribute2="keyphrases",
                    type="fuzzy",
                    threshold=50,
                )
                cosine = Cosine(
                    name="summary_similarity",
                    attribute1="summary_embedding",
                    attribute2="summary_embedding",
                )
                if nodes:
                    assert all(
                        isinstance(node, Node) for node in nodes
                    ), "Nodes must be of type Node"

                nodes, relationships = RelationshipBuilder.form_relations(
                    nodes,
                    relationships,
                    similarity_functions=[jaccard, cosine],
                    node_level=NodeLevel.LEVEL_0,
                )
            except Exception as e:
                raise RuntimeError(f"Error in relationship building: {str(e)}")

            # Question generation setup
            try:
                exec = Executor(
                    desc="Generating",
                    keep_progress_bar=True,
                    raise_exceptions=True,
                    run_config=None,
                )

                for qa in distribution.keys():
                    qa.nodes = nodes
                    qa.relationships = relationships
                    if qa.llm is None:
                        qa.llm = self.llm
                    if qa.embedding is None:
                        qa.embedding = self.embedding
            except Exception as e:
                raise RuntimeError(f"Error in question generation setup: {str(e)}")

            # Question generation (ComparativeQA only)
            try:
                comparative_qa = list(distribution.keys())[0]  # There should be only one
                exec.submit(
                    comparative_qa.generate_questions, query=None, kwargs=None, num_samples=test_size
                )
            except Exception as e:
                raise RuntimeError(f"Error in question generation: {str(e)}")

            # Result compilation
            try:
                results = exec.results()
                results = TestDataset([result for result in results if result is not None])
            except Exception as e:
                raise RuntimeError(f"Error in result compilation: {str(e)}")

            # Analytics tracking
            try:
                track(
                    TestsetGenerationEvent(
                        event_type="testset_generation",
                        evolution_names=[""],
                        evolution_percentages=[0.0],
                        num_rows=test_size,
                        language="",
                        is_experiment=True,
                    )
                )
            except Exception as e:
                print(f"Error in analytics tracking: {str(e)}")

            return results

        except Exception as e:
            print(f"An error occurred in the generate method: {str(e)}")
            raise