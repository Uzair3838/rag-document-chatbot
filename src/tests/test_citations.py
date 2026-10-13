import unittest
from unittest.mock import patch, MagicMock
from src.rag_pipeline import answer_question

class TestCitations(unittest.TestCase):
    @patch('src.rag_pipeline.generate_answer')
    @patch('src.rag_pipeline.rerank_documents')
    @patch('src.rag_pipeline.retrieve_documents')
    def test_answer_question_citations(self, mock_retrieve, mock_rerank, mock_generate):
        # Setup common mocks
        mock_generate.return_value = "The company grew by 20% in 2023."

        # Case 1: Cross-Platform Path Normalization
        # Should treat dir/doc.pdf and dir\\doc.pdf on the same page as one reference
        # Using raw strings to ensure backslashes are preserved in the mock data
        mock_retrieve.return_value = [
            {"text": "chunk 1", "source": "dir/report.pdf", "page": 1},
            {"text": "chunk 2", "source": r"dir\report.pdf", "page": 1},
        ]
        mock_rerank.side_effect = lambda query, documents, top_n: [
            {**doc, "rerank_score": 1.0} for doc in documents
        ]

        _, refs = answer_question("Question 1", "session_1")
        self.assertEqual(refs, ["report.pdf, Page 1"], "Should normalize paths and deduplicate same page")

        # Case 2: Intra-Page Deduplication
        # Multiple chunks from the same page
        mock_retrieve.return_value = [
            {"text": "chunk 1", "source": "report.pdf", "page": 5},
            {"text": "chunk 2", "source": "report.pdf", "page": 5},
            {"text": "chunk 3", "source": "report.pdf", "page": 5},
        ]
        _, refs = answer_question("Question 2", "session_2")
        self.assertEqual(refs, ["report.pdf, Page 5"], "Should deduplicate multiple chunks from the same page")

        # Case 3: Inter-Page Separation
        # Chunks from different pages of the same document
        mock_retrieve.return_value = [
            {"text": "chunk 1", "source": "report.pdf", "page": 5},
            {"text": "chunk 2", "source": "report.pdf", "page": 10},
        ]
        _, refs = answer_question("Question 3", "session_3")
        self.assertEqual(refs, ["report.pdf, Page 5", "report.pdf, Page 10"], "Should keep different pages separate")

        # Case 4: Distinct Files Same Page
        # Different filenames on the same page number
        mock_retrieve.return_value = [
            {"text": "chunk 1", "source": "report.pdf", "page": 1},
            {"text": "chunk 2", "source": "summary.pdf", "page": 1},
        ]
        _, refs = answer_question("Question 4", "session_4")
        self.assertEqual(refs, ["report.pdf, Page 1", "summary.pdf, Page 1"], "Should keep different files separate even on same page")

        # Case 5: Order Preservation
        # Ensure references appear in the order they were retrieved/reranked
        mock_retrieve.return_value = [
            {"text": "chunk 1", "source": "A.pdf", "page": 1},
            {"text": "chunk 2", "source": "B.pdf", "page": 1},
            {"text": "chunk 3", "source": "C.pdf", "page": 1},
        ]
        _, refs = answer_question("Question 5", "session_5")
        self.assertEqual(refs, ["A.pdf, Page 1", "B.pdf, Page 1", "C.pdf, Page 1"], "Should preserve encounter order")

if __name__ == '__main__':
    unittest.main()
