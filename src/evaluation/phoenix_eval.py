import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from loguru import logger

try:
    from phoenix.evals import HallucinationEvaluator, QAEvaluator, OpenAIModel
    from phoenix.evals import evaluate_dataframe
    from phoenix.session.client import PhoenixSession

    PHOENIX_AVAILABLE = True
except ImportError:
    logger.warning(
        "Phoenix not installed. Install with: pip install arize-phoenix[evals]"
    )
    PHOENIX_AVAILABLE = False

from .evaluator import Evaluator


class PhoenixEvaluator:
    """
    Phoenix-based evaluator for agent performance metrics including
    hallucination detection and quality assessment
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize Phoenix evaluator

        Args:
            openai_api_key: OpenAI API key for evaluation models
        """
        if not PHOENIX_AVAILABLE:
            raise ImportError(
                "Phoenix is not installed. Install with: pip install arize-phoenix[evals]"
            )

        self.openai_api_key = openai_api_key

        # Initialize models
        if PHOENIX_AVAILABLE:
            self.openai_model = OpenAIModel(
                model="gpt-3.5-turbo", api_key=openai_api_key
            )

            # Initialize evaluators
            self.hallucination_evaluator = HallucinationEvaluator(
                model=self.openai_model
            )

            self.qa_evaluator = QAEvaluator(model=self.openai_model)

        self.qa_evaluator = QAEvaluator(
            model=self.openai_model,
            template="Given the question and context, evaluate the quality of the answer:\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer: {answer}",
        )

        logger.info("Phoenix evaluator initialized")

    async def evaluate_agent_responses(
        self, responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate agent responses for quality and hallucinations

        Args:
            responses: List of agent responses with context

        Returns:
            Dictionary with evaluation metrics
        """
        if not responses:
            return {"error": "No responses to evaluate"}

        try:
            # Convert to DataFrame for Phoenix evaluation
            df = pd.DataFrame(responses)

            # Ensure required columns
            required_cols = ["context", "response"]
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Missing required column '{col}' for evaluation")
                    return {"error": f"Missing required column '{col}'"}

            # Evaluate hallucinations
            hallucination_scores = await self._evaluate_hallucinations(df)

            # Evaluate QA quality if question column exists
            qa_scores = None
            if "question" in df.columns:
                qa_scores = await self._evaluate_qa_quality(df)

            # Calculate aggregate metrics
            total_responses = len(responses)
            avg_hallucination_score = (
                sum(hallucination_scores) / len(hallucination_scores)
                if hallucination_scores
                else 0
            )
            hallucination_rate = (
                sum(1 for score in hallucination_scores if score < 0.5)
                / len(hallucination_scores)
                if hallucination_scores
                else 0
            )

            results = {
                "total_responses": total_responses,
                "average_hallucination_score": avg_hallucination_score,
                "hallucination_rate": hallucination_rate,
                "hallucination_scores": hallucination_scores,
                "evaluation_timestamp": datetime.now().isoformat(),
            }

            if qa_scores:
                avg_qa_score = sum(qa_scores) / len(qa_scores)
                results.update(
                    {
                        "average_qa_score": avg_qa_score,
                        "qa_scores": qa_scores,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in Phoenix evaluation: {e}")
            return {"error": str(e)}

    async def _evaluate_hallucinations(self, df: pd.DataFrame) -> List[float]:
        """Evaluate hallucinations in responses"""
        try:
            if not PHOENIX_AVAILABLE:
                return []

            # Run Phoenix hallucination evaluation
            hallucination_df = evaluate_dataframe(
                df=df,
                evaluators=[self.hallucination_evaluator],
                provide_explanation=True,
            )

            # Extract scores
            scores = []
            for _, row in hallucination_df.iterrows():
                # Phoenix returns label and score
                if "score" in row:
                    scores.append(float(row["score"]))
                elif "label" in row:
                    # Convert binary label to score
                    label = str(row["label"]).lower()
                    score = 1.0 if label == "yes" or label == "factual" else 0.0
                    scores.append(score)

            return scores

        except Exception as e:
            logger.error(f"Error in hallucination evaluation: {e}")
            return []

    async def _evaluate_qa_quality(self, df: pd.DataFrame) -> List[float]:
        """Evaluate QA quality"""
        try:
            if not PHOENIX_AVAILABLE:
                return []

            # Run Phoenix QA evaluation
            qa_df = evaluate_dataframe(
                df=df, evaluators=[self.qa_evaluator], provide_explanation=True
            )

            # Extract scores
            scores = []
            for _, row in qa_df.iterrows():
                if "score" in row:
                    scores.append(float(row["score"]))
                elif "label" in row:
                    # Convert quality label to score
                    label = str(row["label"]).lower()
                    if label in ["excellent", "good", "high"]:
                        score = 0.8
                    elif label in ["fair", "medium", "moderate"]:
                        score = 0.5
                    else:
                        score = 0.2
                    scores.append(score)

            return scores

        except Exception as e:
            logger.error(f"Error in QA evaluation: {e}")
            return []


class PhoenixIntegration:
    """
    Integration layer for Phoenix evaluation with the main Evaluator
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize Phoenix integration"""
        self.phoenix_evaluator = None
        self.openai_api_key = openai_api_key

        if PHOENIX_AVAILABLE and openai_api_key:
            try:
                self.phoenix_evaluator = PhoenixEvaluator(openai_api_key=openai_api_key)
                logger.info("Phoenix integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Phoenix integration: {e}")
        else:
            logger.warning("Phoenix integration not available")

    async def evaluate_agent_actions(
        self, evaluator: Evaluator, save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Evaluate agent actions using Phoenix

        Args:
            evaluator: Main evaluator instance with action logs
            save_path: Path to save evaluation results

        Returns:
            Dictionary with Phoenix evaluation metrics
        """
        if not self.phoenix_evaluator:
            return {"error": "Phoenix evaluator not available"}

        try:
            # Get action data from evaluator
            actions_data = evaluator.get_evaluation_data()
            if not actions_data or "actions" not in actions_data:
                return {"error": "No action data available for evaluation"}

            # Convert actions to evaluation format
            evaluation_responses = []
            for action in actions_data["actions"]:
                response = {
                    "context": action.get("description", ""),
                    "response": action.get("result", ""),
                    "agent_name": action.get("agent_name", ""),
                    "action_type": action.get("action_type", ""),
                }

                # Add question if available
                if "task_description" in action:
                    response["question"] = action["task_description"]

                evaluation_responses.append(response)

            # Run evaluation
            results = await self.phoenix_evaluator.evaluate_agent_responses(
                evaluation_responses
            )

            # Save results if path provided
            if save_path:
                save_path.mkdir(parents=True, exist_ok=True)
                with open(save_path / "phoenix_evaluation.json", "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Phoenix evaluation saved to {save_path}")

            return results

        except Exception as e:
            logger.error(f"Error in Phoenix integration: {e}")
            return {"error": str(e)}
