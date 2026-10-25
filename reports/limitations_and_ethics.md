# LLM Limitations and Ethical Considerations

## Limitations Identified

The evaluation compared Google FLAN-T5 Small and FLAN-T5 Base on financial forecasting and risk analysis tasks. Both models showed significant limitations in producing accurate, relevant, and reliable financial predictions.

In numerical forecasting, neither model consistently outperformed a simple historical baseline. FLAN-T5 Small generated invalid stock and gold price predictions, while FLAN-T5 Base produced a negative stock price prediction. These results demonstrate the need to validate model outputs before using them for financial decisions.

The risk analysis experiments showed that FLAN-T5 Base performed better at identifying credit, liquidity, and market risks. However, both models struggled to provide practical mitigation strategies. FLAN-T5 Small frequently repeated instructions instead of answering the questions.

Prompt optimization produced mixed results. FLAN-T5 Base's qualitative forecasting score improved from 1.44 to 2.00, while its risk analysis score decreased from 2.56 to 1.89. FLAN-T5 Small experienced reduced performance in both categories. These findings suggest that more detailed prompts do not guarantee better financial analysis.

## Ethical Considerations

**Financial misinformation:** Incorrect forecasts and unsupported investment recommendations could mislead users and lead to financial losses.

**Transparency:** AI-generated financial predictions should clearly communicate assumptions, uncertainty, limitations, and supporting evidence.

**Bias and fairness:** Models may reflect biases in their training data or fail to account for economic conditions affecting different populations and financial institutions.

**Data privacy:** Financial applications must protect sensitive customer information and avoid exposing confidential financial records through model inputs or outputs.

**Human oversight:** LLM-generated forecasts and risk assessments should be reviewed by qualified financial professionals before being used in important financial decisions.

## Recommendations

1. Use domain-specific financial models or fine-tune suitable models with carefully validated financial datasets.
2. Combine LLMs with statistical forecasting methods and structured numerical analysis.
3. Implement output validation to reject invalid values, unsupported claims, and inconsistent recommendations.
4. Improve prompts using relevant historical financial data, clearly defined tasks, and structured response formats.
5. Evaluate forecasting performance across multiple historical periods using quantitative metrics and appropriate baselines.
6. Include uncertainty estimates, data provenance, and human review in financial decision-making workflows.

## Evaluation Limitations

The numerical forecasting experiments used one historical prediction date per financial variable, limiting how broadly the results can be generalized.

The experiments were retrospective, and pretrained models may have encountered some historical evaluation information during training. Additionally, the CPI data may contain revisions made after the forecast period.

The qualitative evaluation scores were manually assigned and may involve subjective judgment. Both evaluated models belong to the FLAN-T5 family, so this comparison does not represent the performance of all LLM architectures.

The original and optimized risk experiments also used slightly different generation settings, meaning observed differences cannot be attributed exclusively to prompt engineering.

These limitations should be considered when interpreting the results.