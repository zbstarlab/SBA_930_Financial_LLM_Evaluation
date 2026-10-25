# SBA 930 – Evaluating LLMs for Financial Forecasting and Risk Analysis

## 1. Introduction and Project Overview
This project evaluates the performance of Large Language Models (LLMs) in financial forecasting and risk analysis. Two Hugging Face models, Google FLAN-T5 Small and FLAN-T5 Base, were tested to determine how effectively they analyze financial data, predict market trends, and identify financial risks. Historical data from the SPY and GLD ETFs and the US Consumer Price Index (CPI) were used to evaluate stock market, commodity price, and inflation forecasting. Model predictions were compared with actual historical values and a simple baseline forecasting method. The project also evaluated the models' ability to identify market volatility, credit risk, and liquidity risk and recommend mitigation strategies. Prompt optimization was applied to determine whether more structured instructions could improve response quality. The results were assessed based on forecasting error, relevance, financial accuracy, reliability, and risk identification performance. The findings highlight the strengths and limitations of using smaller LLMs for financial analysis and provide recommendations for improving their effectiveness.

## 2. Datasets and Methodology
### Datasets

Three historical financial datasets were used to evaluate the models. The SPY ETF dataset contained 1,258 daily observations from 2020 to 2024 and was used for stock market forecasting and volatility analysis. The GLD ETF dataset also contained 1,258 daily observations and served as a proxy for historical gold prices in commodity forecasting. Both datasets were downloaded using the Python yfinance library. The US Consumer Price Index (CPI) dataset contained 60 monthly observations obtained from Federal Reserve Economic Data (FRED) and was used to analyze inflation trends.

The datasets were cleaned using pandas by checking for missing values, removing duplicate dates, and sorting observations chronologically. Daily percentage returns were calculated for the market datasets, while year-over-year inflation rates were calculated using CPI values. The data was split into a historical reference period (2020–2023) and an evaluation period (2024). The historical observations were used to construct forecasting prompts, while the January 2024 outcomes were used to measure prediction accuracy.

### LLM Selection and Implementation

Two instruction-tuned Hugging Face models were selected: Google FLAN-T5 Small, with approximately 80 million parameters, and FLAN-T5 Base, with approximately 250 million parameters. Both models were downloaded through the Transformers library and executed locally using PyTorch with Apple Metal Performance Shaders (MPS) acceleration.

Both models received equivalent forecasting and risk-analysis tasks. Deterministic generation settings were used to improve reproducibility. The generated responses were stored in CSV files for evaluation and comparison.

### Evaluation Methodology

The evaluation included qualitative financial analysis, numerical forecasting, risk identification, and prompt optimization. Qualitative responses were manually rated on a five-point scale based on relevance, financial accuracy, and reliability. Risk-analysis responses were separately rated for risk identification, mitigation quality, and clarity.

Numerical forecasting experiments used December 2023 financial values to predict January 2024 SPY prices, GLD prices, and CPI inflation rates. Predictions were compared against actual historical outcomes using absolute error. A naive persistence baseline, which predicted that the previous period's value would remain unchanged, was included for comparison. Invalid stock and commodity price predictions were identified separately.

Prompt optimization was tested by replacing broad financial questions with more structured instructions. Original and optimized responses were evaluated using the same scoring criteria within each task category, and average score changes were calculated to assess the observed impact.

### Methodological Limitations

The numerical evaluation used only one forecast date for each financial variable. Therefore, the results provide a limited comparison rather than a comprehensive measure of forecasting accuracy. The pretrained models may also have encountered information about 2024 during training, and the CPI observations may include later revisions. Both models belong to the FLAN-T5 family, which limits architectural diversity. Additionally, the original and optimized risk-analysis experiments used slightly different generation settings, so their score differences cannot be attributed entirely to prompt optimization.


## 3. LLM Evaluation for Financial Forecasting
### Stock Market Forecasting

Both models were evaluated on their ability to predict SPY ETF closing prices for January 2024 using the December 2023 closing price as historical context. Their predictions were compared with the actual January closing price of $467.58 and a naive persistence baseline.

| Model | Predicted Price | Actual Price | Absolute Error |
|---|---:|---:|---:|
| FLAN-T5 Small | $0.00 | $467.58 | $467.58 |
| FLAN-T5 Base | -$457.50 | $467.58 | $925.08 |
| Naive Baseline | $460.25 | $467.58 | $7.33 |

Neither model produced a financially valid stock price prediction. FLAN-T5 Small predicted zero, while FLAN-T5 Base generated a negative price. The naive baseline had the lowest absolute error at $7.33. These findings suggest that neither model performed reliably on this numerical stock forecasting task.

### Commodity Price Forecasting

The GLD ETF was used as a proxy for gold prices to evaluate commodity forecasting performance. Both models were asked to predict the GLD closing price at the end of January 2024 using the December 2023 closing price of $191.17.

| Model | Predicted Price | Actual Price | Absolute Error |
|---|---:|---:|---:|
| FLAN-T5 Small | $0.00 | $188.45 | Invalid prediction |
| FLAN-T5 Base | $179.17 | $188.45 | $9.28 |
| Naive Baseline | $191.17 | $188.45 | $2.72 |

FLAN-T5 Small generated an invalid zero-price prediction. FLAN-T5 Base generated a valid prediction but had an absolute error of $9.28, compared with $2.72 for the naive baseline. The baseline therefore outperformed both models in this experiment.

### Qualitative Financial Analysis

The models were also evaluated on their ability to explain stock market conditions, commodity price trends, and inflation-related economic factors. Responses were manually scored from 1 to 5 for relevance, financial accuracy, and reliability.

| Model | Relevance | Financial Accuracy | Reliability | Overall Score |
|---|---:|---:|---:|---:|
| FLAN-T5 Small | 2.00 | 2.00 | 1.67 | 1.89 |
| FLAN-T5 Base | 2.00 | 1.33 | 1.00 | 1.44 |

FLAN-T5 Small achieved the higher average qualitative score. However, both models frequently generated vague or repetitive answers instead of explaining financial relationships. FLAN-T5 Base occasionally made unsupported suggestions and produced particularly repetitive inflation-related responses.

### Key Findings

The numerical experiments showed that both models struggled to generate reliable financial forecasts. The naive baseline was more accurate for both stock and commodity price predictions. The qualitative evaluation also revealed weak financial reasoning, limited supporting explanations, and inconsistent response quality.

These findings indicate that neither FLAN-T5 Small nor FLAN-T5 Base demonstrated sufficient reliability for independent financial forecasting in the tested scenarios. The results are limited to the selected tasks and historical forecast dates and should not be generalized to all LLMs.

## 4. Economic Trend Forecasting
### Inflation Forecasting Experiment

The economic trend forecasting experiment evaluated whether FLAN-T5 Small and FLAN-T5 Base could predict changes in US inflation. Historical Consumer Price Index (CPI) data from the Federal Reserve Economic Data (FRED) database was used to calculate year-over-year inflation rates. Both models received the December 2023 inflation rate of 3.32% and were asked to predict the January 2024 inflation rate.

The predictions were compared with the actual January 2024 inflation rate of approximately 3.09%. A naive persistence baseline was also included, assuming inflation would remain unchanged from December.

### Forecasting Results

| Model | Predicted Inflation | Actual Inflation | Absolute Error |
|---|---:|---:|---:|
| FLAN-T5 Small | 0.00% | 3.09% | 3.088 pp |
| FLAN-T5 Base | 3.32% | 3.09% | 0.232 pp |
| Naive Baseline | 3.32% | 3.09% | 0.228 pp |

The results show that FLAN-T5 Small produced an inaccurate prediction of zero inflation. FLAN-T5 Base generated a prediction close to the actual inflation rate, with an absolute error of approximately 0.232 percentage points. However, the naive baseline performed slightly better, with an error of 0.228 percentage points.

### Analysis of Accuracy and Reliability

FLAN-T5 Base performed substantially better than FLAN-T5 Small on this particular inflation prediction. However, its result was extremely close to the previous month's inflation rate, suggesting that it did not demonstrate a clear forecasting advantage over a simple persistence baseline.

The evaluation also highlighted the importance of distinguishing numerical accuracy from the quality of financial explanations. A model can generate a prediction close to the actual value without demonstrating reliable economic reasoning.

The experiment used only one forecast period, so its results cannot establish long-term forecasting reliability. Additionally, the models were pretrained and may have encountered information about historical economic events. The CPI data may also include revisions published after the forecast date.

### Key Findings

The economic forecasting experiment demonstrated that the larger FLAN-T5 Base model produced a more accurate inflation prediction than FLAN-T5 Small. However, neither model outperformed the naive historical baseline. These findings suggest that LLM-based economic forecasts should be validated against traditional statistical methods and evaluated across multiple periods before being considered reliable for financial decision-making.


## 5. Risk Identification and Mitigation
### Risk Analysis Experiment

The risk analysis experiment evaluated how effectively FLAN-T5 Small and FLAN-T5 Base could identify financial risks and recommend mitigation strategies. Three scenarios were created to represent market volatility, credit risk, and liquidity risk. Both models were given the same scenarios and asked to identify the main risk and suggest one practical mitigation strategy.

The responses were manually evaluated using three criteria: risk identification, mitigation quality, and clarity. Each criterion was scored from 1 to 5, with higher scores indicating better performance.

### Risk Evaluation Results

| Model | Risk Identification | Mitigation Quality | Clarity | Overall Score |
|---|---:|---:|---:|---:|
| FLAN-T5 Small | 1.00 | 1.00 | 1.33 | 1.11 |
| FLAN-T5 Base | 3.33 | 1.33 | 3.00 | 2.56 |

FLAN-T5 Base achieved a higher overall score of 2.56 compared with 1.11 for FLAN-T5 Small. The larger model was more effective at identifying financial risks, especially those related to credit and liquidity. However, its mitigation recommendations were generally incomplete or impractical.

### Analysis of Identified Risks

**Market Volatility:** The scenario described a 15% decline in stock prices over one month. FLAN-T5 Small mentioned interest rates but failed to identify the main risk clearly. FLAN-T5 Base recognized market volatility but confused it with liquidity risk and provided a vague recommendation to reduce volatility.

**Credit Risk:** The scenario involved a company with high debt, declining revenue, and difficulty making loan payments. FLAN-T5 Small generated a repetitive response without identifying credit risk. FLAN-T5 Base recognized debt and declining revenue as financial concerns but did not explain the possibility of default or suggest an effective mitigation strategy.

**Liquidity Risk:** The scenario described a financial institution unable to meet short-term withdrawal requests. FLAN-T5 Small failed to identify the problem. FLAN-T5 Base correctly recognized the inability to meet withdrawal obligations but suggested withdrawing cash from the institution's accounts, which did not adequately address the underlying liquidity shortage.

### Recommended Risk Mitigation Strategies

Based on these findings, several mitigation strategies were identified.

- **Market Volatility:** Diversify investments, establish appropriate risk limits, and consider hedging strategies to reduce exposure to sudden market movements.
- **Credit Risk:** Evaluate borrower creditworthiness, monitor debt obligations, assess repayment capacity, and reduce exposure to borrowers with elevated default risk.
- **Liquidity Risk:** Maintain sufficient liquid assets, monitor short-term cash flow requirements, and establish contingency funding arrangements.

These recommendations represent appropriate risk-management approaches identified during the analysis rather than strategies consistently generated by the evaluated models.

### Key Findings

FLAN-T5 Base demonstrated better risk identification capabilities than FLAN-T5 Small, but both models struggled to provide reliable mitigation strategies. The results suggest that identifying a financial risk does not necessarily mean a model can recommend an effective solution. Human oversight, domain-specific validation, and structured risk-management frameworks remain important when applying LLMs to financial risk assessment.

## 6. Optimization and Enhancement
### Prompt Optimization Approach

Prompt optimization was used to determine whether more structured instructions could improve the financial analysis and risk assessment capabilities of FLAN-T5 Small and FLAN-T5 Base. The original prompts were relatively broad and allowed the models to generate unstructured responses. The optimized prompts introduced financial analyst roles, specific financial factors, and clearer instructions for generating explanations and mitigation recommendations.

Both models were evaluated using six optimized prompts covering stock market forecasting, commodity prices, inflation forecasting, market volatility, credit risk, and liquidity risk. The generated responses were compared with the original responses using the same manual evaluation criteria.

### Optimization Results

| Model | Category | Original Score | Optimized Score | Change |
|---|---|---:|---:|---:|
| FLAN-T5 Small | Financial Forecasting | 1.89 | 1.44 | -0.44 |
| FLAN-T5 Small | Risk Analysis | 1.11 | 1.00 | -0.11 |
| FLAN-T5 Base | Financial Forecasting | 1.44 | 2.00 | +0.56 |
| FLAN-T5 Base | Risk Analysis | 2.56 | 1.89 | -0.67 |

The results showed mixed effects from prompt optimization. FLAN-T5 Base improved its financial forecasting response quality from 1.44 to 2.00, particularly when identifying factors that influence commodity prices. However, its risk analysis score decreased from 2.56 to 1.89.

FLAN-T5 Small performed worse after optimization in both categories. It frequently repeated instructions instead of answering financial questions, suggesting that more detailed prompts sometimes made it harder for the smaller model to generate useful responses.

### Analysis of Optimization Effectiveness

The experiments demonstrated that providing more detailed instructions does not automatically improve model performance. FLAN-T5 Base showed some improvement in identifying relevant financial concepts, but its responses still lacked detailed reasoning, supporting evidence, and reliable predictions.

The optimized prompts also explicitly mentioned some of the factors the models were expected to discuss. Therefore, improvements in relevance may reflect better repetition or recognition of supplied information rather than independent financial reasoning.

The original risk-analysis experiments used a repetition-control generation setting that was not applied during the optimized experiments. This difference limits our ability to attribute changes in risk-analysis scores entirely to prompt optimization.

### Recommendations for Further Improvement

Several methods could improve the reliability of LLM-based financial analysis:

1. **Financial data grounding:** Provide relevant historical observations and verified economic indicators within prompts to reduce unsupported statements.
2. **Structured output formats:** Require models to produce clearly defined predictions, explanations, assumptions, and risk assessments.
3. **Domain-specific training:** Explore models trained or fine-tuned on financial and economic datasets.
4. **Numerical validation:** Apply automated checks to reject invalid predictions, including negative ETF prices and unsupported numerical claims.
5. **Hybrid forecasting:** Combine LLM-generated explanations with established statistical or machine learning forecasting methods.
6. **Expanded evaluation:** Test models across multiple historical periods and compare results against statistical baselines using consistent metrics.

These recommendations were developed from the observed model limitations. They represent proposed future improvements rather than additional experiments completed in this project.

### Key Findings

Prompt optimization produced limited and inconsistent improvements. FLAN-T5 Base demonstrated better qualitative forecasting performance with structured prompts, while both models continued to struggle with practical financial risk mitigation. The findings suggest that prompt engineering alone is insufficient to make these models reliable for financial forecasting and risk analysis. Additional data grounding, numerical validation, stronger domain-specific models, and human oversight would be necessary for more dependable results.

## 7. LLM Limitations
The evaluation revealed several limitations in using smaller general-purpose language models for financial forecasting and risk analysis.

Limited numerical forecasting accuracy: Both FLAN-T5 models struggled to generate reliable financial predictions. FLAN-T5 Small produced invalid zero-price forecasts, while FLAN-T5 Base generated a negative SPY price. Neither model outperformed the naive baseline in the numerical experiments.

Weak financial reasoning: The models often produced vague statements without explaining economic relationships. FLAN-T5 Small frequently repeated instructions, while FLAN-T5 Base occasionally generated irrelevant or repetitive responses.

Inadequate risk mitigation: Although FLAN-T5 Base identified financial risks more effectively, it struggled to recommend practical mitigation strategies. This limits its usefulness for financial decision-making.

Sensitivity to prompts: Structured prompts did not consistently improve response quality. In several cases, optimization reduced performance, demonstrating that detailed instructions cannot compensate for limited model capabilities.

Evaluation constraints: The experiments used a limited number of scenarios and only one historical prediction period per financial variable. Both models belong to the same architecture family, and their pretrained knowledge may include information about the evaluation period. Therefore, the findings cannot be generalized to all LLMs or future financial conditions.

## 8. Ethical Considerations
Using LLMs in financial applications introduces several ethical concerns that must be addressed before deployment.

Financial misinformation: Incorrect predictions or unsupported investment recommendations may lead to financial losses. Model-generated outputs should be validated against reliable financial information.

Bias and fairness: Training data may contain biases that affect how models evaluate financial risks or economic conditions. Financial AI systems should be tested across different market environments and relevant customer groups to identify potentially unfair outcomes.

Transparency and explainability: Financial forecasts should include assumptions, supporting evidence, uncertainty, and limitations. Users should understand that AI-generated predictions are estimates rather than guaranteed outcomes.

Data privacy and security: Financial institutions must protect confidential customer information. Sensitive financial records should not be exposed through unsecured prompts, model outputs, or logging systems.

Human oversight and accountability: Financial professionals should review AI-generated forecasts and risk assessments before making important decisions. Organizations should maintain clear responsibility for how AI recommendations are used.

These considerations highlight the importance of responsible AI practices when applying language models to financial decision-making.

## 9. Recommendations
Based on the experimental findings, several improvements are recommended for future financial AI systems.

Use domain-specific models: Evaluate financial language models trained or fine-tuned on relevant financial datasets to improve task-specific understanding.

Combine LLMs with statistical forecasting: Use established statistical or machine learning methods for numerical predictions, while applying LLMs to interpretation and explanation.

Improve data grounding: Provide verified historical financial observations and relevant economic indicators to reduce unsupported claims.

Implement automated validation: Reject impossible predictions, such as nonpositive ETF prices, and check numerical consistency before presenting results.

Evaluate across multiple time periods: Use rolling historical evaluations and appropriate forecasting metrics to measure performance under different market conditions.

Improve risk assessment frameworks: Use structured risk categories, documented mitigation procedures, and independent validation to support more consistent recommendations.

Maintain human oversight: Require appropriate professional review, particularly when outputs could influence financial decisions or customer outcomes.

These recommendations are based on observed weaknesses and proposed improvements. Their effectiveness would need to be tested through additional experiments.

## 10. Conclusion
This project evaluated Google FLAN-T5 Small and FLAN-T5 Base for financial forecasting and risk analysis using historical SPY, GLD, and US CPI data. The models were tested on numerical forecasting, financial reasoning, risk identification, and mitigation strategies.

The results showed that neither model consistently produced reliable numerical financial forecasts. A simple naive baseline outperformed both models in the stock market, commodity price, and inflation experiments. FLAN-T5 Base performed better at identifying financial risks, while FLAN-T5 Small achieved a slightly higher score in the original qualitative forecasting evaluation.

Prompt optimization produced mixed results. FLAN-T5 Base showed some improvement in qualitative forecasting responses, but both models continued to struggle with detailed reasoning and practical risk mitigation.

Overall, the findings suggest that these smaller LLMs should not be relied upon independently for financial forecasting or risk management. Combining language models with validated financial data, numerical forecasting methods, structured evaluation, and human oversight may provide a more reliable approach for future financial applications.