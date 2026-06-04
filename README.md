\# Credit Risk Model for Bati Bank



\## Credit Scoring Business Understanding



\### Basel II Influence

Basel II requires financial institutions to have robust, documented, and interpretable risk measurement systems. In the context of credit scoring, this means:

\- The model must produce not just a prediction but a clear rationale for that prediction.

\- Documentation must include variable selection, transformation, and validation steps.

\- The model should be auditable and explainable to regulators.



Consequently, I will favour models that provide feature importance or coefficient interpretability (e.g., Logistic Regression with Weight of Evidence) and will use tools like SHAP or LIME if I choose a more complex model.



\### Necessity and Risks of a Proxy Variable

The dataset contains no direct "default" flag. To build a supervised model, I must create a proxy target variable. I will use RFM (Recency, Frequency, Monetary) clustering to label customers as high-risk (least engaged) vs low-risk (engaged).



\*\*Business risks of proxy‑based prediction:\*\*

\- The proxy may not perfectly align with true default behaviour (e.g., a low‑spending customer might still repay a loan).

\- There is a risk of \*\*misclassification\*\* – labelling a good customer as high‑risk could deny them credit, while labelling a bad customer as low‑risk could approve a loan that defaults.

\- These risks must be communicated to the risk team, and the model should be monitored and recalibrated over time.



\### Trade‑off: Interpretability vs Performance

| Model | Interpretability | Performance | Suitability for regulated finance |

|-------|----------------|-------------|-------------------------------------|

| Logistic Regression + WoE | High – coefficients directly show impact of each feature | Moderate – assumes linear relationships | Excellent – easy to explain to regulators |

| Gradient Boosting (XGBoost) | Low – complex interactions, black‑box | High – can capture non‑linear patterns | Acceptable if accompanied by SHAP/LIME explanations |



Given Basel II’s emphasis on interpretability, I will start with a \*\*Logistic Regression\*\* model and only consider more complex models if they provide significantly better discrimination (ROC‑AUC) and can be explained using SHAP.

