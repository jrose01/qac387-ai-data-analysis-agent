# Choosing Simple Models

## RULES

Always check the tool registry to choose an appropriate tool first. If an appropriate tool exists, use that tool. Use code generation ONLY when an appropriate tool does not exist in th tool registry.

## Descriptive analysis

Choose summaries and plots from the tool registry when the goal is exploration rather than formal modeling.

## Multiple linear regression
Choose this from the tool registry when:
- the outcome is numeric
- the relationship is approximately linear
- predictors may be numeric or categorical

## When not to use linear regression
Avoid it when:
- the outcome is binary
- the outcome is categorical
- the outcome is a count with strong skew or zero inflation
- observations are clearly dependent in ways the model ignores

## Practical rule
A simple model is often best for a first pass. The agent should not jump to regression before understanding the data.
